# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compare opt-in native MIMO Kineto traces without treating overlaps as wall time.

Run with ``--static-dir PATH --dynamic-dir PATH --output-dir PATH``. Only CPU
launch correlation/External IDs associate kernels with scopes. Kernel timestamps
are used for device interval unions, never for guessing their originating module.
"""

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _events(path):
    """Stream when ijson is installed; otherwise load only this one trace."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        import ijson
    except ImportError:
        ijson = None
    with opener(path, "rb") as stream:
        if ijson is not None:
            yield from ijson.items(stream, "traceEvents.item", use_float=True)
        else:
            print(f"{path.name}: stdlib JSON, one full trace in RAM", file=sys.stderr, flush=True)
            yield from json.load(stream)["traceEvents"]


def _union_us(intervals):
    """Length of a union; timestamps and durations are Chrome microseconds."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _identity(args, *names):
    for name in names:
        if name in args:
            return str(args[name])
    return None


def _scope_name(name):
    label, separator, fields = name.removeprefix("mimo::").partition("[")
    context = {}
    if separator:
        for field in fields.rstrip("]").split(","):
            key, equals, value = field.partition("=")
            if equals:
                context[key] = value
    return label, context


def _category(name, labels):
    """Exclusive, conservative buckets; HybridEP includes its local bookkeeping."""
    lower, context = name.lower(), " ".join(labels).lower()
    if "nccl" in lower:
        return "nccl"
    if any(word in lower for word in ("hybridep", "hybrid_ep", "nvshmem", "internode")):
        return "hybridep_related"
    if ".hybridep." in context:
        return "hybridep_related"
    if any(word in context for word in ("gateddeltanet", "gateddeltaproduct")) or any(
        word in lower for word in ("gated_delta", "chunk_fwd", "chunk_bwd", "causal_conv")
    ):
        return "gdn"
    if "selfattention:" in context or any(
        word in lower for word in ("flash_fwd", "flash_bwd", "flashattn", "fmha", "cudnn_attn")
    ):
        return "attention"
    if "shared_expert" in context:
        return "moe_shared_experts"
    if "expert" in context and "moe" in context:
        return "moe_experts"
    if "router" in context:
        return "moe_router"
    if ".dispatcher." in context:
        return "moe_dispatcher_other"
    if "moelayer:" in context:
        return "moe_other"
    if "feature_" in context and "transport" in context:
        return "vision_transport_other"
    if "encoder_forward" in context:
        return "encoder_other"
    if "gemm" in lower or "cutlass" in lower:
        return "gemm_unassigned_module"
    return "other"


def _add(stats, key, duration):
    item = stats.setdefault(key, {"calls": 0, "duration_sum_us": 0.0})
    item["calls"] += 1
    item["duration_sum_us"] += duration


def _top(stats, limit):
    return [
        {"name": name, **values}
        for name, values in sorted(
            stats.items(), key=lambda item: item[1]["duration_sum_us"], reverse=True
        )[:limit]
    ]


def _cpu_activity(host, scopes, memory=False):
    """Summarize CPU API work with exclusive scope labels and per-thread unions."""
    apis, by_stage, by_context, threads = {}, {}, {}, {}
    intervals = defaultdict(list)
    stage_names = {
        "prepare_including_encoder_and_auxiliary",
        "encoder_forward",
        "decoder_forward_all_passes",
        "feature_forward_transport",
        "feature_backward_transport",
        "finalize_including_encoder_backward_and_reduction",
        "optimizer",
    }
    allocation_prefixes = (
        "cuMemCreate",
        "cuMemRelease",
        "cuMemMap",
        "cuMemUnmap",
        "cuMemAddressReserve",
        "cuMemAddressFree",
        "cuMemSetAccess",
        "cuMemAlloc",
        "cuMemFree",
        "cudaMalloc",
        "cudaFree",
    )
    for event in host:
        name = event["name"]
        if memory:
            selected = event["category"] in ("cuda_runtime", "cuda_driver") and name.startswith(
                allocation_prefixes
            )
        else:
            selected = any(
                word in name
                for word in ("Synchronize", "StreamWaitEvent", "aten::item", "_local_scalar_dense")
            )
        if not selected:
            continue
        active = [scopes[sid] for sid in event.get("scopes", ())]
        specific = [scope for scope in active if scope["label"] != "native_train_step"]
        label = (
            specific[-1]["label"]
            if specific
            else ("outside_wrapped_stage" if active else "unscoped_cpu_thread")
        )
        stage = next(
            (scope["label"] for scope in reversed(specific) if scope["label"] in stage_names),
            "outside_wrapped_stage" if active else "unscoped_cpu_thread",
        )
        # Root context was captured at step entry and is stale during backward/round gaps.
        trusted = [
            scope
            for scope in specific
            if ":" in scope["label"]
            or ".dispatcher." in scope["label"]
            or ".hybridep." in scope["label"]
            or scope["label"].startswith(("decoder_forward", "feature_"))
        ]
        context = trusted[-1]["context"] if trusted else {}
        phase = context.get("phase", stage if stage in stage_names else "unknown")
        round_id, cp = context.get("round"), context.get("cp")
        key = (name, label, stage, phase, round_id, cp)
        duration = event["duration"]
        _add(apis, name, duration)
        _add(by_stage, (stage, name), duration)
        _add(by_context, key, duration)
        _add(threads, event["thread"], duration)
        intervals[event["thread"]].append((event["start"], event["start"] + duration))
    return dict(
        by_api=apis,
        by_stage_and_api=[
            dict(stage=stage, api=api, **stats) for (stage, api), stats in by_stage.items()
        ],
        by_scope_phase_round=[
            dict(api=api, scope=label, stage=stage, phase=phase, round=round_id, cp=cp, **stats)
            for (api, label, stage, phase, round_id, cp), stats in by_context.items()
        ],
        per_thread=[
            dict(pid=pid, tid=tid, interval_union_us=_union_us(intervals[(pid, tid)]), **stats)
            for (pid, tid), stats in threads.items()
        ],
        scope_policy="Exclusive nearest CPU scope; root-only context is unknown, not prepare.",
        warning="API sums can overlap/nest; per-thread unions are not additive across threads.",
    )


def analyze_trace(path, top=40):
    """Read a single trace, retaining only compact events needed for association."""
    scopes, kernels, host, synchronizations = [], [], [], {}
    metadata_events, categories_seen = [], defaultdict(int)
    for event in _events(path):
        if event.get("ph") == "M":
            metadata_events.append(event)
            continue
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name, category = event.get("name", ""), event.get("cat", "")
        categories_seen[category] += 1
        start, duration = float(event["ts"]), float(event["dur"])
        args = event.get("args", {})
        thread = (str(event.get("pid")), str(event.get("tid")))
        if name.startswith("mimo::") and category in ("user_annotation", "cpu_op"):
            label, context = _scope_name(name)
            scopes.append(
                dict(
                    start=start,
                    stop=start + duration,
                    thread=thread,
                    label=label,
                    context=context,
                    kernel_calls=0,
                    kernel_duration_sum_us=0.0,
                )
            )
        if category == "kernel":
            kernels.append(
                dict(
                    name=name,
                    start=start,
                    duration=duration,
                    device=str(args.get("device", event.get("pid"))),
                    correlation=_identity(args, "correlation", "Correlation ID"),
                    external=_identity(args, "External id", "External ID"),
                )
            )
            continue
        if category not in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"):
            continue
        external = _identity(args, "External id", "External ID")
        correlation = _identity(args, "correlation", "Correlation ID")
        if external is not None or correlation is not None:
            host.append(
                dict(
                    name=name,
                    start=start,
                    duration=duration,
                    thread=thread,
                    external=external,
                    correlation=correlation,
                    category=category,
                )
            )
        if any(
            word in name
            for word in ("Synchronize", "StreamWaitEvent", "aten::item", "_local_scalar_dense")
        ):
            _add(synchronizations, name, duration)

    # Scope containment is evaluated ONLY at host launch/op timestamps, per CPU thread.
    # Scopes are nested record_function regions. Sort longer equal-start scopes first.
    by_thread = defaultdict(list)
    for index, scope in enumerate(scopes):
        by_thread[scope["thread"]].append((scope["start"], 0, -scope["stop"], index))
    for index, event in enumerate(host):
        by_thread[event["thread"]].append((event["start"], 1, 0, index))
    for entries in by_thread.values():
        active = []
        for timestamp, kind, _, index in sorted(entries):
            active = [sid for sid in active if scopes[sid]["stop"] > timestamp]
            if kind == 0:
                active.append(index)
            else:
                host[index]["scopes"] = tuple(active)

    external_ops, launches = {}, {}
    for event in host:
        if event["category"] in ("cuda_runtime", "cuda_driver"):
            if event["correlation"] is not None:
                launches[event["correlation"]] = event
        elif event["external"] is not None:
            old = external_ops.get(event["external"])
            if old is None or event["duration"] < old["duration"]:
                external_ops[event["external"]] = event

    kinds, names, launch_ops, attribution = {}, {}, {}, {}
    unassigned_module_us = 0.0
    intervals, comm_intervals, compute_intervals = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for kernel in kernels:
        launch = launches.get(kernel["correlation"])
        external = (launch["external"] if launch is not None else None) or kernel["external"]
        op = external_ops.get(external)
        owner = op if op is not None and op.get("scopes") else launch
        scope_ids = owner.get("scopes", ()) if owner is not None else ()
        labels = [scopes[sid]["label"] for sid in scope_ids]
        category = _category(kernel["name"], labels)
        duration = kernel["duration"]
        if not any(
            ":" in label or ".dispatcher." in label or ".hybridep." in label for label in labels
        ):
            unassigned_module_us += duration
        _add(kinds, category, duration)
        _add(names, kernel["name"], duration)
        _add(launch_ops, op["name"] if op is not None else "unassigned", duration)
        association = "scope_via_external_id" if op is owner and scope_ids else "unassigned"
        if owner is launch and scope_ids:
            association = "scope_via_cpu_launch"
        _add(attribution, association, duration)
        for sid in scope_ids:
            scopes[sid]["kernel_calls"] += 1
            scopes[sid]["kernel_duration_sum_us"] += duration
        interval = (kernel["start"], kernel["start"] + duration)
        device = kernel["device"]
        intervals[device].append(interval)
        if category in ("nccl", "hybridep_related"):
            comm_intervals[device].append(interval)
        else:
            compute_intervals[device].append(interval)

    device_totals = {}
    for device, values in intervals.items():
        busy = _union_us(values)
        comm = _union_us(comm_intervals[device])
        compute = _union_us(compute_intervals[device])
        span = max(stop for _, stop in values) - min(start for start, _ in values)
        device_totals[device] = dict(
            kernel_duration_sum_us=sum(stop - start for start, stop in values),
            kernel_union_us=busy,
            kernel_span_us=span,
            no_kernel_in_span_us=max(0.0, span - busy),
            communication_related_union_us=comm,
            remaining_kernel_union_us=compute,
            overlap_communication_and_remaining_us=max(0.0, comm + compute - busy),
        )

    scope_totals = {}
    scope_rows = []
    for scope in scopes:
        duration = scope["stop"] - scope["start"]
        _add(scope_totals, scope["label"], duration)
        scope_rows.append(
            dict(
                label=scope["label"],
                context=scope["context"],
                start_us=scope["start"],
                cpu_duration_us=duration,
                kernel_calls=scope["kernel_calls"],
                kernel_duration_sum_us=scope["kernel_duration_sum_us"],
            )
        )
    total_kernel_us = sum(k["duration"] for k in kernels)
    unassigned = attribution.get("unassigned", {"calls": 0, "duration_sum_us": 0.0})
    stem = path.name.removesuffix(".gz").removesuffix(".trace.json")
    artifacts = []
    metadata = None
    for candidate in [path, *sorted(path.parent.glob(f"{stem}.*"))]:
        if any(item["path"] == str(candidate.resolve()) for item in artifacts):
            continue
        artifacts.append(
            dict(
                path=str(candidate.resolve()),
                bytes=candidate.stat().st_size,
                sha256=_sha256(candidate),
            )
        )
        if candidate.name.endswith(".metadata.json"):
            metadata = json.loads(candidate.read_text())
    return dict(
        trace=str(path.resolve()),
        parser_sha256=_sha256(Path(__file__)),
        artifacts=artifacts,
        metadata=metadata,
        chrome_metadata_events=metadata_events,
        chrome_event_category_counts=dict(categories_seen),
        kernel_count=len(kernels),
        kernel_duration_sum_us=total_kernel_us,
        devices=device_totals,
        kernel_categories=kinds,
        top_kernels=_top(names, top),
        top_launch_ops=_top(launch_ops, top),
        scope_attribution=attribution,
        unassigned_kernel_count_fraction=unassigned["calls"] / len(kernels) if kernels else None,
        unassigned_kernel_time_fraction=(
            unassigned["duration_sum_us"] / total_kernel_us if total_kernel_us else None
        ),
        unassigned_module_kernel_time_fraction=(
            unassigned_module_us / total_kernel_us if total_kernel_us else None
        ),
        cpu_synchronizing_calls=synchronizations,
        cpu_allocation_apis=_cpu_activity(host, scopes, memory=True),
        cpu_synchronization_apis=_cpu_activity(host, scopes),
        cpu_scope_totals=scope_totals,
        scopes=scope_rows,
    )


def main():
    """Produce per-trace evidence and a descriptive static/dynamic comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path)
    parser.add_argument("--dynamic-dir", type=Path)
    parser.add_argument("--trace", type=Path, help="Analyze one trace instead of two directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.trace:
        if args.static_dir or args.dynamic_dir:
            parser.error("--trace cannot be combined with directory inputs")
        result = analyze_trace(args.trace, args.top)
        output = args.output_dir / f"{args.trace.name.removesuffix('.gz')}.analysis.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(output)
        return
    if not args.static_dir or not args.dynamic_dir:
        parser.error("Provide --trace or both --static-dir and --dynamic-dir")
    comparison = {}
    for mode, directory in (("static", args.static_dir), ("dynamic", args.dynamic_dir)):
        paths = sorted(
            set(directory.rglob("*.trace.json.gz")) | set(directory.rglob("*.trace.json"))
        )
        if not paths:
            raise ValueError(f"No traces found in {directory}")
        results = []
        for path in paths:
            print(f"Analyzing {mode}: {path}", file=sys.stderr, flush=True)
            result = analyze_trace(path, args.top)
            output = args.output_dir / f"{mode}-{path.name.removesuffix('.gz')}.analysis.json"
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            results.append(
                dict(
                    report=str(output.resolve()),
                    report_sha256=_sha256(output),
                    rank=(result["metadata"] or {}).get("rank"),
                    source_step=(result["metadata"] or {}).get("source_step"),
                    kernel_count=result["kernel_count"],
                    kernel_duration_sum_us=result["kernel_duration_sum_us"],
                    devices=result["devices"],
                    kernel_categories=result["kernel_categories"],
                    unassigned_kernel_time_fraction=result["unassigned_kernel_time_fraction"],
                    unassigned_module_kernel_time_fraction=result[
                        "unassigned_module_kernel_time_fraction"
                    ],
                )
            )
        comparison[mode] = dict(
            traces=results,
            mean_per_trace_kernel_duration_sum_us=statistics.mean(
                row["kernel_duration_sum_us"] for row in results
            ),
        )
    report = dict(
        schema_version=1,
        units="microseconds unless explicitly named otherwise",
        parser_sha256=_sha256(Path(__file__)),
        comparison=comparison,
        limitations=[
            "Profiled timings are not an uninstrumented throughput benchmark.",
            "Kernel sums, CPU scopes and per-rank timings cannot be added to reconstruct wall time.",
            "Scope kernel totals are inclusive and nested; backward may remain unattributed.",
            "Kernel association uses CPU launch/External IDs, not GPU/CPU temporal overlap.",
            "HybridEP-related includes local preprocessing; NCCL duration can include peer waits.",
            "GPU busy is a kernel-interval union, not SM utilization; memcpy is not included.",
            "Cross-rank absolute timestamps are not assumed synchronized.",
            "Name-based categories are heuristics; inspect top kernels and unassigned fractions.",
        ],
    )
    output = args.output_dir / "profile-comparison.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
