# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Optional full, local-token MoE boundary observations, without communication.

The enclosing execution diagnostic owns round/phase and canonical CP metadata.
Each boundary keeps independent invocation records: selective recompute can
repeat a router or norm without repeating the containing MoE module. Later
invocations are deliberately labelled ``recompute_or_repeat``, not inferred
from grad mode. Captured cotangents retain the native training scale.
"""

from functools import wraps

import torch


class MoEBoundaryDiagnostics:
    """Observe selected MoE layers; remove hooks and restore methods on close."""

    def __init__(self, layers, indices, current):
        self.current = current
        self.handles = []
        self.methods = []
        self.installed = {}
        indices = tuple(dict.fromkeys(indices))
        # Validate everything before modifying any live module.
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(layers):
                raise ValueError(f'Invalid MoE diagnostic layer index: {index}')
            layer = layers[index]
            if not hasattr(layer, 'pre_mlp_layernorm') or not hasattr(layer.mlp, 'router'):
                raise ValueError(f'Diagnostic layer {index} lacks MoE/norm boundaries')
        try:
            for index in indices:
                self._install(index, layers[index])
        except Exception:
            self.close()
            raise

    def _observe(self, index, name, tensor):
        current = self.current()
        if current is None or current['phase'] not in ('statistics', 'training'):
            return
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            raise ValueError(f'MoE boundary {name} did not produce a token tensor')
        if tensor.numel() // tensor.shape[-1] != current['local_tokens']:
            raise ValueError(f'MoE boundary {name} has unexpected token layout: {tensor.shape}')
        entry = current.setdefault('moe_layers', {}).setdefault(
            str(index),
            dict(schema_version=1, installed_boundaries=self.installed[index], boundaries={}),
        )
        records = entry['boundaries'].setdefault(name, [])
        record = dict(
            invocation=len(records),
            kind='original' if not records else 'recompute_or_repeat',
            grad_enabled=torch.is_grad_enabled(),
            requires_grad=tensor.requires_grad,
            # copy=True also isolates CPU tests from subsequent in-place mutations.
            value=tensor.detach().to(device='cpu', copy=True),
        )
        records.append(record)
        if current['phase'] == 'training' and tensor.requires_grad:

            def save_gradient(gradient):
                record['gradient'] = gradient.detach().to(device='cpu', copy=True)

            self.handles.append(tensor.register_hook(save_gradient))

    def _module(self, index, module, input_name=None, output_names=()):
        if input_name is not None:
            self.installed[index].append(input_name)

            def before(module, args, kwargs):
                value = args[0] if args else kwargs.get('hidden_states', kwargs.get('input'))
                self._observe(index, input_name, value)

            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        if output_names:
            self.installed[index].extend(output_names)

            def after(module, args, output):
                values = output if isinstance(output, tuple) else (output,)
                for name, value in zip(output_names, values):
                    self._observe(index, name, value)
                if len(values) < len(output_names):
                    raise ValueError(f'MoE boundary output missing: {output_names}')

            self.handles.append(module.register_forward_hook(after))

    def _method_output(self, index, owner, method_name, boundary):
        """Passthrough wrapper; restore the original instance/class lookup on close."""
        original = getattr(owner, method_name)
        had_instance_value = method_name in vars(owner)
        instance_value = vars(owner).get(method_name)

        @wraps(original)
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            self._observe(index, boundary, result)
            return result

        self.methods.append((owner, method_name, had_instance_value, instance_value))
        setattr(owner, method_name, observed)
        self.installed[index].append(boundary)

    def _install(self, index, layer):
        self.installed[index] = []
        self._module(index, layer.pre_mlp_layernorm, 'pre_mlp_norm_input', ('pre_mlp_norm_output',))
        self._module(index, layer.mlp, 'moe_input', ('moe_output',))
        self._module(index, layer.mlp.router, 'router_input', ('router_probs', 'routing_map'))
        if callable(getattr(layer.mlp.router, 'gating', None)):
            # These logits precede optional force-balanced/biased overrides in
            # Router.forward. Do not mislabel them as the post-override logits.
            self._method_output(index, layer.mlp.router, 'gating', 'router_logits')
        if getattr(layer.mlp, 'shared_experts', None) is not None:
            self._module(index, layer.mlp.shared_experts, output_names=('shared_output',))
        dispatcher = getattr(layer.mlp, 'token_dispatcher', None)
        if callable(getattr(dispatcher, 'combine_postprocess', None)) and not getattr(
            layer.mlp, 'shared_expert_overlap', False
        ):
            # This is back in local token order and precedes shared addition.
            # With overlap enabled, combine_postprocess itself may add shared.
            self._method_output(index, dispatcher, 'combine_postprocess', 'routed_output')

    def close(self):
        """Idempotently remove observations without touching model state or math."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, name, had_instance_value, value in reversed(self.methods):
            if had_instance_value:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.methods.clear()
