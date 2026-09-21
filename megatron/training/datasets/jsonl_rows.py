# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Indexed JSONL rows without Arrow coercion of heterogeneous tool arguments."""

import json
import os
from array import array


class JsonlRows:
    """Index byte offsets in memory, and open a separate stream per worker.

    This is an IO index, not tokenization or offline sequence packing. External
    dataset-specific index formats are deliberately not interpreted here.
    """

    def __init__(self, path):
        self.path = str(path)
        self.offsets = array("Q")
        self._file = None
        self._pid = None
        self.column_names = []
        with open(path, "rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    if not self.offsets:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("JSONL rows must be objects")
                        self.column_names = list(row)
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"Empty JSONL: {path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            if self._file is not None:
                self._file.close()
            self._file = open(self.path, "rb")
            self._pid = os.getpid()
        self._file.seek(self.offsets[index])
        row = json.loads(self._file.readline())
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {index} must be an object: {self.path}")
        return row

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
