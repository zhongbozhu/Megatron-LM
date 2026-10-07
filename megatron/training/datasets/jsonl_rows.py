# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Map-style JSONL access without coercing nested records into an Arrow schema."""

import json
import operator
import os
from array import array


class JsonlRows:
    """Index nonblank rows by byte offset; keep one file handle per process.

    Files must remain immutable for the lifetime of the reader. Construction
    scans every physical line and stores offsets/line numbers, but does not
    materialize trajectories or cache tokenization. ``column_names`` describes
    only the first record and must not be used to infer optional row fields.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self.offsets = array("Q")
        self.line_numbers = array("Q")
        self.column_names = []
        self._file = None
        self._pid = None
        with open(self.path, "rb") as stream:
            self._signature = self._file_signature(stream)
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                if line.strip():
                    if not self.offsets:
                        self.column_names = list(self._decode(line, line_number))
                    self.offsets.append(offset)
                    self.line_numbers.append(line_number)
        if not self.offsets:
            raise ValueError(f"Empty JSONL: {self.path}")

    @staticmethod
    def _file_signature(stream):
        stat = os.fstat(stream.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _decode(self, line, line_number):
        location = f"{self.path}:{line_number}"
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"{location}: invalid UTF-8 JSON object") from None
        if not isinstance(row, dict):
            raise ValueError(f"{location}: JSONL row must be an object")
        return row

    def __len__(self):
        return len(self.offsets)

    def location(self, index):
        """Return a safe error location (path and one-based physical line)."""
        return f"{self.path}:{self.line_numbers[index]}"

    def __getitem__(self, index):
        index = operator.index(index)
        offset = self.offsets[index]  # Bounds checking precedes opening a handle.
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = open(self.path, "rb")
            self._pid = os.getpid()
        if self._file_signature(self._file) != self._signature:
            raise ValueError(
                f"{self.path}: JSONL changed after indexing; immutable input is required"
            )
        self._file.seek(offset)
        return self._decode(self._file.readline(), self.line_numbers[index])

    def close(self):
        """Release this process's file handle; a subsequent access reopens it."""
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
