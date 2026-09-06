"""pfsp: a reproducible Permutation Flow-Shop Scheduling Problem toolkit.

This package is the single source of truth for the algorithmic logic of the
project (instance I/O, makespan computation, NEH heuristic, results framework
and reproducibility utilities). Notebooks and the command-line entry point are
clients that import this package; they do not reimplement core logic.
"""


class PfspError(Exception):
    """Base class for all errors raised by the ``pfsp`` package."""


class InstanceFormatError(PfspError):
    """Raised when an instance file cannot be parsed into a valid ``Instance``.

    The message describes the cause (e.g. a dimension mismatch, a negative
    processing time or a malformed line) so the caller can locate the problem.
    Readers never return a partial or invalid instance when this is raised.
    """


class UnknownInstanceFormatError(PfspError):
    """Raised when the benchmark format of a file cannot be determined."""


__all__ = ["PfspError", "InstanceFormatError", "UnknownInstanceFormatError"]
