"""mini-persistent-segment-tree: an array wrapper that preserves prior versions.

This package exposes :class:`PersistentSegmentTreeArray`, an array-like object
whose updates never mutate older versions.  Every update produces a new root
node; previous roots are kept alive so that a query can be executed against any
historical snapshot without copying the underlying data.
"""

from .core import PersistentSegmentTreeArray, VersionNotFound

__all__ = ["PersistentSegmentTreeArray", "VersionNotFound"]
