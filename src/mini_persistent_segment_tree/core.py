"""Path-copying segment tree for point updates and range queries.

The data structure is a complete binary segment tree.  Each update performs a
*path copy*: the leaf and every ancestor up to the root are duplicated, while
unchanged subtrees are shared between versions.  This yields O(log n) time and
O(log n) extra nodes per update, with full persistence of every version.

The public surface is a single class, :class:`PersistentSegmentTreeArray`, that
behaves like a read-only array with an additional ``update`` method that returns
a *new* version rather than mutating in place.  Versions are referenced by small
integer ids starting at ``0`` for the initial state.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence


class VersionNotFound(KeyError):
    """Raised when a version id supplied to a query or update does not exist."""


class PersistentSegmentTreeArray:
    """A persistent array backed by a path-copying segment tree.

    The tree supports two operations:

    * ``query(version, lo, hi)`` — sum of ``array[lo:hi]`` at ``version``.
    * ``update(version, index, value)`` — return a new version whose
      ``array[index]`` is ``value``.

    Parameters
    ----------
    values:
        Initial sequence of integers.  Length is fixed for the life of the
        object; updates only change element values, never the array length.
    combine:
        Binary associative function used to merge child sums into parent sums.
        Defaults to :func:`operator.add`.  Any callable returning the same
        *type* for every pair of inputs works, but the result of ``combine``
        must be a value suitable to re-merge with itself — partial application
        semantics (like ``operator.concat``) will not behave intuitively.
    identity:
        Neutral element for ``combine`` such that ``combine(identity, x) == x``
        and ``combine(x, identity) == x``.  Required so that out-of-range query
        ranges produce a deterministic, safe value rather than raising.

    Notes
    -----
    The tree is built over a power-of-two sized canvas padded with ``identity``.
    Padding is invisible to the caller — queries are clamped to ``[0, len)`` and
    the padding never appears in results — but it keeps the shape invariants of
    the tree simple and uniform.
    """

    __slots__ = ("_n", "_size", "_identity", "_combine", "_roots", "_leaves")

    def __init__(
        self,
        values: Sequence[int],
        *,
        combine: Callable[[int, int], int] = lambda a, b: a + b,
        identity: int = 0,
    ) -> None:
        if not isinstance(values, Sequence):
            raise TypeError("values must be a sequence")
        n = len(values)
        if n < 0:
            raise ValueError("length cannot be negative")

        # Compute the next power of two >= n.  Using a power of two makes every
        # internal node have exactly two children, which keeps the build and
        # path-copy logic branch-free with respect to tree shape.
        size = 1
        while size < n:
            size *= 2
        # Guard against a truly empty canvas so the root always exists; the
        # caller never sees this padding because n == 0 collapses every query
        # to the identity before any tree traversal occurs.
        if size == 0:
            size = 1

        self._n = n
        self._size = size
        self._identity = identity
        self._combine = combine
        self._leaves: List[int] = [identity] * size
        for i, v in enumerate(values):
            self._leaves[i] = v

        root = self._build(0, size)
        self._roots: List[Optional[object]] = [root]

    # ------------------------------------------------------------------ build

    def _build(self, lo: int, hi: int) -> object:
        """Recursively build a node covering ``[lo, hi)`` from ``self._leaves``."""
        if hi - lo == 1:
            return [self._leaves[lo], None, None]
        mid = (lo + hi) >> 1
        left = self._build(lo, mid)
        right = self._build(mid, hi)
        # Node layout: ``[value, left_child, right_child]``.  We use a plain
        # list rather than a dataclass to minimise per-node overhead; a segment
        # tree over one million elements allocates ~2M nodes, so avoiding an
        # extra ``__dict__`` per node matters here.
        return [self._combine(left[0], right[0]), left, right]

    # -------------------------------------------------------------- accessors

    @property
    def length(self) -> int:
        """Length of the underlying array (fixed at construction time)."""
        return self._n

    @property
    def version_count(self) -> int:
        """Number of versions currently available, including the initial one."""
        return len(self._roots)

    # ---------------------------------------------------------------- queries

    def _require_version(self, version: int) -> object:
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("version must be an int")
        if version < 0 or version >= len(self._roots):
            raise VersionNotFound(version)
        root = self._roots[version]
        assert root is not None
        return root

    def query(self, version: int, lo: int = 0, hi: Optional[int] = None) -> int:
        """Return the combined value over ``array[lo:hi]`` at ``version``.

        Parameters
        ----------
        version:
            Version id obtained from construction (``0``) or a prior
            :meth:`update` call.
        lo, hi:
            Half-open range, matching Python slice semantics.  ``hi`` defaults
            to the array length.  An empty or fully out-of-range range returns
            ``identity``.

        Raises
        ------
        VersionNotFound:
            If ``version`` is not a known version id.
        """
        root = self._require_version(version)
        if hi is None:
            hi = self._n
        if not isinstance(lo, int) or isinstance(lo, bool):
            raise TypeError("lo must be an int")
        if not isinstance(hi, int) or isinstance(hi, bool):
            raise TypeError("hi must be an int")
        # Clamp to [0, n] so that negative or oversized indices behave like
        # Python slices: they never reach the padded identity leaves.
        lo = max(0, lo)
        hi = min(self._n, hi)
        if lo >= hi:
            return self._identity
        return self._query(root, 0, self._size, lo, hi)

    def _query(self, node: object, lo: int, hi: int, qlo: int, qhi: int) -> int:
        if qlo <= lo and hi <= qhi:
            return node[0]
        mid = (lo + hi) >> 1
        if qhi <= mid:
            return self._query(node[1], lo, mid, qlo, qhi)
        if qlo >= mid:
            return self._query(node[2], mid, hi, qlo, qhi)
        return self._combine(
            self._query(node[1], lo, mid, qlo, qhi),
            self._query(node[2], mid, hi, qlo, qhi),
        )

    # ---------------------------------------------------------------- updates

    def update(self, version: int, index: int, value: int) -> int:
        """Return a new version id with ``array[index]`` set to ``value``.

        The supplied ``version`` is left untouched; the new version is appended
        to the internal version list and its id is returned.

        Raises
        ------
        VersionNotFound:
            If ``version`` is not a known version id.
        IndexError:
            If ``index`` is outside ``[0, len)``.
        """
        root = self._require_version(version)
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an int")
        if index < 0 or index >= self._n:
            raise IndexError(index)
        new_root = self._update(root, 0, self._size, index, value)
        self._roots.append(new_root)
        return len(self._roots) - 1

    def _update(
        self, node: object, lo: int, hi: int, index: int, value: int
    ) -> object:
        if hi - lo == 1:
            return [value, None, None]
        mid = (lo + hi) >> 1
        # Copy only the path from root to the updated leaf; the sibling subtree
        # pointer is reused verbatim, which is what gives persistence O(log n)
        # memory per update rather than O(n).
        if index < mid:
            new_left = self._update(node[1], lo, mid, index, value)
            return [self._combine(new_left[0], node[2][0]), new_left, node[2]]
        new_right = self._update(node[2], mid, hi, index, value)
        return [self._combine(node[1][0], new_right[0]), node[1], new_right]
