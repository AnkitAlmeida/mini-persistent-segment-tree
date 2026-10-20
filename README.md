# mini-persistent-segment-tree

A segment tree that preserves prior versions after updates, enabling historical
range queries on point-in-time snapshots of an array.

## Usage

```python
from mini_persistent_segment_tree import PersistentSegmentTreeArray

t = PersistentSegmentTreeArray([1, 2, 3, 4, 5])
print(t.query(0, 0, 5))   # 15, sum of the initial array

v1 = t.update(0, 2, 30)   # set index 2 to 30, returns new version id
print(t.query(0, 0, 5))   # 15, original version unchanged
print(t.query(v1, 0, 5))  # 40, updated version
```

The exported names are `PersistentSegmentTreeArray` and `VersionNotFound`. The
class exposes `query(version, lo, hi)`, `update(version, index, value)`, and
the read-only properties `length` and `version_count`.

## Why this exists

Some problems need to query an array as it existed at many different points in
time — after each of a series of updates. A naive approach copies the array on
every update, costing O(n) time and space per snapshot. This library instead
stores the array in a segment tree and performs a path copy on each update: only
the O(log n) nodes from root to the changed leaf are duplicated, while the
unchanged subtrees are shared between versions. The trade-off is that lookups
and updates are O(log n) rather than O(1), but every historical version stays
queryable with no per-version copy cost.

## Edge cases

The array length is fixed at construction; updates only change element values,
never the size. Out-of-range query indices are clamped to `[0, len)` rather than
raising, matching Python slice semantics, and empty ranges return the combine
function's identity (default `0`). An empty initial array is supported but
rejects any update with `IndexError`. Version ids are small integers starting at
`0`; passing an unknown or negative id raises `VersionNotFound`.
