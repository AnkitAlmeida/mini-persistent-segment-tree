import unittest

from mini_persistent_segment_tree import PersistentSegmentTreeArray, VersionNotFound


class TestConstruction(unittest.TestCase):
    def test_empty_array(self):
        t = PersistentSegmentTreeArray([])
        self.assertEqual(t.length, 0)
        self.assertEqual(t.version_count, 1)
        self.assertEqual(t.query(0, 0, 0), 0)
        self.assertEqual(t.query(0), 0)
        with self.assertRaises(IndexError):
            t.update(0, 0, 7)

    def test_single_element(self):
        t = PersistentSegmentTreeArray([42])
        self.assertEqual(t.length, 1)
        self.assertEqual(t.query(0, 0, 1), 42)
        self.assertEqual(t.query(0), 42)

    def test_rejects_non_sequence(self):
        with self.assertRaises(TypeError):
            PersistentSegmentTreeArray(123)  # type: ignore[arg-type]


class TestQueries(unittest.TestCase):
    def test_basic_sum(self):
        t = PersistentSegmentTreeArray([1, 2, 3, 4, 5])
        self.assertEqual(t.query(0, 0, 5), 15)
        self.assertEqual(t.query(0, 1, 4), 9)
        self.assertEqual(t.query(0, 2, 3), 3)

    def test_non_power_of_two(self):
        t = PersistentSegmentTreeArray([10, 20, 30])
        self.assertEqual(t.query(0), 60)
        self.assertEqual(t.query(0, 0, 1), 10)
        self.assertEqual(t.query(0, 1, 3), 50)

    def test_default_range(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        self.assertEqual(t.query(0), 6)

    def test_out_of_range_clamps(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        self.assertEqual(t.query(0, -100, 100), 6)
        self.assertEqual(t.query(0, 2, 100), 3)
        self.assertEqual(t.query(0, -5, 0), 0)

    def test_empty_range_returns_identity(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        self.assertEqual(t.query(0, 2, 2), 0)
        self.assertEqual(t.query(0, 0, 0), 0)
        self.assertEqual(t.query(0, 5, 1), 0)

    def test_custom_combine_min(self):
        t = PersistentSegmentTreeArray([5, 2, 8, 1, 9], combine=min, identity=10 ** 18)
        self.assertEqual(t.query(0, 0, 5), 1)
        self.assertEqual(t.query(0, 0, 3), 2)
        self.assertEqual(t.query(0, 2, 5), 1)

    def test_custom_combine_max(self):
        t = PersistentSegmentTreeArray([5, 2, 8, 1, 9], combine=max, identity=-10 ** 18)
        self.assertEqual(t.query(0, 0, 5), 9)
        self.assertEqual(t.query(0, 0, 3), 8)
        self.assertEqual(t.query(0, 3, 5), 9)


class TestUpdates(unittest.TestCase):
    def test_update_returns_new_version(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        v1 = t.update(0, 1, 20)
        self.assertEqual(v1, 1)
        self.assertEqual(t.version_count, 2)
        self.assertEqual(t.query(0), 6)
        self.assertEqual(t.query(1), 24)
        self.assertEqual(t.query(0, 1, 2), 2)
        self.assertEqual(t.query(1, 1, 2), 20)

    def test_chained_updates_preserve_history(self):
        t = PersistentSegmentTreeArray([0, 0, 0])
        v1 = t.update(0, 0, 1)
        v2 = t.update(v1, 1, 2)
        v3 = t.update(v2, 2, 3)
        self.assertEqual(t.query(0), 0)
        self.assertEqual(t.query(v1), 1)
        self.assertEqual(t.query(v2), 3)
        self.assertEqual(t.query(v3), 6)

    def test_branching_versions(self):
        t = PersistentSegmentTreeArray([1, 1, 1])
        v_a = t.update(0, 0, 10)
        v_b = t.update(0, 0, 20)
        self.assertEqual(t.query(v_a, 0, 1), 10)
        self.assertEqual(t.query(v_b, 0, 1), 20)
        self.assertEqual(t.query(0, 0, 1), 1)
        self.assertEqual(t.query(v_a), 12)
        self.assertEqual(t.query(v_b), 22)

    def test_update_out_of_range_index(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        with self.assertRaises(IndexError):
            t.update(0, -1, 0)
        with self.assertRaises(IndexError):
            t.update(0, 3, 0)
        self.assertEqual(t.version_count, 1)

    def test_update_same_index_twice(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        v1 = t.update(0, 1, 100)
        v2 = t.update(v1, 1, 200)
        self.assertEqual(t.query(v1, 1, 2), 100)
        self.assertEqual(t.query(v2, 1, 2), 200)

    def test_update_with_custom_combine(self):
        t = PersistentSegmentTreeArray([5, 3, 8], combine=max, identity=-10 ** 18)
        v1 = t.update(0, 1, 10)
        self.assertEqual(t.query(0, 0, 3), 8)
        self.assertEqual(t.query(v1, 0, 3), 10)


class TestErrorHandling(unittest.TestCase):
    def test_unknown_version_raises(self):
        t = PersistentSegmentTreeArray([1, 2])
        with self.assertRaises(VersionNotFound):
            t.query(5)
        with self.assertRaises(VersionNotFound):
            t.query(-1)
        with self.assertRaises(VersionNotFound):
            t.update(99, 0, 0)

    def test_bool_version_rejected(self):
        t = PersistentSegmentTreeArray([1, 2])
        with self.assertRaises(TypeError):
            t.query(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            t.update(False, 0, 0)  # type: ignore[arg-type]

    def test_non_int_index_rejected(self):
        t = PersistentSegmentTreeArray([1, 2])
        with self.assertRaises(TypeError):
            t.update(0, "x", 0)  # type: ignore[arg-type]

    def test_non_int_bounds_rejected(self):
        t = PersistentSegmentTreeArray([1, 2])
        with self.assertRaises(TypeError):
            t.query(0, "x")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            t.query(0, 0, "x")  # type: ignore[arg-type]


class TestVersionMetadata(unittest.TestCase):
    def test_length_fixed(self):
        t = PersistentSegmentTreeArray([1, 2, 3])
        self.assertEqual(t.length, 3)
        t.update(0, 0, 99)
        self.assertEqual(t.length, 3)

    def test_version_count_grows(self):
        t = PersistentSegmentTreeArray([1])
        self.assertEqual(t.version_count, 1)
        t.update(0, 0, 2)
        self.assertEqual(t.version_count, 2)
        t.update(1, 0, 3)
        self.assertEqual(t.version_count, 3)


if __name__ == "__main__":
    unittest.main()
