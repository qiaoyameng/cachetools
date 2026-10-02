import unittest

from cachetools import ARCCache, LRUCache

from . import CacheTestMixin


class ARCCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = ARCCache


class ARCCacheBehaviorTest(unittest.TestCase):
    HOT = tuple(f"hot:{i}" for i in range(4))
    SCAN = tuple(f"scan:{i}" for i in range(8))

    @staticmethod
    def queues(cache):
        return {
            "t1": cache._ARCCache__t1,
            "t2": cache._ARCCache__t2,
            "b1": cache._ARCCache__b1,
            "b2": cache._ARCCache__b2,
        }

    def assert_metadata_bounded(self, cache, maxsize=None):
        if maxsize is None:
            maxsize = cache.maxsize
        self.assertLessEqual(cache.currsize, maxsize)
        t1 = cache._ARCCache__t1size
        t2 = cache._ARCCache__t2size
        b1 = len(cache._ARCCache__b1)
        b2 = len(cache._ARCCache__b2)
        self.assertEqual(t1, sum(cache._ARCCache__t1.values()))
        self.assertEqual(t2, sum(cache._ARCCache__t2.values()))
        self.assert_ghost_values_none(cache)
        self.assertLessEqual(t1 + t2, maxsize)
        self.assertLessEqual(len(cache._ARCCache__b1), maxsize)
        self.assertLessEqual(len(cache._ARCCache__b2), 2 * maxsize)
        self.assertLessEqual(t1 + b1, maxsize)
        self.assertLessEqual(b2, 2 * maxsize)
        self.assertLessEqual(t2 + b2, 2 * maxsize)
        self.assertLessEqual(t1 + t2 + b1 + b2, 2 * maxsize)

    @staticmethod
    def assert_ghost_values_none(cache):
        assert all(value is None for value in cache._ARCCache__b1.values())
        assert all(value is None for value in cache._ARCCache__b2.values())

    @staticmethod
    def read(cache, key):
        try:
            cache[key]
            return True
        except KeyError:
            return False

    def repopulate_if_remembered(self, cache, key):
        queues = self.queues(cache)
        remembered = key in queues["b1"] or key in queues["b2"]
        self.read(cache, key)
        if remembered:
            cache[key] = key
        self.assert_metadata_bounded(cache)
        return remembered

    def test_scan_preserves_repeated_set(self):
        arc = ARCCache[str, str](maxsize=4)
        lru = LRUCache[str, str](maxsize=4)

        for _ in range(3):
            for key in self.HOT:
                arc[key] = key
                lru[key] = key
                self.assert_metadata_bounded(arc)

        for key in self.SCAN:
            arc[key] = key
            lru[key] = key
            self.assert_metadata_bounded(arc)
            self.assertEqual(len(arc), arc.currsize)
            self.assertLessEqual(len(arc), 4)
            self.assertLessEqual(sum(map(len, self.queues(arc).values())), 8)
            self.assertTrue(
                all(
                    recorded_size == 1
                    for recorded_size in arc._ARCCache__t1.values()
                )
            )
            self.assertTrue(
                all(
                    recorded_size == 1
                    for recorded_size in arc._ARCCache__t2.values()
                )
            )
            self.assert_ghost_values_none(arc)

        arc_remembered = sum(
            self.repopulate_if_remembered(arc, key) for key in self.HOT
        )
        lru_hits = sum(self.read(lru, key) for key in self.HOT)
        arc_hits_after_repair = sum(self.read(arc, key) for key in self.HOT)

        self.assertEqual(4, arc_remembered)
        self.assertEqual(4, arc_hits_after_repair)
        self.assertEqual(0, lru_hits)
        self.assertEqual(4, len(arc))
        self.assertEqual(4, arc.currsize)
        self.assertEqual(set(self.HOT), set(arc))

    def test_ghost_hits_adjust_target_partition(self):
        cache = ARCCache[int, int](maxsize=4)
        for key in range(4):
            cache[key] = key

        cache[0]
        cache[1]
        cache[2]
        cache[4] = 4
        self.assertIn(3, cache._ARCCache__b1)
        self.assertEqual(0, cache._ARCCache__target)

        cache[3] = 3
        self.assertEqual(1, cache._ARCCache__target)
        self.assertNotIn(3, cache._ARCCache__b1)
        self.assertIn(3, cache._ARCCache__t2)
        self.assert_metadata_bounded(cache)

        cache[5] = 5
        self.assertIn(0, cache._ARCCache__b2)
        cache[0] = 0
        self.assertEqual(0, cache._ARCCache__target)
        self.assertNotIn(0, cache._ARCCache__b2)
        self.assertIn(0, cache._ARCCache__t2)
        self.assert_metadata_bounded(cache)

    def test_weighted_uses_getsizeof_units(self):
        cache = ARCCache[int, int](maxsize=6, getsizeof=lambda value: value)
        cache[2] = 2
        cache[4] = 4
        cache[2]
        cache[3] = 3

        self.assertEqual(5, cache.currsize)
        self.assertIn(3, cache)
        self.assertIn(2, cache)
        self.assertIn(4, cache._ARCCache__b1)
        self.assertNotIn(4, cache)
        self.assert_metadata_bounded(cache)

        cache[4] = 4
        self.assertEqual(1, cache._ARCCache__target)
        self.assertEqual(6, cache.currsize)
        self.assertIn(4, cache)
        self.assertIn(3, cache._ARCCache__b1)
        self.assert_metadata_bounded(cache)

    def test_overwrite_delete_and_clear_update_queues(self):
        cache = ARCCache[int, str](maxsize=3)
        cache[1] = "a"
        cache[2] = "b"
        cache[1]
        cache[3] = "c"
        self.assert_metadata_bounded(cache)

        cache[2] = "bb"
        self.assertIn(2, cache._ARCCache__t2)
        self.assertIn(3, cache._ARCCache__t1)
        self.assertEqual(3, cache.currsize)

        del cache[1]
        self.assertNotIn(1, self.queues(cache)["t2"])
        self.assertNotIn(1, self.queues(cache)["b2"])
        self.assertEqual(2, cache.currsize)

        del cache[2]
        self.assertEqual(1, len(cache))
        self.assertEqual(1, cache.currsize)
        self.assertEqual(0, len(self.queues(cache)["t2"]))

        cache[4] = "d"
        cache.clear()
        for queue in self.queues(cache).values():
            self.assertEqual(0, len(queue))
        self.assertEqual(0, cache.currsize)
        self.assertEqual(0, cache._ARCCache__target)

    def test_oversize_value_rejection_changes_nothing(self):
        cache = ARCCache[int, int](maxsize=3, getsizeof=lambda value: value)
        cache[1] = 1
        cache[2] = 2

        with self.assertRaises(ValueError):
            cache[3] = 4

        queues = self.queues(cache)
        self.assertEqual(2, len(cache))
        self.assertEqual(3, cache.currsize)
        self.assertEqual({1, 2}, set(queues["t1"]))
        self.assertEqual(0, len(queues["t2"]) + len(queues["b1"]) + len(queues["b2"]))
        self.assert_metadata_bounded(cache)

    def test_keys_only_need_hash_and_equality(self):
        class Key:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return self.value

            def __eq__(self, other):
                return isinstance(other, Key) and self.value == other.value

        cache = ARCCache[Key, int](maxsize=2)
        first = Key(1)
        second = Key(2)
        cache[first] = 10
        cache[second] = 20
        cache[first]
        cache[Key(3)] = 30

        self.assertEqual(10, cache[Key(1)])
        self.assertIn(Key(3), cache)
        self.assertIn(second, cache._ARCCache__b1)
        self.assert_metadata_bounded(cache)


if __name__ == "__main__":
    unittest.main()
