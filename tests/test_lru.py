import unittest

from cachetools import LRUCache

from . import CacheTestMixin


class LRUCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = LRUCache

    def test_lru(self):
        cache = LRUCache[int, int](maxsize=2)

        cache[1] = 1
        cache[2] = 2
        cache[3] = 3

        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[2], 2)
        self.assertEqual(cache[3], 3)
        self.assertNotIn(1, cache)

        cache[2]
        cache[4] = 4
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[2], 2)
        self.assertEqual(cache[4], 4)
        self.assertNotIn(3, cache)

        cache[5] = 5
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[4], 4)
        self.assertEqual(cache[5], 5)
        self.assertNotIn(2, cache)

    def test_lru_getsizeof(self):
        cache = LRUCache[int, int](maxsize=3, getsizeof=lambda x: x)

        cache[1] = 1
        cache[2] = 2

        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[1], 1)
        self.assertEqual(cache[2], 2)

        cache[3] = 3

        self.assertEqual(len(cache), 1)
        self.assertEqual(cache[3], 3)
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)

        with self.assertRaises(ValueError):
            cache[4] = 4
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache[3], 3)

    def test_lru_remeasure_eviction_order(self):
        cache = LRUCache[str, list[int]](maxsize=6, getsizeof=len)
        first = [0]
        second = [0]
        third = [0]
        evicted_target = [0]
        cache["first"] = first
        cache["second"] = second
        cache["third"] = third
        cache["target"] = evicted_target

        first.extend([0, 0, 0, 0])
        second.extend([0, 0, 0, 0])

        result = cache.remeasure("first", "second", "target")

        self.assertEqual(
            [("first", first), ("second", second)], result.evicted
        )
        self.assertEqual(("updated", 1, 1), result["target"])
        self.assertEqual(2, cache.currsize)
        self.assertEqual({"third", "target"}, set(cache))

    def test_lru_remeasure_target_evicted(self):
        cache = LRUCache[str, list[int]](maxsize=7, getsizeof=len)
        growing = [0]
        keeper = [0, 0, 0]
        cache["growing"] = growing
        cache["keeper"] = keeper

        growing.extend([0, 0, 0, 0])

        result = cache.remeasure("growing")

        self.assertEqual([("growing", growing)], result.evicted)
        self.assertEqual(("evicted", 1, 5), result["growing"])
        self.assertEqual(3, cache.currsize)
        self.assertEqual({"keeper"}, set(cache))

    def test_lru_remeasure_failure_then_retry_eviction(self):
        cache = LRUCache[str, list[int]](maxsize=5, getsizeof=len)
        growing = [0]
        shrinking = [0, 0, 0]
        failing = [0]
        cache["growing"] = growing
        cache["shrinking"] = shrinking
        cache["failing"] = failing

        growing.extend([0, 0, 0])
        shrinking.pop()
        shrinking.pop()

        def getsizeof(value):
            if value is failing:
                raise RuntimeError("cannot measure")
            return len(value)

        cache.getsizeof = getsizeof

        with self.assertRaisesRegex(RuntimeError, "cannot measure"):
            cache.remeasure("growing", "shrinking", "failing")

        self.assertEqual(5, cache.currsize)
        self.assertEqual({"growing", "shrinking", "failing"}, set(cache))

        cache.getsizeof = len
        result = cache.remeasure("growing", "shrinking", "failing")

        self.assertEqual([("growing", growing)], result.evicted)
        self.assertEqual(("evicted", 1, 4), result["growing"])
        self.assertEqual(("updated", 3, 1), result["shrinking"])
        self.assertEqual(("updated", 1, 1), result["failing"])
        self.assertEqual(2, cache.currsize)
        self.assertEqual({"shrinking", "failing"}, set(cache))

    def test_lru_update_existing(self):
        cache = LRUCache[int, int | str](maxsize=2)

        cache[1] = 1
        cache[2] = 2
        cache[1] = "updated"
        cache[3] = 3

        self.assertEqual(cache[1], "updated")
        self.assertIn(3, cache)
        self.assertNotIn(2, cache)

    def test_lru_clear(self):
        cache = LRUCache[int, int](maxsize=2)

        cache[1] = 1
        cache[2] = 2
        cache.clear()

        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)

        # verify LRU order is reset after clear
        cache[3] = 3
        cache[4] = 4
        cache[3]  # access 3 to make it most recently used
        cache[5] = 5  # should evict 4 (least recently used)

        self.assertEqual(2, len(cache))
        self.assertIn(3, cache)
        self.assertIn(5, cache)
        self.assertNotIn(4, cache)
