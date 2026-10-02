import unittest

import cachetools
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

    def test_lru_resize_many_evict_order(self):
        cache = LRUCache(maxsize=5, getsizeof=len)

        a = [1, 2]
        b = [1, 2]
        c = [1]
        cache["a"] = a
        cache["b"] = b
        cache["c"] = c
        self.assertEqual(5, cache.currsize)
        # LRU order: a (least recently used), b, c

        c.extend([2, 3, 4])  # c grows to 4
        results = cache.resize_many(["c"])

        # total is now 2 + 2 + 4 = 8, so the least recently used items
        # a and b are evicted in that order
        self.assertEqual(cachetools.ResizeInfo("resized", 4), results["c"])
        self.assertEqual(["c"], list(cache))
        self.assertEqual(4, cache.currsize)

    def test_lru_resize_evicted(self):
        cache = LRUCache(maxsize=5, getsizeof=len)

        a = [1, 2]
        cache["a"] = a
        cache["b"] = [1, 2, 3]
        # LRU order: a (least recently used), b

        a.extend([3, 4])  # a grows to 4 without touching the cache
        results = cache.resize_many(["a"])

        # total is now 4 + 3 = 7, so a itself is evicted
        self.assertEqual(cachetools.ResizeInfo("evicted", 4), results["a"])
        self.assertEqual(["b"], list(cache))
        self.assertEqual(3, cache.currsize)

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
