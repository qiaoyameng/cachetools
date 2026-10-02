import unittest

from cachetools import DecayingLFUCache

from . import CacheTestMixin


class DecayingLFUCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = DecayingLFUCache

    def test_decay_period_invalid(self):
        with self.assertRaises(ValueError):
            DecayingLFUCache(maxsize=1, decay_period=0)
        with self.assertRaises(ValueError):
            DecayingLFUCache(maxsize=1, decay_period=-1)

    def test_decaying_lfu(self):
        # with a large decay period, behaves like a plain LFU cache
        cache = DecayingLFUCache[int, int](maxsize=2, decay_period=10**9)

        cache[1] = 1
        cache[1]
        cache[2] = 2
        cache[3] = 3

        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[1], 1)
        self.assertNotIn(2, cache)
        self.assertIn(3, cache)

        cache[4] = 4
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[4], 4)
        self.assertEqual(cache[1], 1)

    def test_tie_broken_by_stable_recency(self):
        cache = DecayingLFUCache[int, int](maxsize=2, decay_period=10**9)

        cache[1] = 1
        cache[2] = 2
        cache[1]  # both now have count 2, 1 reached it first
        cache[2]
        cache[3] = 3

        self.assertNotIn(1, cache)
        self.assertIn(2, cache)
        self.assertIn(3, cache)

    def test_hot_key_loses_protection_after_decay(self):
        cache = DecayingLFUCache[str, str](maxsize=3, decay_period=10)

        cache["A"] = "a"
        for _ in range(9):
            cache["A"]  # make A the hotspot

        cache["B"] = "b"
        cache["C"] = "c"

        # A is hot: a new insertion evicts a cold item, not A
        cache["D"] = "d"
        self.assertIn("A", cache)
        self.assertNotIn("B", cache)
        self.assertIn("C", cache)
        self.assertIn("D", cache)

        # accesses shift to C and D, crossing several decay periods
        for _ in range(30):
            cache["C"]
            cache["D"]

        # A has not been accessed, so its count decayed away and the
        # next insertion evicts A, while the new hotspots survive
        cache["E"] = "e"
        self.assertNotIn("A", cache)
        self.assertIn("C", cache)
        self.assertIn("D", cache)
        self.assertIn("E", cache)

    def test_decay_merges_counts_in_stable_order(self):
        cache = DecayingLFUCache[str, int](maxsize=3, decay_period=2)

        cache["x"] = 1
        cache["x"]  # decay fires, then x is counted again: count 2
        cache["y"] = 2
        cache["y"]  # decay merges y (count 1) and x (count 2) into one
        # count-1 bucket, y before x; then y is counted again
        cache["z"] = 3
        cache["w"] = 4  # evicts x, the least recently counted with count 1

        self.assertNotIn("x", cache)
        self.assertIn("y", cache)
        self.assertIn("z", cache)
        self.assertIn("w", cache)

    def test_clock_based_decay(self):
        now = [0.0]
        cache = DecayingLFUCache[str, int](
            maxsize=2, decay_period=10.0, clock=lambda: now[0]
        )

        cache["a"] = 1
        for _ in range(5):
            cache["a"]  # a is hot
        cache["b"] = 2

        # no time has passed: a new insertion evicts cold b, not a
        cache["c"] = 3
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)

        # let several decay periods pass while only c is accessed
        for _ in range(4):
            now[0] += 10.0
            cache["c"]

        # a's count decayed away, so it is evicted next
        cache["d"] = 4
        self.assertNotIn("a", cache)
        self.assertIn("c", cache)
        self.assertIn("d", cache)

    def test_counts_stay_bounded(self):
        decay_period = 10
        cache = DecayingLFUCache[str, int](maxsize=1, decay_period=decay_period)

        cache["k"] = 1
        for _ in range(10000):
            cache["k"]

        link = cache._DecayingLFUCache__links["k"]
        self.assertLessEqual(link.count, 2 * decay_period)

    def test_decaying_lfu_getsizeof(self):
        cache = DecayingLFUCache[int, int](
            maxsize=3, getsizeof=lambda x: x, decay_period=5
        )

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

    def test_decaying_lfu_update_existing(self):
        cache = DecayingLFUCache[int, int | str](maxsize=2, decay_period=10)

        cache[1] = 1
        cache[2] = 2
        cache[1] = "updated"
        cache[3] = 3

        self.assertEqual(cache[1], "updated")
        self.assertIn(3, cache)
        self.assertNotIn(2, cache)

    def test_decaying_lfu_delete(self):
        cache = DecayingLFUCache[int, int](maxsize=2, decay_period=100)

        cache[1] = 1
        cache[2] = 2
        cache[1]  # count 2
        del cache[1]

        cache[3] = 3
        cache[4] = 4  # evicts 2; stale index of 1 must not interfere

        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertIn(3, cache)
        self.assertIn(4, cache)

    def test_decaying_lfu_clear(self):
        cache = DecayingLFUCache[int, int](maxsize=2, decay_period=3)

        cache[1] = 1
        cache[1]
        cache[1]
        cache[2] = 2

        cache.clear()

        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)

        # frequency tracking and decay state are reset after clear
        cache[3] = 3
        cache[4] = 4
        cache[3]
        cache[5] = 5  # should evict 4

        self.assertEqual(2, len(cache))
        self.assertIn(3, cache)
        self.assertIn(5, cache)
        self.assertNotIn(4, cache)

    def test_zero_capacity(self):
        cache = DecayingLFUCache[int, int](maxsize=0, decay_period=1)

        with self.assertRaises(ValueError):
            cache[1] = 1
        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)

        with self.assertRaises(KeyError):
            cache.popitem()
        with self.assertRaises(KeyError):
            cache[1]

        cache.clear()
        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)
