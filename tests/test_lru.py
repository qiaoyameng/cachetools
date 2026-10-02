import unittest

from cachetools import FrequencySketch, LRUCache, TinyLFU, cached

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


class TinyLFUTest(unittest.TestCase):
    def test_sketch_bounded_and_aging(self):
        sketch = FrequencySketch(width=4, depth=2, sample_size=10)

        self.assertEqual(sketch.memory_size, 8)

        for _ in range(3):
            sketch.add("a")
        sketch.add("b")

        self.assertEqual(sketch.frequency("a"), 3)
        self.assertEqual(sketch.frequency("b"), 1)
        self.assertEqual(sketch.sample_count, 4)
        sketch.age()
        self.assertEqual(sketch.frequency("a"), 1)
        self.assertEqual(sketch.frequency("b"), 0)

        for key in range(10000):
            sketch.add(key)

        self.assertEqual(sketch.memory_size, 8)

    def test_admits_frequent_new_key_and_rejects_one_shot(self):
        hashes = {"a": (0, 1), "b": (4, 1), "c": (8, 1)}
        policy = TinyLFU(capacity=16, hasher=hashes.get)
        cache = LRUCache(maxsize=2, getsizeof=len, admission=policy)

        cache["a"] = "a"
        cache["b"] = "b"

        policy.record("a")

        cache["c"] = "c"
        self.assertNotIn("c", cache)
        self.assertEqual(tuple(cache), ("a", "b"))

        policy.record("c")
        cache["c"] = "c"
        self.assertIn("c", cache)
        self.assertIn("b", cache)
        self.assertNotIn("a", cache)

    def test_rejection_does_not_partially_evict_sized_cache(self):
        policy = TinyLFU(capacity=16)
        cache = LRUCache(maxsize=4, getsizeof=len, admission=policy)

        cache["aa"] = "aa"
        cache["bb"] = "bb"

        policy.record("x")
        cache["xxxx"] = "xxxx"

        self.assertNotIn("x", cache)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.currsize, 4)

    def test_existing_key_can_grow_and_replace(self):
        policy = TinyLFU(capacity=4)
        cache = LRUCache(maxsize=4, getsizeof=len, admission=policy)

        cache["a"] = "a"
        cache["bb"] = "bb"
        cache["a"] = "ccc"

        self.assertEqual(cache["a"], "ccc")
        self.assertEqual(cache.currsize, 3)
        self.assertNotIn("bb", cache)

    def test_too_large_value_is_rejected_before_recording(self):
        policy = TinyLFU(capacity=4)
        cache = LRUCache(maxsize=2, getsizeof=len, admission=policy)

        with self.assertRaises(ValueError):
            cache["large"] = "x" * 3

        self.assertNotIn("large", cache)
        self.assertEqual(policy.frequency("large"), 0)

    def test_decorator_returns_rejected_result_and_counts_miss(self):
        calls = []
        policy = TinyLFU(capacity=16)
        cache = LRUCache(maxsize=2, admission=policy)

        @cached(cache, key=lambda key: key, info=True)
        def func(key):
            calls.append(key)
            return key.upper()

        func("a")
        func("b")

        for _ in range(2):
            policy.record("a")

        self.assertEqual(func("c"), "C")
        self.assertNotIn("c", cache)
        self.assertEqual(func.cache_info(), (0, 3, 2, 2))

        policy.record("c")
        policy.record("c")
        self.assertEqual(func("c"), "C")
        self.assertIn("c", cache)
        self.assertEqual(func("c"), "C")
        self.assertEqual(len(calls), 4)
        self.assertEqual(func.cache_info(), (1, 4, 2, 2))

    def test_hash_collision_cannot_change_return_value(self):
        def run(hasher):
            calls = []
            policy = TinyLFU(capacity=8, hasher=hasher)
            cache = LRUCache(maxsize=2, admission=policy)

            @cached(cache, key=lambda key: key, info=True)
            def func(key):
                calls.append(key)
                return key.upper()

            values = [
                func("a"),
                func("b"),
                func("a"),
                func("c"),
                func("c"),
            ]
            return values, func.cache_info(), calls, cache

        collision_hashes = {"a": (0, 1), "b": (3, 2), "c": (0, 1)}.get
        distinct_hashes = {"a": (0, 1), "b": (3, 2), "c": (4, 3)}.get

        collision_values, collision_info, collision_calls, collision_cache = run(
            collision_hashes
        )
        distinct_values, distinct_info, distinct_calls, _ = run(distinct_hashes)

        self.assertEqual(collision_values, ["A", "B", "A", "C", "C"])
        self.assertEqual(collision_values, distinct_values)
        self.assertGreater(collision_info.hits, distinct_info.hits)
        self.assertEqual(collision_calls.count("c"), 1)
        self.assertEqual(distinct_calls.count("c"), 2)
        self.assertEqual(collision_cache["c"], "C")
        self.assertEqual(tuple(collision_cache), ("a", "c"))

    def test_clear_does_not_reset_external_policy(self):
        policy = TinyLFU(capacity=4)
        cache = LRUCache(maxsize=2, admission=policy)

        cache["a"] = "a"
        cache["a"]
        cache.clear()

        self.assertNotIn("a", cache)
        self.assertEqual(policy.frequency("a"), 2)

        policy.clear()
        self.assertEqual(policy.frequency("a"), 0)
