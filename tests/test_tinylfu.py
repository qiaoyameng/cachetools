import unittest

from cachetools import FIFOCache, LRUCache, TinyLFUCache, cached, keys


def miss(cache, key):
    try:
        return cache[key]
    except KeyError:
        return None


def stable_hash(key):
    if isinstance(key, tuple):
        value = 0
        for item in key:
            value = value * 1_000_003 + stable_hash(item)
    elif isinstance(key, str):
        value = 0
        for char in key:
            value = value * 1_000_003 + ord(char)
    elif isinstance(key, int):
        value = key
    else:
        raise TypeError(key)
    return value & 0xFFFFFFFFFFFFFFFF


class TinyLFUCacheTest(unittest.TestCase):
    def make_cache(self, maxsize=2, **kwargs):
        options = {"hashfunc": stable_hash}
        options.update(kwargs)
        return TinyLFUCache(LRUCache(maxsize), **options)

    def test_admits_and_rejects_with_deterministic_trace(self):
        cache = self.make_cache(capacity=16)

        miss(cache, "hot")
        miss(cache, "hot")
        miss(cache, "hot")
        cache["hot"] = "a"
        cache["kept"] = "b"

        miss(cache, "scan")
        cache["scan"] = "c"

        self.assertIn("hot", cache)
        self.assertIn("kept", cache)
        self.assertNotIn("scan", cache)
        self.assertEqual(cache.cache["hot"], "a")
        self.assertEqual(cache.cache["kept"], "b")

        miss(cache, "new")
        miss(cache, "new")
        miss(cache, "new")
        miss(cache, "new")
        cache["new"] = "d"

        self.assertIn("kept", cache)
        self.assertIn("new", cache)
        self.assertNotIn("hot", cache)

    def test_ties_keep_existing_candidate(self):
        cache = self.make_cache(capacity=16)

        cache["old"] = 1
        cache["new"] = 2
        cache["other"] = 3

        self.assertIn("old", cache)
        self.assertIn("new", cache)
        self.assertNotIn("other", cache)

    def test_frequency_aging(self):
        cache = self.make_cache(capacity=16, sample_size=4)

        for _ in range(4):
            miss(cache, "key")

        self.assertEqual(cache.frequency("key"), 2)
        miss(cache, "key")
        self.assertEqual(cache.frequency("key"), 3)

    def test_counter_memory_is_bounded(self):
        cache = self.make_cache(capacity=16)
        self.assertEqual(cache.capacity, 16)
        self.assertEqual(cache.counter_size, 32)
        self.assertEqual(cache.sample_size, 20)

        for index in range(10000):
            miss(cache, index)

        self.assertEqual(cache.counter_size, 32)

    def test_hash_function_must_return_integer(self):
        with self.assertRaises(TypeError):
            TinyLFUCache(LRUCache(2), hashfunc=lambda key: key)

    def test_existing_key_is_always_overwritten(self):
        cache = self.make_cache(capacity=16)

        for _ in range(10):
            miss(cache, "hot")
        cache["hot"] = 1
        cache["cold"] = 2
        cache["hot"] = "updated"

        self.assertEqual(cache["hot"], "updated")
        self.assertEqual(cache["cold"], 2)

    def test_large_value_is_rejected_before_admission(self):
        cache = TinyLFUCache(
            LRUCache(3, getsizeof=len),
            capacity=16,
            hashfunc=stable_hash,
        )
        cache["small"] = "a"

        with self.assertRaises(ValueError):
            cache["large"] = "abcd"

        self.assertIn("small", cache)
        self.assertEqual(cache["small"], "a")
        self.assertNotIn("large", cache)

    def test_growing_existing_value_still_evicts(self):
        cache = TinyLFUCache(
            LRUCache(3, getsizeof=len),
            capacity=16,
            hashfunc=stable_hash,
        )
        cache["a"] = "a"
        cache["b"] = "bb"
        cache["a"] = "aaa"

        self.assertEqual(cache["a"], "aaa")
        self.assertNotIn("b", cache)

    def test_rejected_multi_candidate_admission_is_atomic(self):
        cache = TinyLFUCache(
            LRUCache(3, getsizeof=len),
            capacity=16,
            hashfunc=stable_hash,
        )
        cache["a"] = "a"
        cache["b"] = "bb"
        cache["b"]
        cache["b"]
        miss(cache, "new")

        cache["new"] = "xxx"

        self.assertEqual(cache["a"], "a")
        self.assertEqual(cache["b"], "bb")
        self.assertNotIn("new", cache)
        self.assertEqual(cache.currsize, 3)

    def test_clear_resets_frequency_and_cache(self):
        cache = self.make_cache(capacity=16)
        miss(cache, "key")
        cache["key"] = 1
        cache.clear()

        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.frequency("key"), 0)
        cache["key"] = 2
        self.assertEqual(cache["key"], 2)

    def test_composes_with_fifo(self):
        cache = TinyLFUCache(
            FIFOCache(2), capacity=16, hashfunc=stable_hash
        )

        for _ in range(3):
            miss(cache, "hot")
        cache["hot"] = 1
        cache["old"] = 2
        miss(cache, "scan")
        cache["scan"] = 3

        self.assertIn("hot", cache)
        self.assertIn("old", cache)
        self.assertNotIn("scan", cache)

    def test_composes_with_plain_mapping(self):
        cache = TinyLFUCache({}, hashfunc=stable_hash)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 3

        self.assertEqual(dict(cache), {"a": 3, "b": 2})

    def test_composes_with_tinylfu(self):
        inner = TinyLFUCache(
            LRUCache(2), capacity=16, hashfunc=stable_hash
        )
        cache = TinyLFUCache(
            inner, capacity=16, hashfunc=stable_hash
        )

        for _ in range(3):
            miss(cache, "hot")
        cache["hot"] = 1
        cache["kept"] = 2
        miss(cache, "scan")
        cache["scan"] = 3

        self.assertIn("hot", cache)
        self.assertIn("kept", cache)
        self.assertNotIn("scan", cache)

    def test_cached_decorator_returns_rejected_result(self):
        calls = []
        inner = LRUCache(2)
        cache = TinyLFUCache(inner, capacity=16, hashfunc=stable_hash)

        @cached(cache, info=True)
        def func(value):
            calls.append(value)
            return value + 10

        for _ in range(3):
            miss(cache, keys.hashkey(0))
        self.assertEqual(func(0), 10)
        self.assertEqual(func(1), 11)
        self.assertEqual(len(calls), 2)

        miss(cache, keys.hashkey(2))
        self.assertEqual(func(2), 12)
        self.assertEqual(len(calls), 3)
        self.assertNotIn(keys.hashkey(2), cache)

        self.assertEqual(func(2), 12)
        self.assertEqual(len(calls), 4)
        self.assertEqual(func.cache_info(), (0, 4, 2, 2))

    def test_cached_decorator_rejects_oversized_value(self):
        inner = LRUCache(1, getsizeof=lambda value: value["size"])
        cache = TinyLFUCache(inner, hashfunc=stable_hash)

        @cached(cache, info=True)
        def func(value):
            return {"size": value}

        self.assertEqual(func(2), {"size": 2})
        self.assertEqual(len(cache), 0)
        self.assertEqual(func.cache_info(), (0, 1, 1, 0))
        self.assertEqual(func(2), {"size": 2})
        self.assertEqual(func.cache_info(), (0, 2, 1, 0))

    def test_hash_collision_cannot_change_return_value(self):
        expected = ["A", "B", "C", "B", "A", "D", "A", "B"]
        stable_cache = self._hash_collision_cache(stable_hash)
        collision_cache = self._hash_collision_cache(lambda key: 0)

        self.assertEqual(
            self._run_hash_collision_trace(stable_cache),
            expected,
        )
        self.assertEqual(
            self._run_hash_collision_trace(collision_cache),
            expected,
        )

        stable_cache = self._hash_collision_cache(stable_hash)
        collision_cache = self._hash_collision_cache(lambda key: 0)
        self.assertEqual(
            self._hash_collision_calls(stable_cache), 8
        )
        self.assertEqual(
            self._hash_collision_calls(collision_cache), 6
        )

    def _hash_collision_cache(self, hashfunc):
        return TinyLFUCache(
            LRUCache(1),
            capacity=16,
            hashfunc=lambda key: hashfunc(keys.hashkey(key)),
        )

    def _run_hash_collision_trace(self, cache):
        @cached(cache)
        def func(value):
            return value.upper()

        trace = ("a", "b", "c", "b", "a", "d", "a", "b")
        return [func(value) for value in trace]

    def _hash_collision_calls(self, cache):
        calls = 0

        @cached(cache)
        def func(value):
            nonlocal calls
            calls += 1
            return value.upper()

        trace = ("a", "b", "c", "b", "a", "d", "a", "b")
        [func(value) for value in trace]
        return calls
