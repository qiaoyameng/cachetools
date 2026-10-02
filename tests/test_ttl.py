import math
import unittest

from cachetools import ExpireResult, TTLCache

from . import CacheTestMixin


class Timer:
    def __init__(self, auto=False):
        self.auto = auto
        self.time = 0

    def __call__(self):
        if self.auto:
            self.time += 1
        return self.time

    def tick(self):
        self.time += 1


class TTLTestCache(TTLCache):
    def __init__(self, maxsize, ttl=math.inf, **kwargs):
        TTLCache.__init__(self, maxsize, ttl=ttl, timer=Timer(), **kwargs)


class TTLCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = TTLTestCache

    def test_ttl(self):
        cache = TTLCache[int, int, int](maxsize=2, ttl=2, timer=Timer())
        self.assertEqual(0, cache.timer())
        self.assertEqual(2, cache.ttl)

        cache[1] = 1
        self.assertEqual(1, cache[1])
        self.assertEqual(1, len(cache))
        self.assertEqual({1}, set(cache))

        cache.timer.tick()
        self.assertEqual(1, cache[1])
        self.assertEqual(1, len(cache))
        self.assertEqual({1}, set(cache))

        cache[2] = 2
        self.assertEqual(1, cache[1])
        self.assertEqual(2, cache[2])
        self.assertEqual(2, len(cache))
        self.assertEqual({1, 2}, set(cache))

        cache.timer.tick()
        self.assertNotIn(1, cache)
        self.assertEqual(2, cache[2])
        self.assertEqual(1, len(cache))
        self.assertEqual({2}, set(cache))

        cache[3] = 3
        self.assertNotIn(1, cache)
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])
        self.assertEqual(2, len(cache))
        self.assertEqual({2, 3}, set(cache))

        cache.timer.tick()
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertEqual(3, cache[3])
        self.assertEqual(1, len(cache))
        self.assertEqual({3}, set(cache))

        cache.timer.tick()
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)

        with self.assertRaises(KeyError):
            del cache[1]
        with self.assertRaises(KeyError):
            cache.pop(2)
        with self.assertRaises(KeyError):
            del cache[3]

        self.assertEqual(0, len(cache))
        self.assertEqual(set(), set(cache))

    def test_ttl_timer(self):
        cache = TTLCache[int, int, int](maxsize=2, ttl=2, timer=Timer())
        self.assertEqual(cache.timer.time, 0)
        self.assertFalse(cache.timer.auto)

        cache[1] = 1
        cache.timer.tick()
        self.assertEqual(cache.timer.time, 1)
        self.assertEqual(1, cache[1])

        cache.timer.tick()
        self.assertEqual(cache.timer.time, 2)
        self.assertNotIn(1, cache)

    def test_ttl_lru(self):
        cache = TTLCache[int, int, int](maxsize=2, ttl=1, timer=Timer())

        cache[1] = 1
        cache[2] = 2
        cache[3] = 3

        self.assertEqual(len(cache), 2)
        self.assertNotIn(1, cache)
        self.assertEqual(cache[2], 2)
        self.assertEqual(cache[3], 3)

        cache[2]
        cache[4] = 4
        self.assertEqual(len(cache), 2)
        self.assertNotIn(1, cache)
        self.assertEqual(cache[2], 2)
        self.assertNotIn(3, cache)
        self.assertEqual(cache[4], 4)

        cache[5] = 5
        self.assertEqual(len(cache), 2)
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)
        self.assertEqual(cache[4], 4)
        self.assertEqual(cache[5], 5)

    def test_ttl_expire(self):
        cache = TTLCache[int, int, int](maxsize=3, ttl=3, timer=Timer())
        with cache.timer as time:
            self.assertEqual(time, cache.timer())
        self.assertEqual(3, cache.ttl)

        cache[1] = 1
        cache.timer.tick()
        cache[2] = 2
        cache.timer.tick()
        cache[3] = 3
        self.assertEqual(2, cache.timer())

        self.assertEqual({1, 2, 3}, set(cache))
        self.assertEqual(3, len(cache))
        self.assertEqual(1, cache[1])
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])

        items = cache.expire()
        self.assertEqual(set(), set(items))
        self.assertEqual({1, 2, 3}, set(cache))
        self.assertEqual(3, len(cache))
        self.assertEqual(1, cache[1])
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])

        items = cache.expire(3)
        self.assertEqual({(1, 1)}, set(items))
        self.assertEqual({2, 3}, set(cache))
        self.assertEqual(2, len(cache))
        self.assertNotIn(1, cache)
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])

        items = cache.expire(4)
        self.assertEqual({(2, 2)}, set(items))
        self.assertEqual({3}, set(cache))
        self.assertEqual(1, len(cache))
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertEqual(3, cache[3])

        items = cache.expire(5)
        self.assertEqual({(3, 3)}, set(items))
        self.assertEqual(set(), set(cache))
        self.assertEqual(0, len(cache))
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)

    def test_ttl_expire_budget(self):
        class VariableTTLCache(TTLCache):
            def __init__(self):
                super().__init__(maxsize=64, ttl=1, timer=Timer())

            def __setitem__(self, key, value):
                if key % 10 == 0:
                    self._TTLCache__ttl = 100
                    super().__setitem__(key, value)
                    self._TTLCache__ttl = 0
                else:
                    self._TTLCache__ttl = 1
                    super().__setitem__(key, value)

        cache = VariableTTLCache()
        for key in range(33):
            cache[key] = key

        cache.timer.tick()

        expired = []
        checked = []
        while True:
            batch = cache.expire(budget=2)
            checked.append(batch.checked)
            expired.extend(batch)
            for long_key in (0, 10, 20, 30):
                self.assertIn(long_key, cache)
                self.assertEqual(long_key, cache[long_key])
            self.assertEqual(4, len(cache))
            self.assertEqual(4, cache.currsize)
            if len(batch) == 0:
                break

        short_keys = [key for key in range(1, 33) if key % 10 != 0]
        self.assertEqual([(key, key) for key in short_keys], expired)
        self.assertEqual([2] * 15 + [1], checked)
        self.assertEqual({0, 10, 20, 30}, set(cache))
        self.assertEqual(4, len(cache))
        self.assertEqual(4, cache.currsize)

    def test_ttl_expire_budget_insertion(self):
        timer = Timer()
        cache = TTLCache(maxsize=64, ttl=1, timer=timer)
        for key in range(8):
            cache[key] = key

        timer.tick()
        first = cache.expire(budget=2)
        self.assertEqual(2, first.checked)
        cache[100] = 100
        for key in range(8):
            self.assertNotIn(key, cache)
        self.assertEqual(100, cache[100])

        batch = cache.expire(budget=2)
        self.assertEqual([], batch)
        self.assertEqual(1, batch.checked)
        self.assertEqual(100, cache[100])

    def test_ttl_expire_budget_custom_timer(self):
        class RawTimer:
            time = 0

            def __call__(self):
                return self.time

        timer = RawTimer()
        cache = TTLCache(maxsize=4, ttl=10, timer=timer)

        timer.time = 100
        cache["long"] = "long"
        timer.time = 0
        cache["short"] = "short"

        batch = cache.expire(time=101, budget=1)
        self.assertEqual([("short", "short")], list(batch))
        self.assertEqual(1, batch.checked)
        self.assertEqual("long", cache["long"])
        self.assertEqual(1, len(cache))

        timer.time = 1
        self.assertEqual("long", cache["long"])
        self.assertNotIn("short", cache)
        self.assertEqual(1, len(cache))

    def test_ttl_expire_budget_validation(self):
        cache = TTLCache(maxsize=1, ttl=1, timer=Timer())
        self.assertEqual(0, cache.expire(budget=0).checked)
        self.assertIsInstance(cache.expire(), ExpireResult)
        self.assertEqual(0, cache.expire(None, 0).checked)
        with self.assertRaises(ValueError):
            cache.expire(budget=-1)
        with self.assertRaises(ValueError):
            cache.expire(budget=1.0)

    def test_ttl_expire_budget_getsizeof(self):
        class RawTimer:
            time = 0

            def __call__(self):
                return self.time

        timer = RawTimer()
        cache = TTLCache(
            maxsize=10, ttl=2, timer=timer, getsizeof=lambda value: value
        )
        cache._TTLCache__ttl = 1
        cache[1] = 1
        cache._TTLCache__ttl = 2
        cache[2] = 2

        timer.time = 1
        result = cache.expire(budget=1)

        self.assertEqual(1, result.checked)
        self.assertEqual([(1, 1)], result)
        self.assertEqual(1, len(cache))
        self.assertEqual(2, cache.currsize)

    def test_ttl_atomic(self):
        cache = TTLCache[int, int, int](maxsize=1, ttl=2, timer=Timer(auto=True))
        cache[1] = 1
        self.assertEqual(1, cache[1])
        cache[1] = 1
        self.assertEqual(1, cache.get(1))
        cache[1] = 1
        self.assertEqual(1, cache.pop(1))
        cache[1] = 1
        self.assertEqual(1, cache.setdefault(1))
        cache[1] = 1
        cache.clear()
        self.assertEqual(0, len(cache))

    def test_ttl_tuple_key(self):
        cache = TTLCache[tuple[int, ...], int, int](maxsize=1, ttl=1, timer=Timer())
        self.assertEqual(1, cache.ttl)

        cache[(1, 2, 3)] = 42
        self.assertEqual(42, cache[(1, 2, 3)])
        cache.timer.tick()
        with self.assertRaises(KeyError):
            cache[(1, 2, 3)]
        self.assertNotIn((1, 2, 3), cache)

    def test_ttl_datetime(self):
        from datetime import datetime, timedelta

        cache = TTLCache[int, int, datetime](
            maxsize=1, ttl=timedelta(days=1), timer=datetime.now
        )

        cache[1] = 1
        self.assertEqual(1, len(cache))
        items = cache.expire(datetime.now())
        self.assertEqual([], list(items))
        self.assertEqual(1, len(cache))
        items = cache.expire(datetime.now() + timedelta(days=1))
        self.assertEqual([(1, 1)], list(items))
        self.assertEqual(0, len(cache))

    def test_ttl_clear(self):
        cache = TTLCache[int, int, int](maxsize=2, ttl=2, timer=Timer())

        cache[1] = 1
        cache[2] = 2
        cache.clear()

        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)

        # verify LRU eviction order is reset after clear
        cache[3] = 3
        cache[4] = 4
        cache[3]  # access 3 to make it most recently used
        cache[5] = 5  # should evict 4 (least recently used)

        self.assertEqual(2, len(cache))
        self.assertIn(3, cache)
        self.assertIn(5, cache)
        self.assertNotIn(4, cache)

        # verify TTL expiry still works after clear
        cache[42] = 42
        cache.timer.tick()
        cache.timer.tick()
        cache.timer.tick()  # past TTL
        self.assertNotIn(42, cache)
