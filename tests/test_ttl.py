import math
import unittest

from cachetools import TTLCache

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

    def test_ttl_expire_budget(self):
        timer = Timer()
        cache = TTLCache[int, int, int](maxsize=64, ttl=100, timer=timer)

        # interleave many short-lived items with a few long-lived ones:
        # the short-lived items are inserted first at t = 0 (expire at
        # t = 100), the long-lived items later at t = 50 (expire 150+)
        for key in range(10):
            cache[key * 10] = key * 10
        timer.time = 50
        for key in range(10):
            cache[-key - 1] = -(key + 1)
        cache["long"] = "L"

        self.assertEqual(21, len(cache))

        # jump past the short-lived TTL but keep long-lived items alive
        timer.time = 100
        self.assertNotIn(0, cache)
        self.assertNotIn(90, cache)
        self.assertEqual("L", cache["long"])
        for key in range(10):
            self.assertEqual(-key - 1, cache[-key - 1])

        # repeatedly clean up with a very small budget and record the
        # work of each round
        removed_total = 0
        checked_rounds = []
        rounds = 0
        while True:
            result = cache.expire(budget=3)
            self.assertLessEqual(result.checked, 3)
            checked_rounds.append(result.checked)
            removed_total += len(result)
            rounds += 1
            # the long-lived items stay readable through every round
            self.assertEqual("L", cache["long"])
            self.assertIn("long", cache)
            for key in range(10):
                self.assertEqual(-key - 1, cache[-key - 1])
            for key, _value in result:
                self.assertEqual(0, key % 10)
            if not result.remaining:
                break
            self.assertLess(rounds, 100)

        self.assertEqual([3, 3, 3, 1], checked_rounds)
        self.assertEqual(10, removed_total)
        self.assertFalse(result.remaining)
        self.assertEqual(11, len(cache))
        self.assertEqual(11, len(set(cache)))
        self.assertEqual("L", cache["long"])
        for key in range(10):
            self.assertEqual(-key - 1, cache[-key - 1])

    def test_ttl_expire_budget_semantics(self):
        timer = Timer()
        cache = TTLCache[int, int, int](maxsize=10, ttl=2, timer=timer)
        cache[1] = 1
        cache[2] = 2
        cache[3] = 3
        timer.time = 3

        # one item physically removed per bounded call; the others stay
        # logically expired but show consistent mapping semantics
        result = cache.expire(budget=1)
        self.assertEqual([(1, 1)], list(result))
        self.assertEqual(1, result.checked)
        self.assertTrue(result.remaining)

        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)
        self.assertIsNone(cache.get(2))
        self.assertEqual("x", cache.get(3, "x"))
        self.assertEqual(set(), set(cache))

        # len() runs a full cleanup and reclaims the remaining items
        self.assertEqual(0, len(cache))
        result = cache.expire(budget=1)
        self.assertEqual([], list(result))
        self.assertEqual(0, result.checked)
        self.assertFalse(result.remaining)

    def test_ttl_expire_budget_zero(self):
        timer = Timer()
        cache = TTLCache[int, int, int](maxsize=4, ttl=1, timer=timer)
        cache[1] = 1
        cache[2] = 2
        timer.time = 2

        result = cache.expire(budget=0)
        self.assertEqual([], list(result))
        self.assertEqual(0, result.checked)
        self.assertTrue(result.remaining)
        # nothing was reclaimed ...
        self.assertEqual([(1, 1)], cache.expire(budget=1))
        # ... but expired items remain inaccessible
        self.assertNotIn(2, cache)

    def test_ttl_expire_budget_invalid(self):
        cache = TTLCache[int, int, int](maxsize=4, ttl=1, timer=Timer())
        with self.assertRaises(ValueError):
            cache.expire(budget=-1)

    def test_ttl_expire_budget_custom_timer(self):
        # a non-monotonic timer can place an expired link behind live
        # links; bounded cleanup must never delete a live value, and the
        # pending link must not be stuck forever once time advances
        timer = Timer()
        cache = TTLCache[int, int, int](maxsize=10, ttl=10, timer=timer)
        cache[1] = 1  # expires at 10
        timer.time = 5
        cache[2] = 2  # expires at 15
        timer.time = 8
        cache[3] = 3  # expires at 18
        timer.time = 12  # key 1 expired, but live key 2 is ahead

        result = cache.expire(budget=2)
        self.assertEqual([(1, 1)], list(result))
        self.assertLessEqual(result.checked, 2)
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])

        # once key 2 expires it is reclaimed; key 3 survives
        timer.time = 16
        result = cache.expire(budget=1)
        self.assertEqual([(2, 2)], list(result))
        self.assertEqual(3, cache[3])
        self.assertEqual(1, len(cache))

    def test_ttl_expire_budget_insertion_between_rounds(self):
        timer = Timer()
        cache = TTLCache[int, int, int](maxsize=100, ttl=10, timer=timer)
        for key in range(6):
            cache[key] = key
        timer.time = 11

        first = cache.expire(budget=2)
        self.assertEqual([(0, 0), (1, 1)], list(first))

        # inserting between two cleanup rounds performs the regular full
        # expiration sweep first; it must not resurrect the removed
        # items, and the new item must remain readable
        cache[100] = 100
        self.assertNotIn(0, cache)
        self.assertNotIn(1, cache)
        self.assertEqual(100, cache[100])

        second = cache.expire(budget=10)
        self.assertEqual([], list(second))
        self.assertFalse(second.remaining)
        self.assertEqual([100], list(cache))
        self.assertEqual(100, cache[100])
