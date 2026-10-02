import math
import unittest

from cachetools import TLRUCache

from . import CacheTestMixin


def default_ttu(_key, _value, _time):
    return math.inf


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


class TLRUTestCache(TLRUCache):
    def __init__(self, maxsize, ttu=default_ttu, **kwargs):
        TLRUCache.__init__(self, maxsize, ttu, timer=Timer(), **kwargs)


class TLRUCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = TLRUTestCache

    def test_ttu(self):
        cache = TLRUCache[int, int, int](
            maxsize=6, ttu=lambda _, v, t: t + v + 1, timer=Timer()
        )
        self.assertEqual(0, cache.timer())
        self.assertEqual(3, cache.ttu(0, 1, 1))

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
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])
        self.assertEqual(2, len(cache))
        self.assertEqual({2, 3}, set(cache))

        cache[1] = 1
        self.assertEqual(1, cache[1])
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])
        self.assertEqual(3, len(cache))
        self.assertEqual({1, 2, 3}, set(cache))

        cache.timer.tick()
        self.assertEqual(1, cache[1])
        self.assertNotIn(2, cache)
        self.assertEqual(3, cache[3])
        self.assertEqual(2, len(cache))
        self.assertEqual({1, 3}, set(cache))

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

    def test_ttu_lru(self):
        def ttu(_k, _v, t):
            return t + 1

        cache = TLRUCache[int, int, int](maxsize=2, ttu=ttu, timer=Timer())
        self.assertEqual(0, cache.timer())
        self.assertEqual(2, cache.ttu(0, 0, 1))

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

    def test_overwrite_existing_with_expired(self):
        # Overwriting an existing key with an already-expired value must not
        # silently retain the previous value: the assignment cannot store the
        # new (dead-on-arrival) value, so the key has to be evicted instead.
        def ttu(_k, value, t):
            return t + value

        cache = TLRUCache[int, int, int](maxsize=2, ttu=ttu, timer=Timer())
        cache[1] = 5
        self.assertEqual(cache[1], 5)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.currsize, 1)

        cache[1] = 0  # ttu -> t, i.e. not (t < t): immediately expired
        self.assertNotIn(1, cache)
        self.assertIsNone(cache.get(1))
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.currsize, 0)
        self.assertEqual(set(cache), set())

    def test_ttu_expire(self):
        def ttu(_k, _v, t):
            return t + 3

        cache = TLRUCache[int, int, int](maxsize=3, ttu=ttu, timer=Timer())
        with cache.timer as time:
            self.assertEqual(time, cache.timer())

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

    def test_ttu_expired(self):
        cache = TLRUCache[int, None, int](
            maxsize=1, ttu=lambda k, _, t: t + k, timer=Timer()
        )
        cache[1] = None
        self.assertEqual(cache[1], None)
        self.assertEqual(1, len(cache))
        cache[0] = None
        self.assertNotIn(0, cache)
        self.assertEqual(cache[1], None)
        self.assertEqual(1, len(cache))
        cache[-1] = None
        self.assertNotIn(-1, cache)
        self.assertNotIn(0, cache)
        self.assertEqual(cache[1], None)
        self.assertEqual(1, len(cache))

    def test_ttu_atomic(self):
        cache = TLRUCache[int, int, int](
            maxsize=1, ttu=lambda k, v, t: t + 2, timer=Timer(auto=True)
        )
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

    def test_ttu_tuple_key(self):
        cache = TLRUCache[tuple[int, ...], int, int](
            maxsize=1, ttu=lambda k, v, t: t + 1, timer=Timer()
        )

        cache[(1, 2, 3)] = 42
        self.assertEqual(42, cache[(1, 2, 3)])
        cache.timer.tick()
        with self.assertRaises(KeyError):
            cache[(1, 2, 3)]
        self.assertNotIn((1, 2, 3), cache)

    def test_ttu_reverse_insert(self):
        def ttu(_k, v, t):
            return t + v

        cache = TLRUCache[int, int, int](maxsize=4, ttu=ttu, timer=Timer())
        self.assertEqual(0, cache.timer())

        cache[3] = 3
        cache[2] = 2
        cache[1] = 1
        cache[0] = 0

        self.assertEqual({1, 2, 3}, set(cache))
        self.assertEqual(3, len(cache))
        self.assertNotIn(0, cache)
        self.assertEqual(1, cache[1])
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])

        cache.timer.tick()

        self.assertEqual({2, 3}, set(cache))
        self.assertEqual(2, len(cache))
        self.assertNotIn(0, cache)
        self.assertNotIn(1, cache)
        self.assertEqual(2, cache[2])
        self.assertEqual(3, cache[3])

        cache.timer.tick()

        self.assertEqual({3}, set(cache))
        self.assertEqual(1, len(cache))
        self.assertNotIn(0, cache)
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertEqual(3, cache[3])

        cache.timer.tick()

        self.assertEqual(set(), set(cache))
        self.assertEqual(0, len(cache))
        self.assertNotIn(0, cache)
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)

    def test_ttu_heap_cleanup(self):
        def ttu(_k, _v, t):
            return t + 1

        cache = TLRUCache[int, int, int](maxsize=4, ttu=ttu, timer=Timer())
        self.assertEqual(0, cache.timer())

        cache[1] = 1
        cache[2] = 2

        # replace items to accumulate removed entries in the internal heap
        for i in range(5):
            cache[1] = 10 + i
            cache[2] = 20 + i

        # this should compact the internal heap
        expired = cache.expire()
        self.assertEqual([], expired)

        # verify cache is functional after cleanup
        self.assertEqual(2, len(cache))
        self.assertEqual(14, cache[1])
        self.assertEqual(24, cache[2])
        cache[3] = 3
        cache[4] = 4
        self.assertEqual(4, len(cache))

        cache.timer.tick()
        expired = cache.expire()
        self.assertEqual(4, len(expired))
        self.assertEqual(0, len(cache))

    def test_tlru_datetime(self):
        from datetime import datetime, timedelta

        def ttu(_k, _v, t):
            return t + timedelta(days=1)

        cache = TLRUCache[int, int, datetime](maxsize=1, ttu=ttu, timer=datetime.now)

        cache[1] = 1
        self.assertEqual(1, len(cache))
        items = cache.expire(datetime.now())
        self.assertEqual([], list(items))
        self.assertEqual(1, len(cache))
        items = cache.expire(datetime.now() + timedelta(days=1))
        self.assertEqual([(1, 1)], list(items))
        self.assertEqual(0, len(cache))

    def test_tlru_clear(self):
        def ttu(_k, _v, t):
            return t + 2

        cache = TLRUCache[int, int, int](maxsize=2, ttu=ttu, timer=Timer())

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

        # verify TLRU expiry still works after clear
        cache[42] = 42
        cache.timer.tick()
        cache.timer.tick()
        cache.timer.tick()  # past TTL
        self.assertNotIn(42, cache)

    def test_tlru_expire_budget(self):
        timer = Timer()

        def ttu(_key, value, now):
            return now + (1 if value < 100 else 1000)

        cache = TLRUCache[int, int, int](maxsize=128, ttu=ttu, timer=timer)

        # interleave many short-lived items with a few long-lived ones
        for key in range(20):
            cache[key] = key
            cache[100 + key] = 100 + key

        self.assertEqual(40, len(cache))
        timer.time = 5
        for key in range(20):
            self.assertNotIn(key, cache)
            self.assertEqual(100 + key, cache[100 + key])

        # repeatedly clean up with a very small budget and record the
        # number of heap nodes popped each round
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
            for key in range(20):
                self.assertEqual(100 + key, cache[100 + key])
            for key, _value in result:
                self.assertLess(key, 100)
            if not result.remaining:
                break
            self.assertLess(rounds, 100)

        self.assertTrue(all(n == 3 for n in checked_rounds[:-1]))
        self.assertLessEqual(checked_rounds[-1], 3)
        self.assertEqual(20, sum(checked_rounds))
        self.assertEqual(20, removed_total)
        self.assertEqual(20, len(cache))
        self.assertEqual(20, len(set(cache)))
        for key in range(20):
            self.assertEqual(100 + key, cache[100 + key])

    def test_tlru_expire_budget_stale_heap_nodes(self):
        timer = Timer()
        cache = TLRUCache[int, int, int](
            maxsize=512,
            ttu=lambda _k, v, t: t + (10 if v < 100 else 1000),
            timer=timer,
        )
        # overwrite keys repeatedly with longer-lived values: stale heap
        # nodes (expiring at t = 10) accumulate.  They are popped from
        # the heap root lazily and each pop is charged to the budget.
        cache[1] = 1
        cache[2] = 2
        for value in range(10):
            cache[1] = 100 + value
            cache[2] = 200 + value

        timer.time = 50
        # stale nodes are at the heap root (they expire at 10), but only
        # two of the 20 stale nodes are reachable by root pops; once the
        # live root is reached, bounded calls must report no more work.
        popped_total = 0
        rounds = 0
        while True:
            result = cache.expire(budget=4)
            popped_total += result.checked
            rounds += 1
            # current live values must never be removed
            self.assertEqual(109, cache[1])
            self.assertEqual(209, cache[2])
            if not result.remaining:
                break
            self.assertLess(rounds, 100)

        self.assertLessEqual(popped_total, 20)
        self.assertEqual([], cache.expire(budget=4))
        self.assertEqual(2, len(cache))

        # buried stale nodes are reclaimed by an unbounded call (the
        # same full expiration sweep used by __setitem__), which never
        # removes live values either
        self.assertEqual([], cache.expire())
        self.assertEqual(2, len(cache))
        self.assertEqual(109, cache[1])
        self.assertEqual(209, cache[2])

    def test_tlru_expire_budget_stale_node_accounting(self):
        # a stale node that is actually reached by a bounded call counts
        # against the budget, not just the returned pairs
        timer = Timer()
        cache = TLRUCache[int, int, int](
            maxsize=16,
            ttu=lambda _k, _v, t: t + 10, timer=timer
        )
        cache[1] = 1
        cache[2] = 2
        cache[1] = 3  # stale node for the old value of key 1
        timer.time = 11  # both live items are expired too

        # first popped node is the stale one: it is charged even though
        # no pair is returned
        items = cache._TLRUCache__items
        result = cache.expire(budget=1)
        self.assertEqual(1, result.checked)
        self.assertEqual([], list(result))
        self.assertTrue(result.remaining)
        # the expired-but-present live items are untouched so far
        self.assertEqual({1, 2}, set(items))

        result = cache.expire(budget=1)
        self.assertEqual(1, result.checked)
        self.assertEqual([(2, 2)], list(result))
        self.assertTrue(result.remaining)
        self.assertEqual({1}, set(items))

        result = cache.expire(budget=1)
        self.assertEqual(1, result.checked)
        self.assertEqual([(1, 3)], list(result))
        self.assertFalse(result.remaining)
        self.assertEqual(set(), set(items))

    def test_tlru_expire_budget_semantics(self):
        timer = Timer()
        cache = TLRUCache[int, int, int](
            maxsize=10, ttu=lambda _k, _v, t: t + 2, timer=timer
        )
        cache[1] = 1
        cache[2] = 2
        cache[3] = 3
        timer.time = 3

        result = cache.expire(budget=1)
        self.assertEqual([(1, 1)], list(result))
        self.assertEqual(1, result.checked)
        self.assertTrue(result.remaining)

        self.assertNotIn(2, cache)
        self.assertNotIn(3, cache)
        self.assertIsNone(cache.get(2))
        self.assertEqual("x", cache.get(3, "x"))
        self.assertEqual(set(), set(cache))

        self.assertEqual(0, len(cache))
        result = cache.expire(budget=1)
        self.assertEqual([], list(result))
        self.assertEqual(0, result.checked)
        self.assertFalse(result.remaining)

    def test_tlru_expire_budget_zero(self):
        timer = Timer()
        cache = TLRUCache[int, int, int](
            maxsize=4, ttu=lambda _k, _v, t: t + 1, timer=timer
        )
        cache[1] = 1
        cache[2] = 2
        timer.time = 2

        result = cache.expire(budget=0)
        self.assertEqual([], list(result))
        self.assertEqual(0, result.checked)
        self.assertTrue(result.remaining)
        self.assertEqual([(1, 1)], cache.expire(budget=1))
        self.assertNotIn(2, cache)

    def test_tlru_expire_budget_invalid(self):
        cache = TLRUCache[int, int, int](
            maxsize=4, ttu=lambda _k, _v, t: t + 1, timer=Timer()
        )
        with self.assertRaises(ValueError):
            cache.expire(budget=-1)

    def test_tlru_expire_budget_custom_timer(self):
        # non-monotonic/non-numeric timer values must keep the heap order
        # and the budgeted cleanup safe
        from datetime import datetime, timedelta

        calls = []

        class DateTimeTimer:
            def __init__(self):
                self.now = datetime(2024, 1, 1)

            def __call__(self):
                return self.now

        timer = DateTimeTimer()

        def ttu(_key, value, now):
            calls.append(value)
            return now + timedelta(seconds=value)

        cache = TLRUCache[int, int, datetime](
            maxsize=10, ttu=ttu, timer=timer
        )
        cache[1] = 1
        cache[2] = 100
        cache[3] = 200
        timer.now += timedelta(seconds=10)

        result = cache.expire(budget=1)
        self.assertEqual([(1, 1)], list(result))
        self.assertEqual(1, result.checked)
        self.assertEqual(100, cache[2])
        self.assertEqual(200, cache[3])
        self.assertEqual(2, len(cache))

    def test_tlru_expire_budget_insertion_between_rounds(self):
        timer = Timer()
        cache = TLRUCache[int, int, int](
            maxsize=128, ttu=lambda _k, _v, t: t + 10, timer=timer
        )
        for key in range(6):
            cache[key] = key  # distinct expiration times 10 .. 15
            timer.tick()
        timer.time = 20

        first = cache.expire(budget=2)
        self.assertEqual([(0, 0), (1, 1)], list(first))

        # inserting between two rounds runs the regular full expiration
        # sweep first; remaining expired keys are reclaimed, and the new
        # live item must stay readable
        cache[100] = 100
        self.assertNotIn(0, cache)
        self.assertNotIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertEqual(100, cache[100])

        second = cache.expire(budget=10)
        self.assertEqual([], list(second))
        self.assertFalse(second.remaining)
        self.assertEqual([100], list(cache))
        self.assertEqual(100, cache[100])
