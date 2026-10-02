import math
import unittest
from functools import partial

from cachetools import DecayingLFUCache, LFUCache

from . import CacheTestMixin


class Timer:
    def __init__(self):
        self.time = 0

    def __call__(self):
        return self.time

    def tick(self, amount=1):
        self.time += amount


class DecayingLFUTestCache(DecayingLFUCache):
    """Constructor shape expected by the shared CacheTestMixin."""

    def __init__(self, maxsize, getsizeof=None):
        DecayingLFUCache.__init__(
            self, maxsize, decay_period=10000, getsizeof=getsizeof
        )


class DecayingLFUCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = DecayingLFUTestCache

    # ------------------------------------------------------------------
    # the motivating scenario: yesterday's hot key must lose protection
    # ------------------------------------------------------------------

    def _hotspot_story(self, make_cache):
        cache = make_cache()

        # A is accessed a million times; B and C are cold bystanders.
        cache["A"] = "a"
        for _ in range(1_000_000):
            cache["A"]
        cache["B"] = "b"
        cache["C"] = "c"
        self.assertEqual({"A", "B", "C"}, set(cache))
        return cache

    def test_old_hotspot_loses_protection_on_clock_decay(self):
        timer = Timer()
        cache = self._hotspot_story(
            lambda: DecayingLFUCache(3, decay_period=10, timer=timer)
        )

        # without any decay, a new item cannot displace A
        cache["probe"] = "p"
        self.assertIn("A", cache)
        self.assertIn("probe", cache)
        cache.pop("probe")
        cache["B"] = "b"

        evicted_round = None
        for round in range(1, 25):
            timer.tick(10)  # one more decay period has passed
            # access concentrates entirely on B and C now
            for _ in range(3):
                cache["B"]
                cache["C"]
            cache[f"probe{round}"] = round
            a_still_alive = "A" in cache
            cache.pop(f"probe{round}", None)
            for key in ("B", "C"):
                if key not in cache:
                    cache[key] = key.lower()
            if not a_still_alive:
                evicted_round = round
                break

        # after the first decay A still dominates, but after several
        # halvings (1e6 // 2**17 == 7) B and C outrank it and it dies
        self.assertIsNotNone(evicted_round)
        self.assertGreater(evicted_round, 1)
        self.assertLess(evicted_round, 25)

    def test_old_hotspot_loses_protection_on_access_decay(self):
        cache = self._hotspot_story(
            lambda: DecayingLFUCache(3, decay_period=4)
        )

        # bursts of accesses focused on B and C; every four of them
        # halve all counters, so A's million hits erode quickly
        for _ in range(8):
            cache["B"]
            cache["C"]
            cache["B"]
            cache["C"]

        cache["new"] = "n"
        self.assertIn("new", cache)
        self.assertNotIn("A", cache)

    def test_plain_lfu_keeps_old_hotspot(self):
        # control: with the unchanged LFUCache, A is never evicted
        cache = self._hotspot_story(lambda: LFUCache(3))
        for _ in range(200):
            cache["B"]
            cache["C"]
            cache["new"] = "n"
            cache.pop("new", None)
            for key in ("B", "C"):
                if key not in cache:
                    cache[key] = key.lower()
        self.assertIn("A", cache)

    # ------------------------------------------------------------------
    # stable recency order within equal-frequency buckets after decay
    # ------------------------------------------------------------------

    def test_equal_frequency_evicts_least_recently_used(self):
        timer = Timer()
        cache = DecayingLFUCache(2, decay_period=10, timer=timer)
        cache["A"] = 1
        cache["B"] = 2  # both at count 1, A is older

        cache["C"] = 3
        self.assertNotIn("A", cache)
        self.assertIn("B", cache)
        self.assertIn("C", cache)

    def test_recency_order_preserved_when_buckets_merge(self):
        timer = Timer()
        cache = DecayingLFUCache(4, decay_period=10, timer=timer)
        cache["A"] = 1
        cache["B"] = 2
        cache["C"] = 3
        for _ in range(3):
            cache["A"]  # counts: A = 4, B = C = 1
        timer.tick(10)
        cache["D"] = 4
        # decay merges B and C (1 // 2 = 0) into one bucket, A -> 2,
        # and D is inserted fresh at count 1; within the merged bucket
        # B (older) must be evicted before C, deterministically
        self.assertEqual([cache.popitem()[0] for _ in range(4)],
                         ["B", "C", "D", "A"])
        with self.assertRaises(KeyError):
            cache.popitem()


    # ------------------------------------------------------------------
    # decay triggers and counters
    # ------------------------------------------------------------------

    def test_decay_happens_after_period_accesses(self):
        cache = DecayingLFUCache(2, decay_period=4)
        cache["A"] = 1
        cache["A"]
        cache["A"]  # A count 3; three accesses recorded
        cache["B"] = 2  # fourth access: decay A 3 -> 1, B inserted at 1

        cache["C"] = 3  # tie at count 1, A is older -> A evicted
        self.assertNotIn("A", cache)
        self.assertIn("C", cache)

    def test_no_decay_without_advancing_clock(self):
        timer = Timer()
        cache = DecayingLFUCache(2, decay_period=10, timer=timer)
        cache["A"] = 1
        for _ in range(20):
            cache["A"]
        cache["B"] = 2

        cache["C"] = 3  # no time elapsed: A must stay protected
        self.assertIn("A", cache)
        self.assertIn("C", cache)  # the cold B (count 1) is evicted
        self.assertNotIn("B", cache)

    def test_counts_are_bounded_in_access_mode(self):
        period = 16
        cache = DecayingLFUCache(4, decay_period=period)
        for _ in range(100):
            cache["hot"] = 1
            for key in ("a", "b", "c"):
                if key not in cache:
                    cache[key] = key
                cache[key]
        root = cache._DecayingLFUCache__root
        counts = []
        link = root.next
        while link is not root:
            counts.append(link.count)
            link = link.next
        self.assertTrue(counts)
        self.assertLessEqual(max(counts), 2 * period)

    def test_empty_cache_decay_is_safe(self):
        cache = DecayingLFUCache(2, decay_period=2)
        for _ in range(10):
            self.assertNotIn("missing", cache)
        cache["A"] = 1
        cache["B"] = 2
        self.assertEqual(len(cache), 2)

    def test_large_clock_jump_decays_repeatedly(self):
        timer = Timer()
        cache = DecayingLFUCache(2, decay_period=10, timer=timer)
        cache["A"] = 1
        for _ in range(30):
            cache["A"]
        cache["B"] = 2
        timer.tick(1000)  # far more than enough halvings to zero out
        cache["C"] = 3  # triggers all pending decays at once

        self.assertNotIn("A", cache)
        self.assertIn("C", cache)

    # ------------------------------------------------------------------
    # index consistency
    # ------------------------------------------------------------------

    def test_delete_keeps_indexes_consistent(self):
        cache = DecayingLFUCache(4, decay_period=100)
        cache["A"] = 1
        cache["B"] = 2
        cache["A"]
        cache["C"] = 3
        cache["B"]
        cache["D"] = 4

        del cache["A"]
        del cache["C"]
        with self.assertRaises(KeyError):
            del cache["A"]

        cache["E"] = 5
        cache["F"] = 6
        self.assertEqual({"B", "D", "E", "F"}, set(cache))
        self.assertEqual(len(cache), len(cache._DecayingLFUCache__order))
        self.assertEqual(len(cache), len(cache._DecayingLFUCache__links))

        # every key reachable through exactly one bucket
        bucket_keys = set()
        root = cache._DecayingLFUCache__root
        link = root.next
        while link is not root:
            bucket_keys.update(link.keys)
            link = link.next
        self.assertEqual(set(cache), bucket_keys)

    def test_clear_resets_frequency_and_decay_state(self):
        cache = DecayingLFUCache(2, decay_period=3)
        cache["A"] = 1
        cache["A"]
        cache["A"]  # one decay already due during these accesses
        cache["B"] = 2
        cache.clear()
        self.assertEqual(0, cache.currsize)

        cache["C"] = 3
        cache["D"] = 4
        cache["C"]
        cache["E"] = 5  # C touched once post-clear -> D evicted
        self.assertIn("C", cache)
        self.assertIn("E", cache)
        self.assertNotIn("D", cache)

    def test_getsizeof_indexes_stay_consistent(self):
        cache = DecayingLFUCache(6, decay_period=100, getsizeof=len)
        cache["a"] = "x"
        cache["b"] = "yy"
        cache["c"] = "zzz"
        self.assertEqual(cache.currsize, 6)

        cache["d"] = "wwww"  # must evict enough entries
        self.assertLessEqual(cache.currsize, 6)
        self.assertEqual(
            len(cache), len(cache._DecayingLFUCache__order)
        )
        self.assertEqual(
            len(cache), len(cache._DecayingLFUCache__links)
        )

        with self.assertRaises(ValueError):
            cache["big"] = "x" * 10
        # failed insertion must not leave partial bookkeeping behind
        self.assertEqual(
            len(cache), len(cache._DecayingLFUCache__order)
        )

        cache["a"] = "longer"
        self.assertEqual(cache.currsize, sum(
            len(v) for v in cache.values()
        ))

    def test_zero_maxsize(self):
        cache = DecayingLFUCache(0, decay_period=5)
        with self.assertRaises(ValueError):
            cache["A"] = 1
        self.assertEqual(0, len(cache))
        self.assertEqual(0, cache.currsize)
        with self.assertRaises(KeyError):
            cache.popitem()

    def test_zero_maxsize_with_timer(self):
        cache = DecayingLFUCache(0, decay_period=5, timer=Timer())
        with self.assertRaises(ValueError):
            cache["A"] = 1
        self.assertEqual(0, len(cache))

    def test_invalid_decay_period(self):
        with self.assertRaises(ValueError):
            DecayingLFUCache(10, 0)
        with self.assertRaises(ValueError):
            DecayingLFUCache(10, -1)
        with self.assertRaises(ValueError):
            DecayingLFUCache(10, 0, timer=Timer())

    def test_properties(self):
        timer = Timer()
        cache = DecayingLFUCache(10, 5, timer=timer)
        self.assertEqual(cache.decay_period, 5)
        self.assertIs(cache.timer, timer)

        cache = DecayingLFUCache(10, 7)
        self.assertEqual(cache.decay_period, 7)
        self.assertIsNone(cache.timer)

    def test_infinite_maxsize_access_mode(self):
        cache = DecayingLFUCache(math.inf, decay_period=8)
        for i in range(100):
            cache[i] = i
        self.assertEqual(100, len(cache))
        for i in range(100):
            self.assertEqual(i, cache[i])

    def test_existing_lfu_cache_unchanged_constructor(self):
        cache = LFUCache(2)
        cache[1] = 1
        cache[1]
        cache[2] = 2
        cache[3] = 3
        self.assertIn(1, cache)
        self.assertTrue(2 in cache or 3 in cache)

    def test_index_consistency_under_random_operations(self):
        import random

        for mode in ("access", "clock"):
            rnd = random.Random(42)
            for _ in range(25):
                if mode == "clock":
                    timer = Timer()
                    cache = DecayingLFUCache(
                        rnd.choice([1, 2, 3, 5, 8]),
                        decay_period=rnd.choice([1, 2, 3, 7]),
                        timer=timer,
                    )
                else:
                    cache = DecayingLFUCache(
                        rnd.choice([1, 2, 3, 5, 8]),
                        decay_period=rnd.choice([1, 2, 3, 7]),
                    )
                for _ in range(150):
                    key = rnd.randrange(6)
                    op = rnd.random()
                    if op < 0.5 and key in cache:
                        cache[key]
                    elif op < 0.85:
                        cache[key] = key
                    elif key in cache:
                        del cache[key]
                    else:
                        cache.clear()
                    if mode == "clock" and rnd.random() < 0.2:
                        timer.tick(rnd.choice([1, 10]))

                    root = cache._DecayingLFUCache__root
                    links = cache._DecayingLFUCache__links
                    order = cache._DecayingLFUCache__order
                    self.assertEqual(set(cache), set(links))
                    self.assertEqual(set(cache), set(order))
                    seen = set()
                    previous = -1
                    link = root.next
                    while link is not root:
                        self.assertGreaterEqual(link.count, previous)
                        self.assertTrue(link.keys)
                        for bucket_key in link.keys:
                            self.assertIs(links[bucket_key], link)
                        seen.update(link.keys)
                        previous = link.count
                        link = link.next
                    self.assertEqual(seen, set(cache))
