import unittest

from cachetools import ARCCache, LRUCache

from . import CacheTestMixin


class ARCCacheTest(unittest.TestCase, CacheTestMixin):
    Cache = ARCCache


class UnsortableKey:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, UnsortableKey) and self.name == other.name

    def __repr__(self):
        return f"UnsortableKey({self.name!r})"


class AutoARCCache(ARCCache):
    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.misses = 0

    def __missing__(self, key):
        self.misses += 1
        self[key] = key
        return key


class AutoLRUCache(LRUCache):
    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.misses = 0

    def __missing__(self, key):
        self.misses += 1
        self[key] = key
        return key


def _arc_queues(cache):
    return (
        cache._ARCCache__recent,
        cache._ARCCache__frequent,
        cache._ARCCache__recent_ghosts,
        cache._ARCCache__frequent_ghosts,
        cache._ARCCache__target_size,
    )


def _assert_arc_state(
    test,
    cache,
    maxsize,
    recent=(),
    frequent=(),
    recent_ghosts=(),
    frequent_ghosts=(),
    target_size=0,
):
    recent_queue, frequent_queue, b1, b2, target = _arc_queues(cache)
    test.assertEqual(tuple(recent_queue), tuple(recent))
    test.assertEqual(tuple(frequent_queue), tuple(frequent))
    test.assertEqual(tuple(b1), tuple(recent_ghosts))
    test.assertEqual(tuple(b2), tuple(frequent_ghosts))
    test.assertEqual(target, target_size)
    test.assertEqual(
        cache._ARCCache__recent_size, sum(recent_queue.values())
    )
    test.assertEqual(
        cache._ARCCache__frequent_size,
        sum(frequent_queue.values()),
    )
    test.assertEqual(cache._ARCCache__recent_ghost_size, sum(b1.values()))
    test.assertEqual(
        cache._ARCCache__frequent_ghost_size, sum(b2.values())
    )
    test.assertEqual(len(cache), len(recent_queue) + len(frequent_queue))
    test.assertLessEqual(cache.currsize, maxsize)
    test.assertEqual(
        cache.currsize,
        sum(recent_queue.values()) + sum(frequent_queue.values()),
    )
    test.assertLessEqual(sum(b1.values()) + sum(b2.values()), maxsize)
    test.assertFalse(set(recent_queue) & set(frequent_queue))
    test.assertFalse(set(b1) & set(b2))
    test.assertFalse(
        (set(recent_queue) | set(frequent_queue))
        & (set(b1) | set(b2))
    )


class ARCCacheBehaviorTest(unittest.TestCase):
    def test_hot_set_scan_and_return(self):
        arc = AutoARCCache(maxsize=4)
        lru = AutoLRUCache(maxsize=4)
        hot = (0, 1)
        scan = (4, 5, 6, 7)

        for index, key in enumerate((0, 1, 2, 3)):
            self.assertEqual(arc[key], key)
            self.assertEqual(lru[key], key)
            _assert_arc_state(
                self, arc, 4, recent=(0, 1, 2, 3)[: index + 1]
            )

        for _ in range(3):
            for key in hot:
                self.assertEqual(arc[key], key)
                self.assertEqual(lru[key], key)
        _assert_arc_state(self, arc, 4, recent=(2, 3), frequent=hot)

        scan_states = (
            ((3, 4), hot, (2,), ()),
            ((4, 5), hot, (2, 3), ()),
            ((5, 6), hot, (2, 3, 4), ()),
            ((6, 7), hot, (2, 3, 4, 5), ()),
        )
        for key in scan:
            self.assertEqual(arc[key], key)
            self.assertEqual(lru[key], key)
            recent, frequent, recent_ghosts, frequent_ghosts = scan_states[
                key - 4
            ]
            _assert_arc_state(
                self,
                arc,
                4,
                recent=recent,
                frequent=frequent,
                recent_ghosts=recent_ghosts,
                frequent_ghosts=frequent_ghosts,
            )
            self.assertEqual(arc.currsize, 4)
            self.assertEqual(
                len(_arc_queues(arc)[2]) + len(_arc_queues(arc)[3]),
                key - 3,
            )

        self.assertTrue(all(key not in lru for key in hot))
        self.assertTrue(all(key in arc for key in hot))
        arc_misses = arc.misses
        lru_misses = lru.misses

        for key in hot:
            self.assertNotIn(key, lru)
            self.assertEqual(arc[key], key)
            self.assertEqual(lru[key], key)

        _assert_arc_state(
            self,
            arc,
            4,
            recent=(6, 7),
            frequent=hot,
            recent_ghosts=(2, 3, 4, 5),
        )
        self.assertEqual(set(lru), {0, 1, 6, 7})
        self.assertEqual(arc.misses - arc_misses, 0)
        self.assertEqual(lru.misses - lru_misses, 2)

    def test_ghost_hits_adjust_target_partition(self):
        cache = AutoARCCache(maxsize=4)

        for key in (1, 2, 3):
            cache[key] = key
        cache[1]
        for key in (4, 5, 6, 7):
            cache[key] = key

        _assert_arc_state(
            self,
            cache,
            4,
            recent=(5, 6, 7),
            frequent=(1,),
            recent_ghosts=(2, 3, 4),
        )

        self.assertEqual(cache[4], 4)
        _assert_arc_state(
            self,
            cache,
            4,
            recent=(6, 7),
            frequent=(1, 4),
            recent_ghosts=(2, 3, 5),
            target_size=1,
        )

    def test_weighted_ghost_hits_use_ghost_weight_units(self):
        class WeightedARCCache(ARCCache[str, int]):
            def __missing__(self, key):
                size = {"a": 2, "b": 3, "c": 5, "d": 5}[key]
                self[key] = size
                return size

        cache = WeightedARCCache(maxsize=10, getsizeof=lambda value: value)

        cache["a"] = 2
        cache["b"] = 3
        cache["c"] = 5
        cache["c"]
        cache["d"] = 5

        _assert_arc_state(
            self,
            cache,
            10,
            recent=("d",),
            frequent=("c",),
            recent_ghosts=("a", "b"),
        )

        self.assertEqual(cache["b"], 3)
        _assert_arc_state(
            self,
            cache,
            10,
            recent=(),
            frequent=("c", "b"),
            recent_ghosts=("a", "d"),
            target_size=3,
        )

        self.assertEqual(cache["a"], 2)
        _assert_arc_state(
            self,
            cache,
            10,
            recent=(),
            frequent=("c", "b", "a"),
            recent_ghosts=("d",),
            target_size=5,
        )

    def test_frequent_ghost_hit_decreases_target_partition(self):
        cache = AutoARCCache(maxsize=4)

        for key in (1, 2, 3, 4):
            cache[key] = key
        cache[1]
        for key in (5, 6, 7):
            cache[key] = key
        for key in (4, 2, 3, 5, 6):
            cache[key] = key

        _assert_arc_state(
            self,
            cache,
            4,
            recent=(7,),
            frequent=(3, 5, 6),
            frequent_ghosts=(1, 4, 2),
            target_size=4,
        )

        self.assertEqual(cache[1], 1)
        _assert_arc_state(
            self,
            cache,
            4,
            recent=(7,),
            frequent=(5, 6, 1),
            frequent_ghosts=(4, 2, 3),
            target_size=3,
        )

    def test_overwrite_delete_clear_and_oversized_value(self):
        cache = ARCCache[str, int](
            maxsize=10, getsizeof=lambda value: value
        )
        cache["a"] = 2
        cache["b"] = 3

        cache["b"] = 8
        _assert_arc_state(self, cache, 10, recent=("a",), frequent=("b",))
        self.assertEqual(cache.currsize, 10)

        cache["b"]
        cache["b"] = 4
        _assert_arc_state(self, cache, 10, recent=("a",), frequent=("b",))
        self.assertEqual(cache.currsize, 6)

        with self.assertRaises(ValueError):
            cache["b"] = 11
        with self.assertRaises(ValueError):
            cache["a"] = 11
        _assert_arc_state(self, cache, 10, recent=("a",), frequent=("b",))

        cache["a"] = 2
        _assert_arc_state(self, cache, 10, frequent=("b", "a"))

        del cache["b"]
        _assert_arc_state(self, cache, 10, frequent=("a",))
        del cache["a"]
        _assert_arc_state(self, cache, 10)

        cache["x"] = 1
        cache.popitem()
        _assert_arc_state(self, cache, 10, recent_ghosts=("x",))
        cache.clear()
        _assert_arc_state(self, cache, 10)

    def test_unsortable_keys_and_ghosts_store_no_values(self):
        cache = AutoARCCache(maxsize=2)
        first = UnsortableKey("first")
        second = UnsortableKey("second")
        third = UnsortableKey("third")
        first_value = object()
        second_value = object()
        third_value = object()

        cache[first] = first_value
        cache[second] = second_value
        cache[first]
        cache[third] = third_value

        recent, frequent, _, frequent_ghosts, _ = _arc_queues(cache)
        self.assertEqual(tuple(recent), (third,))
        self.assertEqual(tuple(frequent), (first,))
        self.assertEqual(tuple(frequent_ghosts), ())

        recent_ghosts = _arc_queues(cache)[2]
        self.assertEqual(tuple(recent_ghosts), (second,))
        self.assertEqual(recent_ghosts[second], 1)
        self.assertIsNot(recent_ghosts[second], second_value)

        self.assertEqual(cache[second], second)
        recent, frequent, _, frequent_ghosts, _ = _arc_queues(cache)
        self.assertEqual(tuple(recent), (third,))
        self.assertEqual(tuple(frequent), (second,))
        self.assertEqual(tuple(frequent_ghosts), (first,))
        self.assertIsNot(frequent_ghosts[first], first_value)


if __name__ == "__main__":
    unittest.main()
