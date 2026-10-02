import unittest

import cachetools

from . import CacheTestMixin


class CacheTest(unittest.TestCase, CacheTestMixin):
    Cache = cachetools.Cache

    def test_admission_policy_rejects_new_key(self):
        class RejectPolicy:
            def record(self, key):
                pass

            def admit(self, key, candidate):
                return False

        cache = self.Cache(maxsize=1, admission=RejectPolicy())
        cache[1] = 1

        self.assertFalse(cache._setitem(2, 2))
        self.assertEqual(cache[1], 1)
        self.assertNotIn(2, cache)

    def test_admission_policy_works_with_custom_popitem(self):
        class Policy:
            def record(self, key):
                pass

            def admit(self, key, candidate):
                return False

        class Cache(self.Cache):
            def __init__(self, maxsize, **kwargs):
                super().__init__(maxsize, **kwargs)
                self.evicted = []

            def popitem(self):
                key, value = super().popitem()
                self.evicted.append(key)
                return key, value

        cache = Cache(maxsize=1, admission=Policy())
        cache[1] = 1
        cache.__setitem__(2, 2)

        self.assertEqual(cache.evicted, [1])
        self.assertEqual(cache[1], 1)
        self.assertNotIn(2, cache)
