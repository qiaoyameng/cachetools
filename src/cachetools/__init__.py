"""Extensible memoizing collections and decorators."""

__all__ = (
    "Cache",
    "FIFOCache",
    "FrequencySketch",
    "LFUCache",
    "LRUCache",
    "RRCache",
    "TinyLFU",
    "TLRUCache",
    "TTLCache",
    "cached",
    "cachedmethod",
)

__version__ = "7.2.0"

import collections
import collections.abc
import functools
import heapq
import math
import random
import time

from . import keys


def _fnv1a(data, seed):
    value = seed
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def _stable_hashes(key):
    data = repr(key).encode("utf-8", "surrogatepass")
    return (
        _fnv1a(data, 0x811C9DC5),
        _fnv1a(data, 0x517CC1B7) | 1,
    )


class FrequencySketch:
    """A bounded Count-Min sketch for approximate key frequencies."""

    __slots__ = (
        "_counters",
        "_depth",
        "_hasher",
        "_sample_count",
        "_sample_size",
        "_width",
    )

    def __init__(self, width, depth=4, *, hasher=None, sample_size=None):
        width = int(width)
        depth = int(depth)
        if width <= 0:
            raise ValueError("width must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if sample_size is None:
            sample_size = width * 10
        sample_size = int(sample_size)
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        self._width = width
        self._depth = depth
        self._hasher = hasher or _stable_hashes
        self._sample_size = sample_size
        self._sample_count = 0
        self._counters = bytearray(width * depth)

    @property
    def width(self):
        """The number of counters per sketch row."""
        return self._width

    @property
    def depth(self):
        """The number of independent counter rows."""
        return self._depth

    @property
    def sample_size(self):
        """The number of additions between two aging operations."""
        return self._sample_size

    @property
    def sample_count(self):
        """The number of additions since the last aging operation."""
        return self._sample_count

    @property
    def memory_size(self):
        """The fixed number of bytes used by counters."""
        return len(self._counters)

    def add(self, key):
        """Record one access to `key`."""
        for position in self._positions(key):
            if self._counters[position] < 15:
                self._counters[position] += 1
        self._sample_count += 1
        if self._sample_count >= self._sample_size:
            self.age()

    def frequency(self, key):
        """Return the estimated access frequency for `key`."""
        counters = self._counters
        return min(counters[position] for position in self._positions(key))

    def estimate(self, key):
        """Return the estimated access frequency for `key`."""
        return self.frequency(key)

    def age(self):
        """Halve all counters and start a new sampling window."""
        counters = self._counters
        for position, value in enumerate(counters):
            counters[position] = value >> 1
        self._sample_count = 0

    def clear(self):
        """Reset all counters and sampling state."""
        self._counters[:] = b"\x00" * len(self._counters)
        self._sample_count = 0

    def _positions(self, key):
        first, second = self._hasher(key)
        width = self._width
        return [(first + row * second) % width for row in range(self._depth)]


class TinyLFU:
    """TinyLFU admission policy based on a frequency sketch."""

    __slots__ = ("_sketch",)

    def __init__(
        self, capacity=128, *, depth=4, hasher=None, sample_size=None
    ):
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError("capacity must be a finite non-negative number")
        width = max(1, math.ceil(capacity))
        self._sketch = FrequencySketch(
            width, depth, hasher=hasher, sample_size=sample_size
        )

    @property
    def sketch(self):
        """The bounded frequency sketch used by this policy."""
        return self._sketch

    def record(self, key):
        """Record an access to `key`."""
        self._sketch.add(key)

    def frequency(self, key):
        """Return the estimated access frequency for `key`."""
        return self._sketch.frequency(key)

    def estimate(self, key):
        """Return the estimated access frequency for `key`."""
        return self.frequency(key)

    def admit(self, key, candidate):
        """Return whether `key` may replace `candidate`."""
        return self._sketch.frequency(key) > self._sketch.frequency(candidate)

    def clear(self):
        """Reset the admission policy."""
        self._sketch.clear()


class _DefaultSize:
    """A minimal "fake" dict that returns a constant size 1 for any key."""

    __slots__ = ()

    def __getitem__(self, _key):
        return 1

    def __setitem__(self, _key, _value):
        pass

    def pop(self, _key):
        return 1

    def clear(self):
        pass


class Cache(collections.abc.MutableMapping):
    """Mutable mapping to serve as a simple cache or cache base class."""

    __marker = object()

    __size = _DefaultSize()

    def __init__(self, maxsize, getsizeof=None, *, admission=None):
        if maxsize < 0:
            raise ValueError("maxsize must be non-negative")
        if getsizeof:
            self.getsizeof = getsizeof
        if self.getsizeof is not Cache.getsizeof:
            self.__size = {}
        self.__data = {}
        self.__currsize = 0
        self.__maxsize = maxsize
        self.__admission = admission

    def __repr__(self):
        return "%s(%s, maxsize=%r, currsize=%r)" % (
            type(self).__name__,
            repr(self.__data),
            self.__maxsize,
            self.__currsize,
        )

    def __getitem__(self, key):
        if key not in self.__data:
            return self.__missing__(key)
        value = self.__data[key]
        if self.__admission is not None:
            self.__admission.record(key)
        return value

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def _setitem(self, key, value):
        maxsize = self.__maxsize
        size = self.getsizeof(value)
        if size < 0:
            raise ValueError("value size must be non-negative")
        if size > maxsize:
            raise ValueError("value too large")
        admission = self.__admission
        if admission is not None:
            admission.record(key)
        if key not in self.__data:
            diffsize = size
            if self.__currsize + diffsize > maxsize:
                candidates, evicted, restore = self._eviction_plan(
                    self.__currsize + diffsize - maxsize
                )
                if admission is not None:
                    for candidate in candidates:
                        if not admission.admit(key, candidate):
                            if restore:
                                self._restore_evicted(evicted)
                            return False
                if restore:
                    self._commit_evicted(evicted)
                for candidate in candidates:
                    del self[candidate]
        else:
            diffsize = size - self.__size[key]
            while self.__currsize + diffsize > maxsize:
                self.popitem()
                if key not in self.__data:
                    diffsize = size
        self.__data[key] = value
        self.__size[key] = size
        self.__currsize += diffsize
        return True

    def _eviction_plan(self, required_size):
        builtin_popitems = (
            Cache.popitem,
            FIFOCache.popitem,
            LFUCache.popitem,
            LRUCache.popitem,
            RRCache.popitem,
            TTLCache.popitem,
            TLRUCache.popitem,
        )
        if type(self).popitem not in builtin_popitems:
            return self._custom_eviction_plan(required_size)
        candidates = []
        released_size = 0
        selected = set()
        while released_size < required_size:
            try:
                candidate = self._eviction_candidate(exclude=selected)
            except (IndexError, StopIteration):
                raise KeyError(
                    "%s is empty" % type(self).__name__
                ) from None
            if candidate not in self or candidate in selected:
                raise KeyError("%s is empty" % type(self).__name__) from None
            selected.add(candidate)
            candidates.append(candidate)
            released_size += self.__size[candidate]
        return candidates, [], False

    def _custom_eviction_plan(self, required_size):
        evicted = []
        candidates = []
        released_size = 0
        while released_size < required_size:
            previous_size = self.__currsize
            try:
                candidate, value = self.popitem()
            except KeyError:
                self._restore_evicted(evicted)
                raise
            size = previous_size - self.__currsize
            evicted.append((candidate, value, size))
            candidates.append(candidate)
            released_size += size
        return candidates, evicted, True

    def _restore_evicted(self, evicted):
        for candidate, value, _size in reversed(evicted):
            self.__setitem__(candidate, value)

    def _commit_evicted(self, evicted):
        pass

    def _eviction_candidate(self, exclude=()):
        for candidate in self:
            if candidate not in exclude:
                return candidate
        raise StopIteration

    def _peek(self, key):
        return self.__data[key]

    def _discard(self, key):
        value = self.__data[key]
        type(self).__delitem__(self, key)
        return value

    def popitem(self):
        """Remove and return an arbitrary `(key, value)` pair."""
        try:
            key = self._eviction_candidate()
        except (IndexError, StopIteration):
            raise KeyError("%s is empty" % type(self).__name__) from None
        return (key, self._discard(key))

    def __delitem__(self, key):
        size = self.__size.pop(key)
        del self.__data[key]
        self.__currsize -= size

    def __contains__(self, key):
        return key in self.__data

    def __missing__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(self.__data)

    def __len__(self):
        return len(self.__data)

    # Note that we cannot simply inherit get(), pop() and setdefault()
    # from MutableMapping, since these rely on __getitem__ throwing a
    # KeyError on cache miss.  This is not the case if __missing__ is
    # implemented for a Cache subclass, so we have to roll our own,
    # somewhat less elegant versions.

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def pop(self, key, default=__marker):
        if key in self:
            value = self[key]
            del self[key]
        elif default is self.__marker:
            raise KeyError(key)
        else:
            value = default
        return value

    def setdefault(self, key, default=None):
        if key in self:
            value = self[key]
        else:
            self[key] = value = default
        return value

    # Although the MutableMapping.clear() default implementation works
    # perfectly well, it calls popitem() in a loop until the cache is
    # empty, resulting in O(n) complexity.  For large caches, this
    # becomes a significant performance bottleneck, so we provide an
    # optimized version for each Cache subclass.

    def clear(self):
        self.__data.clear()
        self.__size.clear()
        self.__currsize = 0

    @property
    def maxsize(self):
        """The maximum size of the cache."""
        return self.__maxsize

    @property
    def currsize(self):
        """The current size of the cache."""
        return self.__currsize

    @staticmethod
    def getsizeof(value):
        """Return the size of a cache element's value."""
        return 1


class FIFOCache(Cache):
    """First In First Out (FIFO) cache implementation."""

    def __init__(self, maxsize, getsizeof=None, *, admission=None):
        Cache.__init__(self, maxsize, getsizeof, admission=admission)
        self.__order = collections.OrderedDict()

    def __setitem__(self, key, value):
        Cache.__setitem__(self, key, value)
        self.__track(key)

    def _setitem(self, key, value, cache_setitem=Cache._setitem):
        if not cache_setitem(self, key, value):
            return False
        self.__track(key)
        return True

    def __track(self, key):
        if key in self.__order:
            self.__order.move_to_end(key)
        else:
            self.__order[key] = None

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        del self.__order[key]

    def popitem(self):
        """Remove and return the `(key, value)` pair first inserted."""
        try:
            key = next(iter(self.__order))
        except StopIteration:
            raise KeyError("%s is empty" % type(self).__name__) from None
        else:
            return (key, self._discard(key))

    def _eviction_candidate(self, exclude=()):
        for candidate in self.__order:
            if candidate not in exclude:
                return candidate
        raise StopIteration

    def clear(self):
        Cache.clear(self)
        self.__order.clear()


class LFUCache(Cache):
    """Least Frequently Used (LFU) cache implementation."""

    class _Link:
        __slots__ = ("count", "keys", "next", "prev")

        def __init__(self, count):
            self.count = count
            self.keys = set()

        def unlink(self):
            next = self.next
            prev = self.prev
            prev.next = next
            next.prev = prev

    def __init__(self, maxsize, getsizeof=None, *, admission=None):
        Cache.__init__(self, maxsize, getsizeof, admission=admission)
        self.__root = root = LFUCache._Link(0)  # sentinel
        root.prev = root.next = root
        self.__links = {}

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        value = cache_getitem(self, key)
        if key in self:  # __missing__ may not store item
            self.__touch(key)
        return value

    def __setitem__(self, key, value):
        Cache.__setitem__(self, key, value)

    def _setitem(self, key, value, cache_setitem=Cache._setitem):
        if not cache_setitem(self, key, value):
            return False
        if key in self.__links:
            self.__touch(key)
            return True
        root = self.__root
        link = root.next
        if link.count != 1:
            link = LFUCache._Link(1)
            link.next = root.next
            root.next = link.next.prev = link
            link.prev = root
        link.keys.add(key)
        self.__links[key] = link

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        link = self.__links.pop(key)
        link.keys.remove(key)
        if not link.keys:
            link.unlink()

    def popitem(self):
        """Remove and return the `(key, value)` pair least frequently used."""
        root = self.__root
        curr = root.next
        if curr is root:
            raise KeyError("%s is empty" % type(self).__name__) from None
        key = next(iter(curr.keys))  # remove an arbitrary element
        return (key, self._discard(key))

    def _eviction_candidate(self, exclude=()):
        root = self.__root
        link = root.next
        while link is not root:
            for candidate in link.keys:
                if candidate not in exclude:
                    return candidate
            link = link.next
        raise StopIteration

    def clear(self):
        Cache.clear(self)
        root = self.__root
        root.prev = root.next = root
        self.__links.clear()

    def __touch(self, key):
        """Increment use count"""
        link = self.__links[key]
        curr = link.next
        if curr.count != link.count + 1:
            if len(link.keys) == 1:
                link.count += 1
                return
            curr = LFUCache._Link(link.count + 1)
            curr.next = link.next
            link.next = curr.next.prev = curr
            curr.prev = link
        curr.keys.add(key)
        link.keys.remove(key)
        if not link.keys:
            link.unlink()
        self.__links[key] = curr


class LRUCache(Cache):
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, maxsize, getsizeof=None, *, admission=None):
        Cache.__init__(self, maxsize, getsizeof, admission=admission)
        self.__order = collections.OrderedDict()

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        value = cache_getitem(self, key)
        if key in self:  # __missing__ may not store item
            self.__touch(key)
        return value

    def __setitem__(self, key, value):
        Cache.__setitem__(self, key, value)
        self.__touch(key)

    def _setitem(self, key, value, cache_setitem=Cache._setitem):
        if cache_setitem(self, key, value):
            self.__touch(key)
            return True
        return False

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        del self.__order[key]

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used."""
        try:
            key = next(iter(self.__order))
        except StopIteration:
            raise KeyError("%s is empty" % type(self).__name__) from None
        else:
            return (key, self._discard(key))

    def _eviction_candidate(self, exclude=()):
        for candidate in self.__order:
            if candidate not in exclude:
                return candidate
        raise StopIteration

    def clear(self):
        Cache.clear(self)
        self.__order.clear()

    def __touch(self, key):
        """Mark as recently used"""
        try:
            self.__order.move_to_end(key)
        except KeyError:
            self.__order[key] = None


class RRCache(Cache):
    """Random Replacement (RR) cache implementation."""

    def __init__(
        self,
        maxsize,
        choice=random.choice,
        getsizeof=None,
        *,
        admission=None,
    ):
        Cache.__init__(self, maxsize, getsizeof, admission=admission)
        self.__choice = choice
        self.__index = {}
        self.__keys = []

    @property
    def choice(self):
        """The `choice` function used by the cache."""
        return self.__choice

    def __setitem__(self, key, value):
        Cache.__setitem__(self, key, value)
        self.__track(key)

    def _setitem(self, key, value, cache_setitem=Cache._setitem):
        if not cache_setitem(self, key, value):
            return False
        self.__track(key)
        return True

    def __track(self, key):
        if key not in self.__index:
            self.__index[key] = len(self.__keys)
            self.__keys.append(key)

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        index = self.__index.pop(key)
        if index != len(self.__keys) - 1:
            last = self.__keys[-1]
            self.__keys[index] = last
            self.__index[last] = index
        self.__keys.pop()

    def popitem(self):
        """Remove and return a random `(key, value)` pair."""
        try:
            key = self.__choice(self.__keys)
        except IndexError:
            raise KeyError("%s is empty" % type(self).__name__) from None
        else:
            return (key, self._discard(key))

    def _eviction_candidate(self, exclude=()):
        available = [key for key in self.__keys if key not in exclude]
        return self.__choice(available)

    def clear(self):
        Cache.clear(self)
        self.__index.clear()
        del self.__keys[:]


class _TimedCache(Cache):
    """Base class for time aware cache implementations."""

    class _Timer:
        def __init__(self, timer):
            self.__timer = timer
            self.__nesting = 0

        def __call__(self):
            if self.__nesting == 0:
                return self.__timer()
            else:
                return self.__time

        def __enter__(self):
            if self.__nesting == 0:
                self.__time = time = self.__timer()
            else:
                time = self.__time
            self.__nesting += 1
            return time

        def __exit__(self, *exc):
            self.__nesting -= 1

        def __reduce__(self):
            return _TimedCache._Timer, (self.__timer,)

        def __getattr__(self, name):
            return getattr(self.__timer, name)

    def __init__(self, maxsize, timer, getsizeof=None, *, admission=None):
        Cache.__init__(self, maxsize, getsizeof, admission=admission)
        self.__timer = _TimedCache._Timer(timer)

    def __repr__(self, cache_repr=Cache.__repr__):
        with self.__timer as time:
            self.expire(time)
            return cache_repr(self)

    def __len__(self, cache_len=Cache.__len__):
        with self.__timer as time:
            self.expire(time)
            return cache_len(self)

    @property
    def currsize(self):
        with self.__timer as time:
            self.expire(time)
            return super().currsize

    @property
    def timer(self):
        """The timer function used by the cache."""
        return self.__timer

    def get(self, *args, **kwargs):
        with self.__timer:
            return Cache.get(self, *args, **kwargs)

    def pop(self, *args, **kwargs):
        with self.__timer:
            return Cache.pop(self, *args, **kwargs)

    def setdefault(self, *args, **kwargs):
        with self.__timer:
            return Cache.setdefault(self, *args, **kwargs)

    def clear(self):
        # Subclasses must override to also reset their own time-tracking
        # structures; we do not call expire() here since clear() should
        # be O(1) regardless of cache contents.
        Cache.clear(self)

    def expire(self, time=None):  # pragma: no cover
        raise NotImplementedError


class TTLCache(_TimedCache):
    """LRU Cache implementation with per-item time-to-live (TTL) value."""

    class _Link:
        __slots__ = ("expires", "key", "next", "prev")

        def __init__(self, key=None, expires=None):
            self.key = key
            self.expires = expires

        def __reduce__(self):
            return TTLCache._Link, (self.key, self.expires)

        def unlink(self):
            next = self.next
            prev = self.prev
            prev.next = next
            next.prev = prev

    def __init__(
        self,
        maxsize,
        ttl,
        timer=time.monotonic,
        getsizeof=None,
        *,
        admission=None,
    ):
        _TimedCache.__init__(self, maxsize, timer, getsizeof, admission=admission)
        self.__root = root = TTLCache._Link()
        root.prev = root.next = root
        self.__links = collections.OrderedDict()
        self.__ttl = ttl

    def __contains__(self, key):
        try:
            link = self.__links[key]  # no reordering
        except KeyError:
            return False
        else:
            return self.timer() < link.expires

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        try:
            link = self.__getlink(key)
        except KeyError:
            expired = False
        else:
            expired = not (self.timer() < link.expires)
        if expired:
            return self.__missing__(key)
        else:
            return cache_getitem(self, key)

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def _setitem(self, key, value, cache_setitem=Cache._setitem):
        with self.timer as time:
            self.expire(time)
            stored = cache_setitem(self, key, value)
        if not stored:
            return False
        try:
            link = self.__getlink(key)
        except KeyError:
            self.__links[key] = link = TTLCache._Link(key)
        else:
            link.unlink()
        link.expires = time + self.__ttl
        link.next = root = self.__root
        link.prev = prev = root.prev
        prev.next = root.prev = link
        return True

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        link = self.__links.pop(key)
        link.unlink()
        if not (self.timer() < link.expires):
            raise KeyError(key)

    def __iter__(self):
        root = self.__root
        curr = root.next
        while curr is not root:
            # "freeze" time for iterator access
            with self.timer as time:
                if time < curr.expires:
                    yield curr.key
            curr = curr.next

    def __setstate__(self, state):
        self.__dict__.update(state)
        root = self.__root
        root.prev = root.next = root
        for link in sorted(self.__links.values(), key=lambda obj: obj.expires):
            link.next = root
            link.prev = prev = root.prev
            prev.next = root.prev = link
        self.expire(self.timer())

    @property
    def ttl(self):
        """The time-to-live value of the cache's items."""
        return self.__ttl

    def expire(self, time=None):
        """Remove expired items from the cache and return an iterable of the
        expired `(key, value)` pairs.

        """
        if time is None:
            time = self.timer()
        root = self.__root
        curr = root.next
        links = self.__links
        expired = []
        cache_delitem = Cache.__delitem__
        while curr is not root and not (time < curr.expires):
            expired.append((curr.key, self._peek(curr.key)))
            cache_delitem(self, curr.key)
            del links[curr.key]
            next = curr.next
            curr.unlink()
            curr = next
        return expired

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used that
        has not already expired.

        """
        with self.timer as time:
            self.expire(time)
            try:
                key = next(iter(self.__links))
            except StopIteration:
                raise KeyError("%s is empty" % type(self).__name__) from None
            else:
                return (key, self._discard(key))

    def _eviction_candidate(self, exclude=()):
        with self.timer as time:
            self.expire(time)
            for candidate in self.__links:
                if candidate not in exclude:
                    return candidate
        raise StopIteration

    def clear(self):
        _TimedCache.clear(self)
        root = self.__root
        root.prev = root.next = root
        self.__links.clear()

    def __getlink(self, key):
        value = self.__links[key]
        self.__links.move_to_end(key)
        return value


class TLRUCache(_TimedCache):
    """Time aware Least Recently Used (TLRU) cache implementation."""

    __HEAP_CLEANUP_FACTOR = 2  # clean up the heap if size > N * len(items)

    @functools.total_ordering
    class _Item:
        __slots__ = ("expires", "key", "removed")

        def __init__(self, key=None, expires=None):
            self.key = key
            self.expires = expires
            self.removed = False

        def __lt__(self, other):
            return self.expires < other.expires

    def __init__(
        self,
        maxsize,
        ttu,
        timer=time.monotonic,
        getsizeof=None,
        *,
        admission=None,
    ):
        _TimedCache.__init__(self, maxsize, timer, getsizeof, admission=admission)
        self.__items = collections.OrderedDict()
        self.__order = []
        self.__ttu = ttu

    def __contains__(self, key):
        try:
            item = self.__items[key]  # no reordering
        except KeyError:
            return False
        else:
            return self.timer() < item.expires

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        try:
            item = self.__getitem(key)
        except KeyError:
            expired = False
        else:
            expired = not (self.timer() < item.expires)
        if expired:
            return self.__missing__(key)
        else:
            return cache_getitem(self, key)

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def _setitem(self, key, value, cache_setitem=Cache._setitem):
        with self.timer as time:
            self.expire(time)
            expires = self.__ttu(key, value, time)
            if not (time < expires):
                # updating an existing item with an already expired
                # one should remove the existing item
                self.__delitem(key)
                return True
            if not cache_setitem(self, key, value):
                return False
        # removing an existing item would break the heap structure, so
        # only mark it as removed for now
        try:
            self.__getitem(key).removed = True
        except KeyError:
            pass
        self.__items[key] = item = TLRUCache._Item(key, expires)
        heapq.heappush(self.__order, item)
        return True

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        with self.timer as time:
            # no self.expire() for performance reasons, e.g. self.clear() [#67]
            cache_delitem(self, key)
        item = self.__items.pop(key)
        item.removed = True
        if not (time < item.expires):
            raise KeyError(key)

    def __iter__(self):
        for curr in self.__order:
            # "freeze" time for iterator access
            with self.timer as time:
                if time < curr.expires and not curr.removed:
                    yield curr.key

    @property
    def ttu(self):
        """The local time-to-use function used by the cache."""
        return self.__ttu

    def expire(self, time=None):
        """Remove expired items from the cache and return an iterable of the
        expired `(key, value)` pairs.

        """
        if time is None:
            time = self.timer()
        items = self.__items
        order = self.__order
        # clean up the heap if too many items are marked as removed
        if len(order) > len(items) * self.__HEAP_CLEANUP_FACTOR:
            self.__order = order = [item for item in order if not item.removed]
            heapq.heapify(order)
        expired = []
        cache_delitem = Cache.__delitem__
        while order and (order[0].removed or not (time < order[0].expires)):
            item = heapq.heappop(order)
            if not item.removed:
                expired.append((item.key, self._peek(item.key)))
                cache_delitem(self, item.key)
                del items[item.key]
        return expired

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used that
        has not already expired.

        """
        with self.timer as time:
            self.expire(time)
            try:
                key = next(iter(self.__items))
            except StopIteration:
                raise KeyError("%s is empty" % type(self).__name__) from None
            else:
                return (key, self._discard(key))

    def _eviction_candidate(self, exclude=()):
        with self.timer as time:
            self.expire(time)
            for candidate in self.__items:
                if candidate not in exclude:
                    return candidate
        raise StopIteration

    def clear(self):
        _TimedCache.clear(self)
        self.__items.clear()
        del self.__order[:]

    def __getitem(self, key):
        value = self.__items[key]
        self.__items.move_to_end(key)
        return value

    def __delitem(self, key, cache_delitem=Cache.__delitem__):
        try:
            self.__items.pop(key).removed = True
        except KeyError:
            pass
        else:
            cache_delitem(self, key)


# note that the runtime __name__ is "CacheInfo", as in stdlib:
# https://github.com/python/cpython/blob/3.14/Lib/functools.py#L520
_CacheInfo = collections.namedtuple(
    "CacheInfo", ["hits", "misses", "maxsize", "currsize"]
)


def cached(cache, key=keys.hashkey, lock=None, condition=None, info=False):
    """Decorator to wrap a function with a memoizing callable that saves
    results in a cache.

    """
    from ._cached import _wrapper

    def decorator(func):
        if info:
            if isinstance(cache, Cache):

                def make_info(hits, misses):
                    return _CacheInfo(hits, misses, cache.maxsize, cache.currsize)

            elif isinstance(cache, collections.abc.Mapping):

                def make_info(hits, misses):
                    return _CacheInfo(hits, misses, None, len(cache))

            else:

                def make_info(hits, misses):
                    return _CacheInfo(hits, misses, 0, 0)

            return _wrapper(func, cache, key, lock, condition, info=make_info)
        else:
            return _wrapper(func, cache, key, lock, condition)

    return decorator


def cachedmethod(cache, key=keys.methodkey, lock=None, condition=None, info=False):
    """Decorator to wrap a method with a memoizing callable that saves
    results in a cache.

    """
    from ._cachedmethod import _wrapper

    def decorator(method):
        if info:

            def make_info(cache, hits, misses):
                if isinstance(cache, Cache):
                    return _CacheInfo(hits, misses, cache.maxsize, cache.currsize)
                elif isinstance(cache, collections.abc.Mapping):
                    return _CacheInfo(hits, misses, None, len(cache))
                else:
                    raise TypeError("cache(self) must return a mutable mapping")

            return _wrapper(method, cache, key, lock, condition, info=make_info)
        else:
            return _wrapper(method, cache, key, lock, condition)

    return decorator
