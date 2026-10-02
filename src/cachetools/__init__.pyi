import random
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from contextlib import AbstractContextManager
from typing import (
    Any,
    Concatenate,
    Final,
    Generic,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
    type_check_only,
)

__all__: Final = (
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
__version__: str

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_TT = TypeVar("_TT", default=float)
_KT2 = TypeVar("_KT2")
_VT2 = TypeVar("_VT2")

@type_check_only
class _AdmissionPolicy(Protocol[_KT]):
    def record(self, key: _KT) -> None: ...
    def admit(self, key: _KT, candidate: _KT) -> bool: ...

class FrequencySketch(Generic[_KT]):
    def __init__(
        self,
        width: int,
        depth: int = 4,
        *,
        hasher: Callable[[_KT], tuple[int, int]] | None = None,
        sample_size: int | None = None,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def depth(self) -> int: ...
    @property
    def sample_size(self) -> int: ...
    @property
    def sample_count(self) -> int: ...
    @property
    def memory_size(self) -> int: ...
    def add(self, key: _KT) -> None: ...
    def frequency(self, key: _KT) -> int: ...
    def estimate(self, key: _KT) -> int: ...
    def age(self) -> None: ...
    def clear(self) -> None: ...

class TinyLFU(Generic[_KT]):
    def __init__(
        self,
        capacity: float = 128,
        *,
        depth: int = 4,
        hasher: Callable[[_KT], tuple[int, int]] | None = None,
        sample_size: int | None = None,
    ) -> None: ...
    @property
    def sketch(self) -> FrequencySketch[_KT]: ...
    def record(self, key: _KT) -> None: ...
    def frequency(self, key: _KT) -> int: ...
    def estimate(self, key: _KT) -> int: ...
    def admit(self, key: _KT, candidate: _KT) -> bool: ...
    def clear(self) -> None: ...

class Cache(MutableMapping[_KT, _VT]):
    def __init__(
        self,
        maxsize: float,
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...
    def __getitem__(self, key: _KT) -> _VT: ...
    def __setitem__(self, key: _KT, value: _VT) -> None: ...
    def __delitem__(self, key: _KT) -> None: ...
    def __missing__(self, key: _KT) -> _VT: ...
    def __iter__(self) -> Iterator[_KT]: ...
    def __len__(self) -> int: ...
    @overload
    def pop(self, key: _KT) -> _VT: ...
    @overload
    def pop(self, key: _KT, default: _VT | _T) -> _VT | _T: ...
    def setdefault(self, key: _KT, default: _VT | None = None) -> _VT: ...
    @property
    def maxsize(self) -> float: ...
    @property
    def currsize(self) -> float: ...
    @staticmethod
    def getsizeof(value: _VT) -> float: ...

class FIFOCache(Cache[_KT, _VT]):
    def __init__(
        self,
        maxsize: float,
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...

class LFUCache(Cache[_KT, _VT]):
    def __init__(
        self,
        maxsize: float,
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...

class LRUCache(Cache[_KT, _VT]):
    def __init__(
        self,
        maxsize: float,
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...

class RRCache(Cache[_KT, _VT]):
    def __init__(
        self,
        maxsize: float,
        choice: Callable[[Sequence[_KT]], _KT] = random.choice,
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...
    @property
    def choice(self) -> Callable[[Sequence[_KT]], _KT]: ...

class _TimedCache(Cache[_KT, _VT], Generic[_KT, _VT, _TT]):
    def __init__(
        self,
        maxsize: float,
        timer: Callable[[], _TT],
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...

    class _Timer(AbstractContextManager[_T]):
        def __init__(self, timer: Callable[[], _T]) -> None: ...
        def __call__(self) -> _T: ...
        def __enter__(self) -> _T: ...
        def __exit__(self, *exc: object) -> None: ...
        def __getattr__(self, name: str) -> Any: ...

    @property
    def timer(self) -> _Timer[_TT]: ...

class TTLCache(_TimedCache[_KT, _VT, _TT]):
    @overload
    def __init__(
        self: TTLCache[_KT2, _VT2, float],
        maxsize: float,
        ttl: float,
        *,
        getsizeof: Callable[[_VT2], float] | None = None,
        admission: _AdmissionPolicy[_KT2] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        maxsize: float,
        ttl: Any,  # FIXME: must be "addable" to _TT
        timer: Callable[[], _TT],
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...
    @property
    def ttl(self) -> Any: ...
    def expire(self, time: _TT | None = None) -> list[tuple[_KT, _VT]]: ...

class TLRUCache(_TimedCache[_KT, _VT, _TT]):
    @overload
    def __init__(
        self: TLRUCache[_KT2, _VT2, float],
        maxsize: float,
        ttu: Callable[[_KT2, _VT2, float], float],
        *,
        getsizeof: Callable[[_VT2], float] | None = None,
        admission: _AdmissionPolicy[_KT2] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        maxsize: float,
        ttu: Callable[[_KT, _VT, _TT], _TT],
        timer: Callable[[], _TT],
        getsizeof: Callable[[_VT], float] | None = None,
        *,
        admission: _AdmissionPolicy[_KT] | None = None,
    ) -> None: ...
    @property
    def ttu(self) -> Callable[[_KT, _VT, _TT], _TT]: ...
    def expire(self, time: _TT | None = None) -> list[tuple[_KT, _VT]]: ...

class _CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: float | None
    currsize: float

@type_check_only
class _AbstractCondition(AbstractContextManager[Any], Protocol):
    # implementation and unit tests do not use plain wait() and notify()
    def wait(self, timeout: float | None = None) -> bool: ...
    def wait_for(
        self, predicate: Callable[[], _T], timeout: float | None = None
    ) -> _T: ...
    def notify(self, n: int = 1) -> None: ...
    def notify_all(self) -> None: ...

@type_check_only
class _cached_wrapper(Generic[_P, _R]):
    __wrapped__: Callable[_P, _R]
    __name__: str
    __doc__: str | None
    cache: MutableMapping[Any, Any] | None
    cache_key: Callable[..., Any] = ...
    cache_lock: AbstractContextManager[Any] | None = None
    cache_condition: _AbstractCondition | None = None
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def cache_clear(self) -> None: ...

@type_check_only
class _cached_wrapper_info(_cached_wrapper[_P, _R]):
    def cache_info(self) -> _CacheInfo: ...

@overload
def cached(
    cache: MutableMapping[_KT, Any] | None,
    key: Callable[..., _KT] = ...,
    lock: AbstractContextManager[Any] | None = None,
    condition: _AbstractCondition | None = None,
    info: Literal[False] = ...,
) -> Callable[[Callable[_P, _R]], _cached_wrapper[_P, _R]]: ...
@overload
def cached(
    cache: MutableMapping[_KT, Any] | None,
    key: Callable[..., _KT] = ...,
    lock: AbstractContextManager[Any] | None = None,
    condition: _AbstractCondition | None = None,
    *,
    info: Literal[True],
) -> Callable[[Callable[_P, _R]], _cached_wrapper_info[_P, _R]]: ...
@overload
def cached(
    cache: MutableMapping[_KT, Any] | None,
    key: Callable[..., _KT],
    lock: AbstractContextManager[Any] | None,
    condition: _AbstractCondition | None,
    info: Literal[True],
) -> Callable[[Callable[_P, _R]], _cached_wrapper_info[_P, _R]]: ...

@type_check_only
class _cachedmethod_wrapper(Generic[_P, _R]):
    __wrapped__: Callable[Concatenate[Any, _P], _R]
    __name__: str
    __doc__: str | None
    cache: MutableMapping[Any, Any]
    cache_key: Callable[..., Any] = ...
    cache_lock: AbstractContextManager[Any] | None = None
    cache_condition: _AbstractCondition | None = None
    def __set_name__(self, owner: type, name: str) -> None: ...
    def __get__(
        self, obj: Any, objtype: type | None = None
    ) -> _cachedmethod_wrapper[_P, _R]: ...
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def cache_clear(self) -> None: ...

@type_check_only
class _cachedmethod_wrapper_info(_cachedmethod_wrapper[_P, _R]):
    def __get__(
        self, obj: Any, objtype: type | None = None
    ) -> _cachedmethod_wrapper_info[_P, _R]: ...
    def cache_info(self) -> _CacheInfo: ...

@overload
def cachedmethod(
    cache: Callable[[Any], MutableMapping[_KT, Any]],
    key: Callable[..., _KT] = ...,
    lock: Callable[[Any], AbstractContextManager[Any]] | None = None,
    condition: Callable[[Any], _AbstractCondition] | None = None,
    info: Literal[False] = ...,
) -> Callable[[Callable[Concatenate[Any, _P], _R]], _cachedmethod_wrapper[_P, _R]]: ...
@overload
def cachedmethod(
    cache: Callable[[Any], MutableMapping[_KT, Any]],
    key: Callable[..., _KT] = ...,
    lock: Callable[[Any], AbstractContextManager[Any]] | None = None,
    condition: Callable[[Any], _AbstractCondition] | None = None,
    *,
    info: Literal[True],
) -> Callable[
    [Callable[Concatenate[Any, _P], _R]], _cachedmethod_wrapper_info[_P, _R]
]: ...
@overload
def cachedmethod(
    cache: Callable[[Any], MutableMapping[_KT, Any]],
    key: Callable[..., _KT],
    lock: Callable[[Any], AbstractContextManager[Any]] | None,
    condition: Callable[[Any], _AbstractCondition] | None,
    info: Literal[True],
) -> Callable[
    [Callable[Concatenate[Any, _P], _R]], _cachedmethod_wrapper_info[_P, _R]
]: ...
