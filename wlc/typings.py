from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NotRequired, Protocol, TypeAlias, TypedDict, TypeVar

from .job import Job

D = TypeVar("D")
Family: TypeAlias = str
Machine: TypeAlias = str
Probability: TypeAlias = float
Timestamp: TypeAlias = float
PerFamily: TypeAlias = dict[Family, D]
RoutingProbabilities: TypeAlias = dict[Machine, Probability]
Routing: TypeAlias = tuple[tuple[Machine, float], ...]


class TimeDistribution(TypedDict):
    func: str | Callable
    args: NotRequired[tuple[Any, ...]]
    kwargs: NotRequired[dict[str, Any]]
    mean: float


class DueDateDistribution(TypedDict):
    func: str | Callable
    args: NotRequired[Sequence[Any]]
    kwargs: NotRequired[dict[str, Any]]


class JobSorter(Protocol):
    def __call__(self, job: Job) -> float:
        ...


class ShopFloorParams(TypedDict):
    machines: Sequence[Machine]
    bottlenecks: tuple[tuple[Machine, float], ...] | None
    sorter_psp: JobSorter
    sorter_queues: JobSorter
    effective: bool
    corrected_load: bool
    warmup: float
    sampling: int
    sample_norms_and_queues: bool
    status: Callable[[Job], Any] | None
    check_timeout: float
    record_insert_status: bool
    due_date_setter: Callable[[Job], float] | None


class RouterParams(TypedDict):
    time_distributions: PerFamily[TimeDistribution]
    due_dates: PerFamily[DueDateDistribution]
    routing: PerFamily[RoutingProbabilities]
    weights: PerFamily[Probability]
    timeout_params: TimeDistribution


class BuildParams(TypedDict):
    shopfloor_params: ShopFloorParams
    router_params: RouterParams


class RunParams(TypedDict):
    until: NotRequired[float]
    seed: int | None
