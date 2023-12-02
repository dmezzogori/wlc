from __future__ import annotations

import random
from collections.abc import Generator
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from simpy import Event

from .job import Job

if TYPE_CHECKING:
    from .shopfloor import ShopFloor
    from .typings import (
        DueDateDistribution,
        PerFamily,
        Probability,
        RoutingProbabilities,
        TimeDistribution,
    )


class Router:
    def __init__(
        self,
        shopfloor: ShopFloor,
        time_distributions: PerFamily[TimeDistribution],
        due_dates: PerFamily[DueDateDistribution],
        routing: PerFamily[RoutingProbabilities],
        weights: PerFamily[Probability],
        timeout_params: TimeDistribution,
    ):
        self.shopfloor = shopfloor
        self.time_distributions = time_distributions
        self.due_dates = due_dates
        self.routing = routing
        self.weights = weights
        self.timeout_params = timeout_params

        s1 = set(self.time_distributions.keys())
        s2 = set(self.routing.keys())
        s3 = set(self.weights.keys())
        if s1 != s2 or s2 != s3 or s1 != s3:
            raise ValueError("Controllare le famiglie")

        self.shopfloor.env.process(self())
        self.new_job_event = Event(env=self.shopfloor.env)

    def __repr__(self) -> str:
        return (
            f"Router({self.shopfloor!r}, {self.time_distributions!r}, "
            f"{self.time_distributions!r}, {self.due_dates!r}, "
            f"{self.routing!r}, {self.weights!r}, "
            f"{self.timeout_func.__name__!r})"
        )

    def __call__(self) -> Generator[Event, None, None]:
        while True:
            yield self.shopfloor.env.timeout(self.timeout)
            self.shopfloor.insert(self.new_job)
            self.new_job_event.succeed()
            self.new_job_event = Event(env=self.shopfloor.env)

    def __getstate__(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("time_funcs")
        d.pop("dd_funcs")
        d.pop("timeout_func")
        return d

    def __setstate__(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, v)

    @staticmethod
    def randomator(
        params: TimeDistribution | DueDateDistribution,
    ) -> Callable[[], float]:
        f = params["func"]
        func: Callable[..., float]
        if isinstance(f, str):
            func = getattr(random, f)
        else:
            func = f
        args = params.get("args", ())
        kwargs = params.get("kwargs", {})

        @wraps(func)
        def inner_func() -> float:
            return func(*args, **kwargs)

        return inner_func

    def build_funcs(
        self, params: PerFamily[TimeDistribution] | PerFamily[DueDateDistribution]
    ) -> PerFamily[Callable[[], float]]:
        return {family: self.randomator(p) for family, p in params.items()}

    @property
    def time_distributions(self) -> PerFamily[TimeDistribution]:
        return self._time_distributions

    @time_distributions.setter
    def time_distributions(self, value: PerFamily[TimeDistribution]) -> None:
        self._time_distributions = value
        self.time_funcs = self.build_funcs(value)

    @property
    def due_dates(self) -> PerFamily[DueDateDistribution]:
        return self._due_dates

    @due_dates.setter
    def due_dates(self, value: PerFamily[DueDateDistribution]) -> None:
        self._due_dates = value
        self.dd_funcs = self.build_funcs(value)

    @property
    def timeout_params(self) -> TimeDistribution:
        return self._timeout_params

    @timeout_params.setter
    def timeout_params(self, value: TimeDistribution):
        self._timeout_params = value
        self.timeout_func = self.randomator(value)

    @property
    def timeout(self) -> float:
        return self.timeout_func()

    @property
    def new_job(self) -> Job:
        env = self.shopfloor.env

        #  family
        family = random.choices(
            list(self.weights.keys()), weights=list(self.weights.values())
        )[0]

        #  routing
        time_func = self.time_funcs[family]
        mean = self.time_distributions[family]["mean"]
        routing = tuple(
            (machine, time_func())
            for machine, prob in self.routing[family].items()
            if random.random() < prob
        )

        #  due_dates
        due_date = self.dd_funcs[family]() + self.shopfloor.env.now

        #  effective
        effective = self.shopfloor.effective
        corrected_load = self.shopfloor.corrected_load

        return Job(env, family, routing, due_date, mean, effective, corrected_load)
