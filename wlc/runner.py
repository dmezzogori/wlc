import random
from collections.abc import Iterable, Mapping
from multiprocessing import Pool
from typing import Generic, TypeVar

import numpy as np
import progressbar
import simpy
from simpy import Environment
from tqdm import tqdm

from .router import Router
from .shopfloor import ShopFloor
from .typings import BuildParams, RouterParams, RunParams, ShopFloorParams

R = TypeVar("R", bound=Mapping)


def build(
    shopfloor_params: ShopFloorParams, router_params: RouterParams
) -> tuple[Environment, Router, ShopFloor]:
    env = simpy.Environment()
    shopfloor = ShopFloor(env, **shopfloor_params)
    router = Router(shopfloor, **router_params)
    return env, router, shopfloor


def run(env: Environment, until=3650, seed: int | None = None) -> None:
    if seed:
        random.seed(seed)
    env.run(until=until)


def worker(args: tuple[BuildParams, RunParams, bool]):
    build_params, run_params, return_stats = args
    env, router, shopfloor = build(**build_params)
    run(env, **run_params)
    return shopfloor.stats if return_stats else shopfloor


class Runner(Generic[R]):
    def __init__(
        self,
        build_params: BuildParams,
        run_params: RunParams | None = None,
        return_stats: bool = False,
        seed: int | None = None,
        n: int = 4,
    ) -> None:
        self.build_params = build_params
        self.run_params: RunParams = run_params or {"seed": seed}
        self.return_stats = return_stats
        self.seed = seed
        self.n = n
        self.bar = progressbar.ProgressBar(maxval=self.n)
        self.results: list[R] = []

    def __call__(self, *, parallel=False, processes: int | None) -> list[R]:
        self.results = (
            self._parallel(processes=processes) if parallel else self._serial()
        )
        return self.results

    def __repr__(self) -> str:
        return (
            f"Test(build_params={self.build_params!r}, "
            f"run_params={self.run_params!r}, "
            f"seed={self.seed}, n={self.n})"
        )

    def __str__(self) -> str:
        res = self.results
        stats = list(res[0].keys())
        return "\n".join(
            f"{stat.replace('_', ' ').title()}: {np.mean([i[stat] for i in res])}"
            for stat in stats
        )

    def __getitem__(self, idx: int) -> R:
        return self.results[idx]

    def _parallel(self, processes=4) -> list[R]:
        pool = Pool(processes=processes)

        results: list[R] = []
        for result in tqdm(
            pool.imap_unordered(func=worker, iterable=self.todo), total=self.n
        ):
            results.append(result)

        return results

    def _serial(self) -> list[R]:
        return [worker(args) for args in tqdm(self.todo, total=self.n)]

    @property
    def todo(self) -> Iterable[tuple[BuildParams, RunParams, bool]]:
        for _ in range(self.n):
            yield self.build_params, self.run_params, self.return_stats
