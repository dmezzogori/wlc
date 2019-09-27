import numpy as np
import matplotlib.pyplot as plt
import progressbar
import scipy.stats as st

from multiprocessing import Pool
from sklearn.neighbors import KernelDensity

from .factory import simpy, random, ShopFloor, Router


def set_seed(seed):
    if seed:
        random.seed(seed)

def confint(arr):
    return st.norm.interval(0.95, loc=np.mean(arr), scale=st.sem(arr))

# def density(arr):
#     X = np.array(arr).reshape(-1, 1)

#     start = min(arr)
#     stop = max(arr)
#     rng = stop - start

#     X_plot = np.linspace(
#         start=start - 0.1 * rng,
#         stop=stop + 0.1 * rng,
#         num=2000)[:, np.newaxis]
#     kde = KernelDensity(kernel="gaussian", bandwidth=10).fit(X)
#     log_dens = kde.score_samples(X_plot)
#     with plt.style.context("ggplot"):
#         plt.plot(X_plot, np.exp(log_dens))

def density(**kwargs):
    for name, arr in kwargs.items():
        X = np.array(arr).reshape(-1, 1)

        start = min(arr)
        stop = max(arr)
        rng = stop - start

        X_plot = np.linspace(
            start=start - 0.1 * rng,
            stop=stop + 0.1 * rng,
            num=2000)[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=10).fit(X)
        log_dens = kde.score_samples(X_plot)
        with plt.style.context("ggplot"):
            plt.plot(X_plot, np.exp(log_dens), label=name)
        plt.legend()

def build(shopfloor_params, router_params):
    env = simpy.Environment()
    shopfloor = ShopFloor(env, **shopfloor_params)
    router = Router(shopfloor, **router_params)
    return env, router, shopfloor

def run(env, until=3650, seed=None):
    set_seed(seed)
    env.run(until=until)

def worker(args):
    build_params, run_params, return_stats = args
    env, router, shopfloor = build(**build_params)
    run(env, **run_params)
    return shopfloor.stats if return_stats else shopfloor


class Test:
    def __init__(self, build_params, run_params=None, return_stats=False, seed=None, n=4):
        self.build_params = build_params
        self.run_params = run_params or {"seed": seed}
        self.return_stats = return_stats
        self.seed = seed
        self.n = n
        self.bar = progressbar.ProgressBar(max_value=self.n)
        self.results = []

    def __call__(self, parallel=False):
        self.results = self._parallel() if parallel else self._serial()

    def __repr__(self):
        return (
            f"Test(build_params={self.build_params!r}, "
            f"run_params={self.run_params!r}, "
            f"seed={self.seed}, n={self.n})"
        )

    def __str__(self):
        res = self.results
        stats = list(res[0].keys())
        return "\n".join(
            f"{stat.replace('_', ' ').title()}: {np.mean(i[stat] for i in res)}"
            for stat in stats
        )

    def __getitem__(self, idx):
        return self.results[idx]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['bar']
        return state

    def _parallel(self):
        results = []

        with Pool(4) as pool:
            multiple_results = pool.map_async(
                worker, self.todo, callback=results.extend
            )
            self.track_job(multiple_results)
            multiple_results.wait()

        return results

    def _serial(self):
        return [worker(args) for args in self.bar(self.todo)]

    @property
    def todo(self):
        for _ in range(self.n):
            yield (self.build_params, self.run_params, self.return_stats)

    def track_job(self, job):
        while job._number_left > 0:
            self.bar.update(self.n - job._number_left)
        self.bar.finish()

class Lambda:
    def __init__(self, env, lambd, noise=None):
        self.env = env
        self._lambd = lambd
        self._noise = noise

    @property
    def static(self):
        return self.evolving_lambd if self._noise else self._lambd

    @property
    def evolving(self):
        return abs(np.cos(self.env.now) + random.gauss(self._lambd, self._noise))

def truncated_2_erlang(mean):

    lambd = 1 / mean

    out = float('inf')
    while out > 4.0:
        out = random.expovariate(lambd) + random.expovariate(lambd)
    return out
