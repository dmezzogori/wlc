from .typings import (
    DueDateDistribution,
    Machine,
    PerFamily,
    Probability,
    RoutingProbabilities,
    TimeDistribution,
)

machines = ["1", "2", "3", "4", "5", "6"]
bottlenecks = (("3", 14), ("6", 35))

time_distributions = {
    "F1": {"func": "gammavariate", "kwargs": {"alpha": 2, "beta": 2}, "mean": 4},
    "F2": {"func": "gammavariate", "kwargs": {"alpha": 4, "beta": 0.5}, "mean": 2},
    "F3": {"func": "gammavariate", "kwargs": {"alpha": 6, "beta": 1 / 6}, "mean": 1},
}
due_dates = {
    "F1": {"func": "uniform", "args": [30, 50]},
    "F2": {"func": "uniform", "args": [30, 50]},
    "F3": {"func": "uniform", "args": [30, 50]},
}
routing = {
    "F1": {"1": 1, "2": 1, "3": 0, "4": 1, "5": 1, "6": 1},
    "F2": {"1": 0.8, "2": 0.8, "3": 1, "4": 0.8, "5": 0.8, "6": 0.75},
    "F3": {"1": 0, "2": 0, "3": 1, "4": 0, "5": 0, "6": 0.75},
}
weights = {"F1": 0.1, "F2": 0.52, "F3": 0.38}
timeout = {"func": "expovariate", "kwargs": {"lambd": 0.65}, "mean": 0.65}


def get_build_params(
    *,
    machines: tuple[Machine],
    bottlenecks: tuple[tuple[Machine, float], ...] = bottlenecks,
    effective=True,
    accept=True,
    warmup=700,
    sampling=5,
    time_distributions: PerFamily[TimeDistribution],
    due_dates: PerFamily[DueDateDistribution],
    routing: PerFamily[RoutingProbabilities],
    weights: PerFamily[Probability],
    timeout_params: TimeDistribution,
):
    return {
        "shopfloor_params": {
            "machines": machines,
            "bottlenecks": bottlenecks,
            "accept": accept,
            "effective": effective,
            "warmup": warmup,
            "sampling": sampling,
        },
        "router_params": {
            "time_distributions": time_distributions,
            "due_dates": due_dates,
            "routing": routing,
            "weights": weights,
            "timeout_params": timeout_params,
        },
    }
