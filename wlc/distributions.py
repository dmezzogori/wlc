import random


def truncated_2_erlang(mean: float, max_value=4.0) -> float:
    lambd = 1 / mean

    out = float("inf")
    while out > max_value:
        out = random.expovariate(lambd) + random.expovariate(lambd)
    return out
