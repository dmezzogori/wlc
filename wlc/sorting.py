from __future__ import annotations


def earliest_due_date(job):
    return job.due_date


class operation_due_date:
    def __init__(self, c):
        self.c = c

    def __call__(self, job):
        env = job.env
        due_date = job.due_date
        n = len(job.machines)
        i = sum(job._visited.values())
        ret = (due_date - env.now) - (n - i) * self.c
        return ret
