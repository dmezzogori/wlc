from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import simpy
from sortedcontainers import SortedList

from .job import Job
from .sorting import earliest_due_date
from .typings import JobSorter, Machine


class ShopFloor:
    __slots__ = (
        "env",
        "machines",
        "bottlenecks",
        "sorter_psp",
        "sorter_queues",
        "effective",
        "corrected_load",
        "warmup",
        "sampling",
        "sample_norms_and_queues",
        "_status",
        "check_timeout",
        "record_insert_status",
        "psp",
        "jobs",
        "jobs_doing",
        "jobs_done",
        "total_wip",
        "machine_wip",
        "machine_workload",
        "records",
    )

    due_date_setter: Callable[[Job], float] | None = None

    def __init__(
        self,
        env: simpy.Environment,
        machines: Sequence[Machine],
        bottlenecks: tuple[tuple[Machine, float], ...] | None,
        sorter_psp: JobSorter = earliest_due_date,
        sorter_queues: JobSorter = earliest_due_date,
        effective=True,
        corrected_load=True,
        warmup=700,
        sampling=5,
        sample_norms_and_queues=False,
        status: Callable[[Job], Any] | None = None,
        check_timeout=1.0,
        record_insert_status=False,
        due_date_setter: Callable[[Job], float] | None = None,
    ):
        self.env = env
        self.machines = {m: simpy.PriorityResource(env, capacity=1) for m in machines}
        self.bottlenecks = bottlenecks
        self.sorter_psp = sorter_psp
        self.sorter_queues = sorter_queues
        self.effective = effective
        self.corrected_load = corrected_load
        self.warmup = warmup
        self.sampling = sampling
        self.sample_norms_and_queues = sample_norms_and_queues
        self.check_timeout = check_timeout
        self.record_insert_status = record_insert_status
        ShopFloor.due_date_setter = due_date_setter
        self._status = status

        self.psp: SortedList[Job] = SortedList(key=self.sorter_psp)
        self.jobs: list[Job] = []
        self.jobs_doing: list[Job] = []
        self.jobs_done: list[Job] = []

        self.total_wip = 0
        self.machine_wip: dict[Machine, float] = {m: 0 for m in machines}
        self.machine_workload: dict[Machine, float] = {m: 0 for m in machines}

        self.records: dict = {
            "total_wip": [],
            "machine_wip": defaultdict(list),  # queue_sampling
            "machine_workload": defaultdict(list),  # norms_sampling
            "machine_utilization": defaultdict(int),
            "psp_queue": [],
        }

        self.env.process(self())

    def __repr__(self):
        return (
            f"ShopFloor(env, machines={list(self.machines.keys())!r}, "
            f"bottlenecks={self.bottlenecks!r}, "
            f"effective={self.effective!r}, warmup={self.warmup!r}, "
            f"corrected_load={self.corrected_load!r},"
            f"sampling={self.sampling!r})"
        )

    def __str__(self):
        return "\n".join(
            (
                f"{k.replace('_', ' ').title()}"
                f" [{round(v, 2) if isinstance(v, float) else v}]"
            )
            for k, v in self.stats.items()
        )

    def __getstate__(self):
        to_skip = (
            "env",
            "action_needed",
            "action_taken",
            "machines",
            "sorter_psp",
            "sorter_queues",
            "status",
            "due_date_setter",
        )
        d = dict(
            (attr, getattr(self, attr, None))
            for attr in self.__slots__
            if attr not in to_skip
        )
        d["psp"] = list(d["psp"])
        return d

    def __setstate__(self, data):
        for attr, value in data.items():
            setattr(self, attr, value)

    def __call__(self):
        while True:
            self.sample()
            yield self.env.timeout(self.check_timeout)

            N = len(self.psp)
            j = 0
            for _ in range(N):
                job = self.psp[j]

                if job.push_it:
                    accept = True
                else:
                    if self.bottlenecks:
                        accept = all(
                            self.machine_workload[machine] + job[machine] <= threshold
                            for machine, threshold in self.bottlenecks
                        )
                    else:
                        accept = True

                if accept:
                    self.psp.pop(j)

                    # registriamo la presenza del job nello shopfloor
                    self.jobs.append(job)
                    # registriamo la presenza del job tra quelli in lavorazione
                    self.jobs_doing.append(job)

                    # aggiornamento workload del job, per tutte le macchine da visitare
                    for (
                        machine,
                        workload_contribution,
                    ) in job.workload_contribution.items():
                        self.machine_workload[machine] += job.workload_contribution[
                            machine
                        ]

                    # aggiornamento del wip totale
                    self.total_wip += job.total_wip

                    self.env.process(self.work(job))

                else:
                    j += 1

    def render(self):
        print(str(self).replace("\n", " "), end="\r", flush=True)

    def insert(self, job):
        if self.record_insert_status and job.entry_state is None:
            job.entry_state = self.status(job)  # registriamo lo stato del sistema

        if self.env.now > self.warmup:  # warmup terminato
            # prima volta che il job entra nel psp
            if self.due_date_setter:  # occorre concordare la due date?
                new_due_date = self.due_date_setter(job)
                if new_due_date is not None:
                    job.due_date = new_due_date
        self.psp.add(job)

    def work(self, job):
        for machine, processing_time in job:
            res = self.machines[machine]
            with res.request(priority=self.sorter_queues(job)) as req:
                with job:
                    # aggiornamento del wip della macchina
                    self.machine_wip[machine] += (
                        processing_time if self.effective else job.mean_processing_time
                    )
                    yield req

                # il job è uscito dalla coda della macchina, aggiorniamo il wip totale e della macchina
                wip_contribution = (
                    processing_time if self.effective else job.mean_processing_time
                ) / 2
                self.total_wip -= wip_contribution
                self.machine_wip[machine] -= wip_contribution

                # inizio processamento job
                yield self.env.timeout(processing_time)
                # termine processamento job

                # aggiorniamo il tempo di utilizzo della macchina
                self.records["machine_utilization"][machine] += processing_time

                # aggiorniamo il workload sulla macchina,
                self.machine_workload[machine] -= job.workload_contribution[machine]

                # aggiorniamo il wip generale e della macchina
                self.total_wip -= wip_contribution
                self.machine_wip[machine] -= wip_contribution

        # rimuoviamo il job tra quelli in lavorazione
        self.jobs_doing.pop(self.jobs_doing.index(job))
        # registriamo il job tra quelli terminati
        self.jobs_done.append(job)

    def sample(self):
        now = self.env.now
        if now > self.warmup and round(now % self.sampling) == 0:
            self.records["total_wip"].append(self.total_wip)
            self.records["psp_queue"].append(len(self.psp))
            if self.sample_norms_and_queues:
                for machine in self.machines:
                    self.records["machine_wip"][machine].append(
                        self.machine_wip[machine]
                    )
                    self.records["machine_workload"][machine].append(
                        self.machine_workload[machine]
                    )

    @property
    def stats(self):
        total_wip = self.records["total_wip"]
        wip_mean = statistics.mean(total_wip) if total_wip else 0

        psp_queue = self.records["psp_queue"]
        psp_queue_mean = statistics.mean(psp_queue) if psp_queue else 0

        if self.jobs_done:
            throughput_time = []
            psp_waiting_time = []
            tardy = []
            tardiness = []
            lateness = []
            for job in self.jobs_done:
                throughput_time.append(job.total_throughput_time)
                psp_waiting_time.append(job.psp_waiting)
                tardy.append(1 if job.tardy else 0)
                tardiness.append(job.tardiness)
                if job.tardy:
                    lateness.append(job.tardiness)

            throughput_time = statistics.mean(throughput_time)
            psp_waiting_time = statistics.mean(psp_waiting_time)
            tardy = statistics.mean(tardy)
            tardiness = statistics.mean(tardiness)
            lateness = statistics.mean(lateness)

        clock = self.env.now or 1
        utilization_rate = {}
        for machine in self.machines:
            utilization_rate[machine] = (
                self.records["machine_utilization"][machine] / clock
            )

        return {
            "clock": clock,
            "psp_queue": len(self.psp),
            "psp_queue_mean": psp_queue_mean,
            "jobs_doing": len(self.jobs_doing),
            "jobs_done": len(self.jobs_done),
            "th_rate": len(self.jobs_done) / clock,
            "wip_mean": wip_mean,
            "throughput_time_mean": throughput_time,
            "psp_waiting_time_mean": psp_waiting_time,
            "utilization_rate": utilization_rate,
            "tardy": tardy,
            "tardiness_mean": tardiness,
            "lateness_mean": lateness,
        }

    def status(self, job_to_insert):
        if self._status == "base":
            psp = (
                sum(
                    job.processing_time(machine) if job.visits(machine) else 0
                    for job in self.psp
                )
                for machine in self.machines
            )

            doing = (
                sum(
                    job.processing_time(machine)
                    if (job.visits(machine) and not job.visited(machine))
                    else 0
                    for job in self.jobs_doing
                )
                for machine in self.machines
            )

            return tuple((*psp, *doing))

        elif self._status == "extra":
            jobs_machines = job_to_insert.machines

            psp = (
                tuple(
                    job.processing_time(m) if m in jobs_machines else 0
                    for m in self.machines
                )
                for job in reversed(self.psp)
            )

            separator = tuple(-1 for _ in range(len(self.machines)))
            doing = []
            last_machine = "0"
            for job in sorted(
                self.jobs_doing,
                key=lambda j: int(j.active_machine)
                if j.active_machine is not None
                else 0,
                reverse=False,
            ):
                if job.active_machine != last_machine:
                    doing.append(separator)
                    last_machine = job.active_machine

                i = tuple(
                    job.processing_time(m)
                    if ((not job.visited(m)) and m in jobs_machines)
                    else 0
                    for m in self.machines
                )

                if sum(i) != 0:
                    doing.append(i)

            return (*psp, *doing)

        return None

    def plot(self, bottlenecks=False, machines=None):
        machines = (
            [i[0] for i in self.bottlenecks]
            if bottlenecks
            else (machines or self.machines)
        )
        n = len(machines)
        with plt.style.context("ggplot"):
            fig, axes = plt.subplots(n, 2, sharex=True, sharey=True, figsize=(8, 6))

            axes[0][0].set_title(f"Norms [t{'' if self.effective else '-mean'}]")
            axes[0][1].set_title(f"Queues [t{'' if self.effective else '-mean'}]")

            for machine, ax, color in zip(
                machines, axes, plt.rcParams["axes.prop_cycle"]
            ):
                y = self.norms_sampling[machine]
                ax[0].plot(y, linestyle="-", linewidth=0.8, color=color["color"])
                ax[0].set_ylabel(f"WC {machine}")

                y = self.queues_sampling[machine]
                ax[1].plot(y, linestyle="-", linewidth=0.8, color=color["color"])

            fig.tight_layout()

        with plt.style.context("ggplot"):
            fig, axes = plt.subplots(
                1, 2, figsize=(8, (6 / n) + 0.3), sharey=True, sharex=True
            )

            axes[0].plot(
                self.total_wips,
                color=list(plt.rcParams["axes.prop_cycle"])[-1]["color"],
            )
            axes[0].set_ylabel("WIP")
            axes[0].set_title("System")

            fig.tight_layout()

        print(self)


class UniformRouting:
    def __init__(self, machines, flow=False):
        self.machines = machines
        self.flow = flow

    def __repr__(self):
        return f"UniformRouting(machines={self.machines!r}, flow={self.flow!r})"

    def __iter__(self):
        n_machines = random.randint(1, len(self.machines))
        _routing = []
        while len(_routing) != n_machines:
            machine = random.choice(self.machines)
            if machine not in _routing:
                _routing.append(machine)  # machine, probability

        if self.flow:
            _routing = sorted(_routing, key=lambda x: x[0])

        _routing = tuple((machine, 1) for machine in _routing)

        self._routing = _routing
        self.routing = iter(self._routing)
        return self

    def __next__(self):
        return next(self.routing)

    def items(self):
        return self
