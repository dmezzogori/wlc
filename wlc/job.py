from __future__ import annotations

import pprint
from typing import TYPE_CHECKING, Any, Self

import simpy

if TYPE_CHECKING:
    from .typings import Family, Machine, Routing, Timestamp


class Job:
    __slots__ = (
        "_entry_time",
        "_exit_time",
        "machines",
        "_processing_times",
        "_routing",
        "_visited",
        "active_machine",
        "due_date",
        "arrival_date",
        "effective",
        "corrected_load",
        "env",
        "family",
        "finished",
        "mean_processing_time",
        "routing",
        "total_processing_time",
        "total_wip",
        "machine_wip",
        "working_on",
        "entry_state",
        "push_it",
        "due_date_changed",
        "workload_contribution",
    )

    def __init__(
        self,
        env: simpy.Environment,
        family: Family,
        routing: Routing,
        due_date: Timestamp,
        mean_processing_time: float,
        effective: bool,
        corrected_load: bool,
    ) -> None:
        """
        Params:
            - env: reference to simpy environment
            - family: the family the job belongs to. str
            - routing: a tuple of (`machine`, `processing_time`) tuples
            - due_date: the timestamp at which the job is required to be finished
            - mean: the mean processing time of the corresponding family
            - effective: boolean, whether the processing_times are effective or not
        """
        self.env = env
        self.family = family
        self.routing = routing
        self.due_date = due_date
        self.due_date_changed = False
        self.arrival_date = self.env.now
        self.mean_processing_time = mean_processing_time
        self.effective = effective
        self.corrected_load = corrected_load

        self._processing_times = dict(self.routing)
        self.total_processing_time = sum(self._processing_times.values())

        self.machines = tuple(machine for machine, _ in self.routing)
        self._visited = {m: False for m in self.machines}

        # as soon as the job is created the total wip is set equal to the sum
        # of the effective/mean processing times.
        self.total_wip = (
            self.total_processing_time
            if self.effective
            else self.mean_processing_time * len(self.machines)
        )

        # as soon as the job is created the workload contribution
        # and the wip per machine are calculated,
        # accordingly to corrected_load and effective.
        self.workload_contribution: dict[str, float] = {}
        self.machine_wip: dict[str, float] = {}
        for i, (machine, processing_time) in enumerate(self.routing):
            n = processing_time if self.effective else self.mean_processing_time
            d = (i + 1) if self.corrected_load else 1
            self.workload_contribution[machine] = n / d
            self.machine_wip[machine] = n

        self._entry_time: dict[Machine, Timestamp] = {}
        self._exit_time: dict[Machine, Timestamp] = {}
        self.active_machine: Machine | None = None
        self.working_on = False
        self.finished = False
        self.entry_state = None
        self.push_it = False

    def __repr__(self) -> str:
        return (
            f"Job(env, family={self.family!r}, routing={self.routing!r}, "
            f"due_date={self.due_date}, mean_processing_time={self.mean_processing_time!r}, "
            f"effective={self.effective!r})"
        )

    def __str__(self) -> str:
        d = {
            machine: {
                "1.Queue in": self.queue_entry_time(machine),
                "2.Queue out / Machine In": self.machine_entry_time(machine),
                "3.Waiting Time": self.queue_time(machine),
                "4.Processing Time": self.processing_time(machine),
                "5.Machine Out": self.machine_exit_time(machine),
            }
            for machine in self.machines
        }

        return (
            f"{id(self)} - {self.family}\n"
            f"DD: {self.due_date} Start: {self.start} - Stop: {self.stop}\n"
            f"Tardiness: {self.tardiness} - "
            f"Queue Time: {self.total_queue_time}\n"
            f"{pprint.pformat(d, indent=2)}"
        )

    def __getstate__(self) -> dict[str, Any]:
        return dict([(k, getattr(self, k, None)) for k in self.__slots__ if k != "env"])

    def __setstate__(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, v)

    def __iter__(self) -> Self:
        self._routing = iter(self.routing)
        self.active_machine = None
        self.working_on = False
        return self

    def __next__(self):
        """Main iterative loop.
        Each time returns the next `current_machine`
        and the corresponding `processing_time`.

        It keeps track of:
            - the machine which the job is at (`self.active_machine`)
            - the remaining total processing time (`self.total_wip`),
            - if the job is being processed (`self.working_on`)
            - which machines have been visited (`self._visited`)
            - if the processing has been completed (`self.finished`)
        """
        self.total_wip -= self[self.active_machine]

        self.working_on = False
        if self.active_machine:
            self._visited[self.active_machine] = True

        try:
            curr_machine, processing_time = next(self._routing)
        except StopIteration as e:
            self.finished = True
            self.active_machine = None
            raise e

        self.active_machine = curr_machine
        return curr_machine, processing_time

    def __getitem__(self, machine):
        """job[machine] returns the processing time at `machine`,
        either the effective processing time or the mean."""
        return (
            self.processing_time(machine)
            if self.effective
            else self.mean_processing_time
        )

    def __enter__(self) -> Self:
        """Triggered as soon as the job enters the `self.active_machine` queue"""
        if self.active_machine is None:
            raise RuntimeError("Job has no active machine")

        self._entry_time[self.active_machine] = self.env.now
        return self

    def __exit__(self, typ, value, traceback) -> None:
        """
        Triggered as soon as the job exits the `self.active_machine` queue,
        and starts being processed.
        """

        self.working_on = True
        if self.active_machine is None:
            raise RuntimeError("Job has no active machine")

        self._exit_time[self.active_machine] = self.env.now

    def visits(self, machine: str) -> bool:
        """
        Does the job visits the `machine`?
        """

        return machine in self.machines

    def visited(self, machine: str) -> bool:
        """
        Has the job already visited the machine?
        """

        return self._visited.get(machine, False)

    def queue_entry_time(self, machine: str) -> float | None:
        """
        Returns the timestamp at which the job entered the `machine`'s queue.
        """

        return self._entry_time.get(machine, None)

    def machine_entry_time(self, machine: str) -> float | None:
        """
        Returns the timestamp at which the job exited the `machine`'s queue and began being processed
        """

        return self._exit_time.get(machine, None)

    queue_exit_time = machine_entry_time

    def machine_exit_time(self, machine: str) -> float | None:
        """
        Returns the exit timestamp of the job from the `machine`.

        If the job is in queue or being processed returns `None`
        """

        machine_entry_time = self.machine_entry_time(machine)

        if machine_entry_time is None:
            return None

        if self.processing_at(machine):
            return None

        return machine_entry_time + self.processing_time(machine)

    def in_queue_at(self, machine: str) -> bool:
        """
        Is the job in queue at `machine`?
        """

        return self.machine_entry_time(machine) is None

    def processing_at(self, machine: str) -> bool:
        """
        Is the job being processed by `machine`?
        """

        machine_entry_time = self.machine_entry_time(machine)

        if machine_entry_time is None:
            return False

        return machine_entry_time + self.processing_time(machine) > self.env.now

    def queue_time(self, machine: str) -> float | None:
        """
        Returns the amount of time spent in queue at `machine`
        """

        queue_exit_time = self.queue_exit_time(machine)
        queue_entry_time = self.queue_entry_time(machine)

        if queue_exit_time is not None and queue_entry_time is not None:
            return queue_exit_time - queue_entry_time
        return None

    @property
    def queue_times(self) -> dict[str, float | None]:
        """
        Returns a dictionary with machine as key and queue times as values.
        """

        return {machine: self.queue_time(machine) for machine in self.machines}

    @property
    def total_queue_time(self) -> float:
        """
        Returns the sum of queue times on each machine.
        """

        return sum(t for t in self.queue_times.values() if t is not None)

    def processing_time(self, machine: str) -> float:
        """
        Returns the processing time at `machine`.
        """

        return self._processing_times.get(machine, 0)

    @property
    def start(self) -> float | None:
        """
        Returns the timestamp of the job entering the shop-floor.
        """

        first_machine = self.machines[0]
        return self.queue_entry_time(first_machine)

    @property
    def stop(self) -> float | None:
        """
        Returns the timestamp of the job exiting the shop-floor.
        """

        last_machine = self.machines[-1]
        return self.machine_exit_time(last_machine)

    @property
    def total_throughput_time(self) -> float | None:
        """
        Returns the Total Throughput Time of the job
        """

        return (self.stop or self.env.now) - self.start if self.start else None

    @property
    def psp_waiting(self):
        """
        Returns the waiting time in the PSP
        """

        return self.start - self.arrival_date

    @property
    def _due_date_delta(self):
        """
        Returns the gap between the due date and the moment the job
        was finished being processed by the last machine in its routing.
        Returns 0 if the job is yet to be finished.
        """

        return self.stop - self.due_date if self.stop is not None else 0

    @property
    def tardiness(self) -> float:
        """
        Returns how much the job is tardy.
        """

        return max(self._due_date_delta, 0)

    @property
    def tardy(self) -> bool:
        """
        Returns whether the job is tardy or not.
        """

        return self.tardiness > 0

    @property
    def earliness(self) -> float:
        """
        Returns how much the job is early.
        """

        return -min(self._due_date_delta, 0)

    @property
    def early(self) -> bool:
        """
        Returns whether the job is early or not.
        """

        return self.earliness > 0
