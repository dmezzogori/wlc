import matplotlib.pyplot as plt
import pprint
import random
import simpy
import statistics

from collections import defaultdict
from functools import wraps
from sortedcontainers import SortedList


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
        ret = ((due_date - env.now) - (n - i) * self.c)
        return ret


class Job:

    __slots__ = (
        "_entry_time",
        "_exit_time",
        "_machines",
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
        "workload_contribution"
    )

    def __init__(self, env, family, routing, due_date, mean_processing_time, effective, corrected_load):
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

        self._machines = tuple(machine for machine, _  in self.routing)
        self._visited = {m: False for m in self._machines}
        
        # as soon as the job is created the total wip is set equal to the sum
        # of the effective/mean processing times.
        self.total_wip = (
            self.total_processing_time
            if self.effective
            else self.mean_processing_time * len(self._machines)
        )

        # as soon as the job is created the workload contribution
        # and the wip per machine are calculated,
        # accordingly to corrected_load and effective.
        self.workload_contribution = {}
        self.machine_wip = {}
        for i, (machine, processing_time) in enumerate(self.routing):
            n = processing_time if self.effective else self.mean_processing_time
            d = (i + 1) if self.corrected_load else 1
            self.workload_contribution[machine] = n / d
            self.machine_wip[machine] = n

        self._entry_time = {}
        self._exit_time = {}
        self.active_machine = None
        self.working_on = False
        self.finished = False
        self.entry_state = None
        self.push_it = False

    def __repr__(self):
        return (
            f"Job(env, family={self.family!r}, routing={self.routing!r}, "
            f"due_date={self.due_date}, mean_processing_time={self.mean_processing_time!r}, "
            f"effective={self.effective!r})")

    def __str__(self):
        d = {
            machine: {
                "1.Queue in": self.queue_entry_time(machine),
                "2.Queue out / Machine In": self.machine_entry_time(machine),
                "3.Waiting Time": self.queue_time(machine),
                "4.Processing Time": self.processing_time(machine),
                "5.Machine Out": self.machine_exit_time(machine),
            }
            for machine in self._machines
        }

        return (
            f"{id(self)} - {self.family}\n"
            f"DD: {self.due_date} Start: {self.start} - Stop: {self.stop}\n"
            f"Tardiness: {self.tardiness} - "
            f"Queue Time: {self.total_queue_time}\n"
            f"{pprint.pformat(d, indent=2)}"
        )

    def __getstate__(self):
        return dict([(k, getattr(self,k,None)) for k in self.__slots__ if k!='env'])

    def __setstate__(self, data):
        for k,v in data.items():
            setattr(self,k,v)

    def __iter__(self):
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
        return self.processing_time(machine) if self.effective else self.mean_processing_time

    def __enter__(self):
        """Triggered as soon as the job enters the `self.active_machine` queue"""
        self._entry_time[self.active_machine] = self.env.now
        return self

    def __exit__(self, typ, value, traceback):
        """Triggered as soon as the job exits the `self.active_machine` queue,
        and starts being processed."""
        self.working_on = True
        self._exit_time[self.active_machine] = self.env.now

    @property
    def machines(self):
        return self._machines

    def visits(self, machine):
        """Does the job visits the `machine`?"""
        return machine in self._machines

    def visited(self, machine):
        """Has the job already visited the machine?"""
        return self._visited.get(machine, False)

    def queue_entry_time(self, machine):
        """Returns the timestamp at which the job entered the `machine`'s queue."""
        return self._entry_time.get(machine, None)

    def machine_entry_time(self, machine):
        """Returns the timestamp at which the job exited the `machine`'s queue
        and began being processed"""
        return self._exit_time.get(machine, None)

    queue_exit_time = machine_entry_time

    def machine_exit_time(self, machine):
        """Returns the exit timestamp of the job from the `machine`.
        If the job is in queue or being processed returns `None`"""
        if self.in_queue_at(machine) or self.processing_at(machine):
            return None
        return self.machine_entry_time(machine) + self.processing_time(machine)

    def in_queue_at(self, machine):
        """Is the job in queue at `machine`?"""
        return self.machine_entry_time(machine) is None

    def processing_at(self, machine):
        """Is the job being processed by `machine`?"""
        return self.machine_entry_time(machine) + \
               self.processing_time(machine) > self.env.now

    def queue_time(self, machine):
        """Returns the amount of time spent in queue at `machine`"""
        queue_exit_time = self.queue_exit_time(machine)
        queue_entry_time = self.queue_entry_time(machine)

        if queue_exit_time is not None and queue_entry_time is not None:
            return queue_exit_time - queue_entry_time
        return None

    @property
    def queue_times(self):
        """Returns a dictionary with machine as key and queue times as values."""
        return {machine: self.queue_time(machine) for machine in self._machines}

    @property
    def total_queue_time(self):
        """Returns the sum of queue times on each machine."""
        return sum(t for t in self.queue_times.values() if t is not None)

    def processing_time(self, machine):
        """Returns the processing time at `machine`."""
        return self._processing_times.get(machine, 0)

    @property
    def start(self):
        """Returns the timestamp of the job entering the shop-floor."""
        first_machine = self._machines[0]
        return self.queue_entry_time(first_machine)

    @property
    def stop(self):
        """Returns the timestamp of the job exiting the shop-floor."""
        last_machine = self._machines[-1]
        return self.machine_exit_time(last_machine)

    @property
    def total_throughput_time(self):
        """Returns the Total Throughput Time of the job"""
        return (self.stop or self.env.now) - self.start if self.start else None

    @property
    def psp_waiting(self):
        """Returns the waiting time in the PSP"""
        return self.start - self.arrival_date

    @property
    def _due_date_delta(self):
        """Returns the gap between the due date and the moment the job
        was finished being processed by the last machine in its routing.
        Returns 0 if the job is yet to be finished."""
        return self.stop - self.due_date if self.stop else 0

    @property
    def tardiness(self):
        """Returns how much the job is tardy."""
        return max(self._due_date_delta, 0)

    @property
    def tardy(self):
        """Returns whether the job is tardy or not."""
        return self.tardiness > 0

    @property
    def earliness(self):
        """Returns how much the job is early."""
        return -min(self._due_date_delta, 0)

    @property
    def early(self):
        """Returns whether the job is early or not."""
        return self.earliness > 0


class Router:
    def __init__(
        self, shopfloor, time_distributions, due_dates, routing, weights, timeout_params):
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

    def __repr__(self):
        return (
            f"Router({self.shopfloor!r}, {self.time_distributions!r}, "
            f"{self.time_distributions!r}, {self.due_dates!r}, "
            f"{self.routing!r}, {self.weights!r}, "
            f"{self.timeout_func.__name__!r})"
        )

    def __call__(self):
        while True:
            yield self.shopfloor.env.timeout(self.timeout)
            self.shopfloor.insert(self.new_job)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('time_funcs')
        d.pop('dd_funcs')
        d.pop('timeout_func')
        return d

    def __setstate__(self, data):
        for k,v in data.items():
            setattr(self,k,v)

    @staticmethod
    def randomator(**params):
        f = params["func"]
        func = f if callable(f) else getattr(random, params["func"])
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})

        @wraps(func)
        def inner_func():
            return func(*args, **kwargs)

        return inner_func

    def build_funcs(self, params):
        return {family: self.randomator(**p) for family, p in params.items()}

    @property
    def time_distributions(self):
        return self._time_distributions

    @time_distributions.setter
    def time_distributions(self, value):
        self._time_distributions = value
        self.time_funcs = self.build_funcs(value)

    @property
    def due_dates(self):
        return self._due_dates

    @due_dates.setter
    def due_dates(self, value):
        self._due_dates = value
        self.dd_funcs = self.build_funcs(value)

    @property
    def timeout_params(self):
        return self._timeout_params

    @timeout_params.setter
    def timeout_params(self, value):
        self._timeout_params = value
        self.timeout_func = self.randomator(**value)

    @property
    def timeout(self):
        return self.timeout_func()

    @property
    def new_job(self):
        env = self.shopfloor.env

        #  family
        family = random.choices(
            list(self.weights.keys()),
            weights=list(self.weights.values())
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
        "due_date_setter",

        "psp",
        "jobs",
        "jobs_doing",
        "jobs_done",

        "total_wip",
        "machine_wip",
        "machine_workload",

        "records"
    )

    def __init__(
        self,
        env,
        machines,
        bottlenecks,
        sorter_psp=earliest_due_date,
        sorter_queues=earliest_due_date,
        effective=True,
        corrected_load=True,
        warmup=700,
        sampling=5,
        sample_norms_and_queues=False,
        status=None,
        check_timeout=1.,
        record_insert_status=False,
        due_date_setter=None):

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

        self.psp = SortedList(key=self.sorter_psp)
        self.jobs = []
        self.jobs_doing = []
        self.jobs_done = []

        self.total_wip = 0
        self.machine_wip = {m: 0 for m in machines}
        self.machine_workload = {m: 0 for m in machines}

        self.records = {
            'total_wip': [],
            'machine_wip': defaultdict(list), # queue_sampling
            'machine_workload': defaultdict(list), # norms_sampling
            'machine_utilization': defaultdict(int),
            'psp_queue': []
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
                (f"{k.replace('_', ' ').title()}"
                 f" [{round(v, 2) if isinstance(v, float) else v}]")
                for k, v in self.stats.items())

    def __getstate__(self):
        to_skip = ('env', 'action_needed', 'action_taken', 'machines', 'sorter_psp', 'sorter_queues', 'status', 'due_date_setter')
        d =  dict((attr, getattr(self, attr, None)) for attr in self.__slots__ if attr not in to_skip)
        d['psp'] = list(d['psp'])
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
                            for machine, threshold in self.bottlenecks)
                    else:
                        accept = True

                if accept:
                    self.psp.pop(j) 

                    # registriamo la presenza del job nello shopfloor
                    self.jobs.append(job)
                    # registriamo la presenza del job tra quelli in lavorazione
                    self.jobs_doing.append(job)

                    # aggiornamento workload del job, per tutte le macchine da visitare
                    for machine, workload_contribution in job.workload_contribution.items():
                        self.machine_workload[machine] += job.workload_contribution[machine]

                    # aggiornamento del wip totale
                    self.total_wip += job.total_wip

                    self.env.process(self.work(job))

                else:
                    j += 1

    def render(self):
        print(str(self).replace('\n', ' '), end="\r", flush=True)

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
                    self.machine_wip[machine] += processing_time if self.effective else job.mean_processing_time
                    yield req

                # il job è uscito dalla coda della macchina, aggiorniamo il wip totale e della macchina
                wip_contribution = (processing_time if self.effective else job.mean_processing_time) / 2
                self.total_wip -= wip_contribution
                self.machine_wip[machine] -= wip_contribution

                # inizio processamento job
                yield self.env.timeout(processing_time)
                # termine processamento job

                # aggiorniamo il tempo di utilizzo della macchina
                self.records['machine_utilization'][machine] += processing_time

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
            self.records['total_wip'].append(self.total_wip)
            self.records['psp_queue'].append(len(self.psp))
            if self.sample_norms_and_queues:
                for machine in self.machines:
                    self.records['machine_wip'][machine].append(self.machine_wip[machine])
                    self.records['machine_workload'][machine].append(self.machine_workload[machine])

    @property
    def stats(self):
        total_wip = self.records['total_wip']
        wip_mean = statistics.mean(total_wip) if total_wip else 0

        psp_queue = self.records['psp_queue']
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
            utilization_rate[machine] = self.records['machine_utilization'][machine] / clock

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
            "lateness_mean": lateness
        }

    def status(self, job_to_insert):

        if self._status == 'base':
            psp = (
                sum(job.processing_time(machine) if job.visits(machine) else 0
                    for job in self.psp)
                for machine in self.machines)

            doing = (
                sum(job.processing_time(machine)
                    if (job.visits(machine) and not job.visited(machine)) else 0
                    for job in self.jobs_doing)
                for machine in self.machines)

            return tuple((*psp, *doing))

        elif self._status == 'extra':

            jobs_machines = job_to_insert.machines

            psp = (
                tuple(
                    job.processing_time(m)
                    if m in jobs_machines else 0 
                    for m in self.machines)
                for job in reversed(self.psp)
            )

            separator = tuple(-1 for _ in range(len(self.machines)))
            doing = []
            last_machine = '0'
            for job in sorted(
                self.jobs_doing,
                key=lambda j: int(j.active_machine) if j.active_machine is not None else 0,
                reverse=False):

                if job.active_machine != last_machine:
                    doing.append(separator)
                    last_machine = job.active_machine

                i = tuple(
                    job.processing_time(m)
                    if ((not job.visited(m)) and m in jobs_machines) else 0
                    for m in self.machines)
                
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
                ax[0].plot(y, linestyle="-", linewidth=.8, color=color["color"])
                ax[0].set_ylabel(f"WC {machine}")

                y = self.queues_sampling[machine]
                ax[1].plot(y, linestyle="-", linewidth=.8, color=color["color"])

            fig.tight_layout()

        with plt.style.context("ggplot"):

            fig, axes = plt.subplots(
                1, 2, figsize=(8, (6 / n) + .3), sharey=True, sharex=True)

            axes[0].plot(
                self.total_wips, color=list(plt.rcParams["axes.prop_cycle"])[-1]["color"]
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
