import matplotlib.pyplot as plt
import pprint
import random
import simpy
import statistics

from collections import defaultdict
from functools import wraps
from sortedcontainers import SortedList


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
        "env",
        "family",
        "finished",
        "mean",
        "routing",
        "total_processing_time",
        "wip",
        "working_on",
        "entry_state",
    )

    def __init__(self, env, family, routing, due_date, mean, effective):
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
        self.arrival_date = self.env.now
        self.mean = mean
        self.effective = effective

        self._processing_times = dict(self.routing)
        self.total_processing_time = sum(self._processing_times.values())
        
        # as soon as the job is created the wip is set equal to the sum
        # of the effective/mean processing times.
        self.wip = (
            self.total_processing_time
            if self.effective
            else self.mean * len(self._machines)
        )

        self._machines = tuple(i[0] for i in self.routing)
        self._visited = {m: False for m in self._machines}

        self._entry_time = {}
        self._exit_time = {}
        self.active_machine = None
        self.working_on = False
        self.finished = False

        self.entry_state = None

    def __repr__(self):
        return (
            f"Job(env, family={self.family!r}, routing={self.routing!r}, "
            f"due_date={self.due_date}, mean={self.mean!r}, "
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
            - the remaining total processing time (`self.wip`), 
            - if the job is being processed (`self.working_on`)
            - which machines have been visited (`self._visited`)
            - if the processing has been completed (`self.finished`)
        """
        self.wip -= self[self.active_machine]

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
        return self.processing_time(machine) if self.effective else self.mean

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
        return sum(t for t in self.queue_times.values() if t)

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
    def lead_time(self):
        """Returns the lead time of the job in the shop-floor"""
        return (self.stop or self.env.now) - self.start if self.start else None

    def workload(self, machine):
        """Returns the job's workload at `machine`, if the job is currently 
        being processed at `machine`, otherwise returns 0."""
        if machine == self.active_machine:
            return self[machine] / (1 + self.working_on)
        return 0

    def norm(self, machine):
        """Returns the job's contribution to the norm at `machine`, if the job has yet
        to visit the `machine`, otherwise returns 0."""
        if self.visits(machine) and not self.visited(machine):
            return self[machine]
        return 0

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

    @property
    def psp_waiting(self):
        """Returns the waiting time in the PSP"""
        return self.arrival_date - self.start


class Router:
    def __init__(
        self, shopfloor, time_distributions, due_dates, routing, weights, timeout_params
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
        due_date = int(self.dd_funcs[family]() + self.shopfloor.env.now)

        #  effective
        effective = self.shopfloor.effective

        return Job(env, family, routing, due_date, mean, effective)


class ShopFloor:
    __slots__ = (
        "env",
        "machines",
        "bottlenecks",
        "sorter",
        "norms",
        "queues",
        "psp",
        "jobs",
        "action_needed",
        "action_taken",
        "wips",
        "accept",
        "effective",
        "warmup",
        "sampling",
        "check_timeout",
        "due_date_setter",

        "_status",
        "rejected"
    )

    def __init__(
        self,
        env,
        machines,
        bottlenecks,
        sorter=lambda job: job.due_date,
        accept=True,
        effective=True,
        warmup=700,
        sampling=5,
        status='null',
        check_timeout=1,
        due_date_setter=None):

        self.env = env
        self.machines = {m: simpy.PriorityResource(env, capacity=1) for m in machines}
        self.bottlenecks = bottlenecks
        self.sorter = sorter
        self.accept = accept
        self.effective = effective
        self.warmup = warmup
        self.sampling = sampling
        self.check_timeout = check_timeout
        ShopFloor.due_date_setter = due_date_setter

        self._status = status

        self.psp = SortedList(key=self.sorter)
        self.rejected = 0
        self.jobs = []

        self.action_needed = simpy.Event(self.env)
        self.action_taken = simpy.Event(self.env)

        self.wips = []
        self.norms = defaultdict(list)
        self.queues = defaultdict(list)

        if self.accept:
            env.process(self.workload_control())

        self.env.process(self())

    def __repr__(self):
        return (
            f"ShopFloor(env, machines={list(self.machines.keys())!r}, "
            f"bottlenecks={self.bottlenecks!r}, "
            f"accept={self.accept!r}, "
            f"effective={self.effective!r}, warmup={self.warmup!r}, "
            f"sampling={self.sampling!r})"
        )

    def __str__(self):
        return "\n".join(
                (f"{k.replace('_', ' ').title()}"
                 f" [{round(v, 2) if isinstance(v, float) else v}]")
                for k, v in self.stats.items())

    def __getstate__(self):
        to_skip = ('env', 'action_needed', 'action_taken', 'machines', 'sorter', 'status', 'due_date_setter')
        d =  dict((attr, getattr(self, attr, None)) for attr in self.__slots__ if attr not in to_skip)
        d['psp'] = list(d['psp'])
        return d

    def __setstate__(self, data):
        for attr, value in data.items():
            setattr(self, attr, value)

    def __iter__(self):
        return self

    # def __next__(self):
    #     job = None
    #     if self.psp:
    #         job = self.psp.pop(0)

    #         if self.accept(job):
    #             p = self.env.process(self.work(job))
    #             #p = Work(self, job)
    #             self.jobs.append(job)
    #             self.processes.append(p)
    #         else:
    #             self.psp.add(job)
    #             job = None
    #     else:
    #         if not self.psp_empty.triggered:
    #             self.psp_empty.succeed()

    #     return job

    # def __call__(self):
    #     yield self.env.timeout(0.0101)
    #     for _ in self:
    #         self.sample()
    #         yield self.env.timeout(0.01)
    #         # yield self.env.timeout(1)

    #         # if all(p.triggered for p in self.processes) and not self.all_done.triggered:
    #         #    self.all_done.succeed()

    def __next__(self):
        job = None
        if self.psp:
            job = self.psp.pop(0)
        return job

    def __call__(self):
        for job in self:

            if job:
                self.action_needed.succeed(value=job)
                accept = yield self.action_taken
                self.reset_event("action_taken")
                if accept:
                    self.env.process(self.work(job))
                    self.jobs.append(job)
                else:
                    self.rejected += 1
                    self.insert(job)

            self.sample()
            yield self.env.timeout(self.check_timeout)

    def render(self):
        print(str(self).replace('\n', ' '), end="\r", flush=True)

    def reset_event(self, event):
        setattr(self, event, simpy.Event(self.env))

    def insert(self, job):
        self.psp.add(job)
        now = self.env.now
        if now > self.warmup:
            job.entry_state = self.status(job)

            if self.due_date_setter:
                due_date = self.due_date_setter(job)
                if due_date:
                    index = self.psp.index(job)
                    job = self.psp.pop(index)
                    job.due_date = due_date
                    self.psp.add(job)

    def workload(self, machine):
        return sum(job.workload(machine) for job in self.doing)

    def norm(self, machine):
        return sum(job.norm(machine) for job in self.doing)

    def workload_control(self):
        while True:
            job = yield self.action_needed
            self.reset_event("action_needed")
            
            l = [self.norm(machine) + job[machine] < threshold
                for machine, threshold in self.bottlenecks]
            
            r = all(l)
            self.action_taken.succeed(value=r)
    
    def load(self, machine, relative=True):
        n_jobs = sum(job.visits(machine) for job in self.jobs)
        proc_time = sum(job.processing_time(machine) for job in self.jobs)
        if relative:
            proc_time /= self.env.now
        return n_jobs, proc_time

    def work(self, job):
        for idm, proc_time in job:
            machine = self.machines[idm]
            with machine.request(priority=job.due_date) as req:
                with job:
                    yield req

                yield self.env.timeout(proc_time)

    @property
    def wip(self):
        return sum(job.wip for job in self.doing)

    @property
    def done(self):
        return (job for job in self.jobs if job.finished)

    @property
    def doing(self):
        return (job for job in self.jobs if not job.finished)

    def sample(self):
        now = self.env.now
        if now > self.warmup and round(now % self.sampling) == 0:
            self.wips.append(self.wip)
            machines = (
                (m for m, _ in self.bottlenecks) if self.bottlenecks else self.machines
            )
            for machine in machines:
                self.norms[machine].append(self.norm(machine))
                self.queues[machine].append(self.workload(machine))


            


    @property
    def stats(self):
        wip_mean = statistics.mean(self.wips) if self.wips else 0
        try:
            lead_time = statistics.mean(job.lead_time for job in self.done)
        except statistics.StatisticsError:
            lead_time = 0

        clock = self.env.now or 1
        jobs_done = len(list(self.done))

        return {
            "clock": clock,
            "psp_queue": len(self.psp),
            "jobs_doing": len(list(self.doing)),
            "jobs_done": jobs_done,
            "th_rate": jobs_done / clock,
            "wip_mean": wip_mean,
            "lead_time": lead_time,
        }

    def status(self, job_to_insert):

        if self._status == 'base':
            psp = [
                sum(job.processing_time(m) for job in self.psp)
                for m in self.machines]

            doing = [
                sum(job.processing_time(m)
                    if not job.visited(m) else 0
                    for job in self.doing)
                for m in self.machines]

            return [*psp, *doing]

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
                self.doing,
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
                y = self.norms[machine]
                ax[0].plot(y, linestyle="-", linewidth=.8, color=color["color"])
                ax[0].set_ylabel(f"WC {machine}")

                y = self.queues[machine]
                ax[1].plot(y, linestyle="-", linewidth=.8, color=color["color"])

            fig.tight_layout()

        with plt.style.context("ggplot"):

            fig, axes = plt.subplots(
                1, 2, figsize=(8, (6 / n) + .3), sharey=True, sharex=True)

            axes[0].plot(
                self.wips, color=list(plt.rcParams["axes.prop_cycle"])[-1]["color"]
            )
            axes[0].set_ylabel("WIP")
            axes[0].set_title("System")

            fig.tight_layout()

        print(self)


class UniformRouting:
    
    def __init__(self, machines, flow=False):
        self.machines = machines
        self.flow = flow
    
    def __iter__(self):
        n_machines = random.randrange(1, len(self.machines))
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
