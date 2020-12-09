import logging
import simpy
from contextlib import ContextDecorator

def make_logger():
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    logger = logging.getLogger()
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger

def set_level(level):
    logger.setLevel(level)

class tracker(ContextDecorator):
    def __init__(self, flowshop, job, where):
        self.flowshop = flowshop
        self.job = job
        self.where = where

    def __enter__(self):
        logger.debug(f"[{self.flowshop.env.now}] {self.job.family!r} Entering {self.where} @ WC{self.job.active_machine}")
        logger.debug(f"[{self.flowshop.env.now}] WC3: Norm: {self.flowshop.norm('3')}, Queue: {self.flowshop.workload('3')}")
        logger.debug(f"[{self.flowshop.env.now}] WC3: Norm: {self.flowshop.norm('6')}, Queue: {self.flowshop.workload('6')}")

    def __exit__(self, exc_type, exc, exc_tb):
        logger.debug(f"[{self.flowshop.env.now}] {self.job.family!r} Exiting {self.where} @ WC{self.job.active_machine}")
        logger.debug(f"[{self.flowshop.env.now}] WC3: Norm: {self.flowshop.norm('3')}, Queue: {self.flowshop.workload('3')}")
        logger.debug(f"[{self.flowshop.env.now}] WC3: Norm: {self.flowshop.norm('6')}, Queue: {self.flowshop.workload('6')}")


logger = make_logger()


class Work(simpy.Process):
    def __init__(self, flowshop, job):
        super().__init__(flowshop.env, flowshop.work(job))

        self.job = job

        self.first_machine = job._machines[0]

        self.flowshop = flowshop

        self.in_wc3 = None
        self.in_wc6 = None
        self.out_ = None

    @property
    def flowshop_status(self):
        psp = sum(job.wip for job in self.flowshop.psp)
        wc3 = (self.flowshop.workload("3"), self.flowshop.norm("3"))
        wc6 = (self.flowshop.workload("6"), self.flowshop.norm("6"))
        return [round(self.flowshop.env.now, 4), psp, *wc3, *wc6]

    def _resume(self, event):

        if isinstance(event, simpy.resources.resource.Request):

            if self.job.active_machine == self.first_machine:
                self.in_ = self.flowshop_status

            if self.job.active_machine == "3":
                self.in_wc3 = self.flowshop_status

            if self.job.active_machine == "6":
                self.in_wc6 = self.flowshop_status

        # if isinstance(event, simpy.events.Timeout):
        #    print(f'[{self.job.env.now}] job {self.job.family} terminato su macchina {self.job.active_machine}')

        super()._resume(event)

        if not self.is_alive:
            self.out_ = self.flowshop_status

    @property
    def stats(self):
        return (self.in_, self.in_wc3, self.in_wc6, self.out_)
