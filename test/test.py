import unittest
import wlc


class WLCTestCase(unittest.TestCase):
    def setUp(self):

        build_params = wlc.get_build_params()
        env, router, shopfloor = wlc.build(**build_params)

        self.env = env
        self.router = router
        self.shopfloor = shopfloor

        wlc.run(env)

    def test_queues_vs_norms(self):
        """
        Verifica che per ciascuna macchina
        la coda effettiva sia minore o uguale alla norma
        """
        for machine in self.shopfloor.machines.keys():
            workload = self.shopfloor.workload(machine)
            norm = self.shopfloor.norm(machine)
            self.assertTrue(workload <= norm)

    def test_norms_vs_wip(self):
        """
        Verifica che la somma delle norme di ogni macchina
        sia uguale al WIP a livello di sistema
        """
        norms = sum(
            self.shopfloor.norm(machine) for machine in self.shopfloor.machines.keys()
        )
        wip = self.shopfloor.wip
        self.assertTrue(norms <= wip)

    def test_active_machines_vs_requests_queues(self):
        for machine in self.shopfloor.machines.keys():

            jobs_actives_at_machine = sum(
                job.active_machine == machine for job in self.shopfloor.jobs
            )

            requests_at_machines = len(self.shopfloor.machines[machine].queue) + len(
                self.shopfloor.machines[machine].users
            )

            self.assertTrue(jobs_actives_at_machine == requests_at_machines)

    def test_doing_vs_jobs_active(self):
        """
        Verifica la congruenza tra numero di job in corso di lavorazione
        `shopfloor.doing` e il numero di job che attestano di essere attivi
        presso qualche macchina `job.active_machine`)
        """
        jobs_actives_at_machine = 0
        for machine in self.shopfloor.machines.keys():

            jobs_actives_at_machine += sum(
                job.active_machine == machine for job in self.shopfloor.jobs
            )

        self.assertTrue(jobs_actives_at_machine == len(list(self.shopfloor.doing)))

    def test_working_on(self):
        """
        Verifica che l'attributo `working_on` dei job sia correttamente gestito
        ovvero che se `working_on==True` allora il tempo di entrata nella
        macchina + il tempo di processamento sulla macchina sia minore (o uguale?)
        all'attuale tempo di simulazione
        """
        for job in self.shopfloor.doing:
            if job.working_on:
                try:
                    self.assertTrue(
                        job.machine_entry_time(job.active_machine)
                        + job[job.active_machine]
                        > job.env.now
                    )
                except AssertionError:
                    print(job)
                    print(job.env.now)
                    raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
