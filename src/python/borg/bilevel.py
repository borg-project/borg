"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import time
import uuid
import multiprocessing
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

class MixturePortfolio(object):
    """Hybrid mixture-model portfolio."""

    def __init__(self, bundle, training, budget_interval):
        """Initialize."""

        # acquire running time data
        self._solver_names = list(bundle.solvers)
        self._solver_name_index = dict(map(reversed, enumerate(self._solver_names)))

        logger.info("solvers: %s", dict(enumerate(self._solver_names)))

        # fit our model
        self._model = \
            borg.models.DeltaModel.fit(
                self._solver_name_index,
                training.get_run_lists().values(),
                )
        self._planner = \
            borg.planners.KnapsackMultiversePlanner(
                self._solver_name_index,
                budget_interval,
                )

    def __call__(self, task, bundle, budget, cores = 1):
        """Run the portfolio."""

        with borg.accounting():
            return self._solve(task, bundle, budget, cores)

    def _solve(self, task, bundle, budget, cores):
        """Run the portfolio."""

        # select a solver
        S = len(self._solver_names)
        queue = multiprocessing.Queue()
        running = {}
        paused = []
        failed = []
        answer = None

        while True:
            # obtain model predictions
            failures = []

            for solver in failed + paused + running.values():
                failures.append((solver.s, solver.cpu_budgeted))

            posterior = self._model.condition(failures)

            # make a plan...
            remaining = budget - borg.get_accountant().total
            normal_cpu_budget = borg.machine_to_normal(borg.unicore_cpu_budget(remaining))
            raw_plan = self._planner.plan(posterior, normal_cpu_budget)

            # XXX use up all remaining time
            if len(raw_plan) == 0:
                break

            # interpret the plan's first action
            (a, planned_cpu_cost) = raw_plan[0]

            if a >= S:
                solver = paused.pop(a - S)
                s = solver.s
                name = self._solver_names[s]
            else:
                s = a
                name = self._solver_names[s]
                solver = bundle.solvers[name](task, queue, uuid.uuid4())
                solver.s = s
                solver.cpu_cost = 0.0

            # be informative
            logger.info(
                "running %s@%i for %i with %i remaining" % (
                    name,
                    borg.normal_to_machine(solver.cpu_cost),
                    borg.normal_to_machine(planned_cpu_cost),
                    remaining.cpu_seconds,
                    ),
                )

            # ... and follow through
            solver.unpause_for(borg.normal_to_machine(planned_cpu_cost))

            running[solver._solver_id] = solver

            solver.cpu_budgeted = solver.cpu_cost + planned_cpu_cost

            if len(running) == cores:
                response = queue.get()

                if isinstance(response, Exception):
                    raise response
                else:
                    (solver_id, run_cpu_seconds, answer, terminated) = response

                borg.get_accountant().charge_cpu(run_cpu_seconds)

                solver = running.pop(solver_id)

                solver.cpu_cost += borg.machine_to_normal(run_cpu_seconds)

                if bundle.domain.is_final(task, answer):
                    break
                elif terminated:
                    failed.append(solver)
                else:
                    paused.append(solver)

        for process in paused + running.values():
            process.stop()

        return answer

borg.portfolios.named["mixture"] = MixturePortfolio

