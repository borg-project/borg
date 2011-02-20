"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.solvers import (
    RunAttempt,
    AbstractSolver,
    )

log = get_logger(__name__)

class ZeframRunAttempt(RunAttempt):
    """
    Outcome of a Zefram run.
    """

    def __init__(
        self,
        solver,
        task,
        answer,
        seed,
        generation_run,
        compilation_run,
        solution_run,
        ):
        """
        Initialize.
        """

        Attempt.__init__(
            self,
            solver,
            run.limit,
            sum([
                generation_run.proc_elapsed,
                compilation_run.proc_elapsed,
                solution_run.proc_elapsed,
                ]),
            task,
            answer,
            )

        self._seed = seed
        self._generation_run  = generation_run
        self._compilation_run = compilation_run
        self._solution_run    = solution_run

    def get_new_row(self, session, solver_row = None):
        """
        Return a database description of this result.
        """

        from borg.data import ZeframAttemptRow

        return self._get_new_row(session, ZeframAttemptRow, solver_row = solver_row)

    def _get_new_row(self, session, Row, solver_row = None, **kwargs):
        """
        Create or obtain an ORM row for this object.
        """

        from borg.data import CPU_LimitedRunRow

        if solver_row is None:
            solver_row = self._solver.get_row(session)

        return \
            Attempt._get_new_row(
                self,
                session,
                Row,
                solver = solver_row,
                seed   = self._seed,
                run    = CPU_LimitedRunRow.from_run(self._solution_run),
                **kwargs
                )

    @property
    def seed(self):
        """
        The PRNG seed used for the run.
        """

        return self._seed

    @property
    def run(self):
        """
        The details of the run.
        """

        return self._solution_run

class ZeframSolver(Rowed, AbstractSolver):
    """
    Interface to the Zefram solver.
    """

    def __init__(self, command, relative = ""):
        """
        Initialize this solver.
        """

        # base
        Rowed.__init__(self)

        # members
        self._command  = command
        self._relative = relative

    def solve(self, task, budget, random, environment):
        """
        Execute the solver and return its outcome.
        """

        # argument sanity
        from borg.tasks import AbstractFileTask

        assert isinstance(task, AbstractFileTask)

        # prepare to run the solver
        from cargo.io import mkdtemp_scoped

        remaining = budget

        with mkdtemp_scoped(prefix = "borg.solvers.zefram.") as tmpdir:
            # generate bitcode
            from os.path               import join
            from cargo.unix.accounting import run_cpu_limited
            from borg.solvers          import get_random_seed

            seed               = get_random_seed(random)
            bitcode_path       = join(tmpdir, "solve_instance.bc")
            generation_command = [
                self._command.replace("$HERE", self._relative),
                task.path,
                str(seed),
                bitcode_path,
                ]

            log.note("generating LLVM bitcode: %s", generation_command)

            generation_run = \
                run_cpu_limited(
                    generation_command,
                    remaining,
                    pty         = True,
                    environment = { "TMPDIR" : tmpdir },
                    )

            # compile bitcode
            from datetime import timedelta

            target_path         = join(tmpdir, "solve_instance")
            compilation_command = ["llvmc", bitcode_path, "-o", target_path]

            log.note("compiling LLVM bitcode: %s", compilation_command)

            remaining -= generation_run.proc_elapsed

            if remaining <= timedelta():
                compilation_run = None
            else:
                compilation_run = \
                    run_cpu_limited(
                        compilation_command,
                        remaining,
                        pty         = True,
                        environment = { "TMPDIR" : tmpdir },
                        )

            # run compiled solver
            solution_command = [target_path]

            log.note("running compiled solver: %s", solution_command)

            if compilation_run is None:
                solution_run = None
            else:
                remaining -= compilation_run.proc_elapsed

                if remaining <= timedelta():
                    solution_run = None
                else:
                    solution_run = \
                        run_cpu_limited(
                            solution_command,
                            remaining,
                            pty         = True,
                            environment = { "TMPDIR" : tmpdir },
                            )

        # return our attempt
        # FIXME doesn't bother to decode the answer
        # FIXME doesn't use ZeframAttempt
        if solution_run is not None:
            last_run = solution_run
        elif compilation_run is not None:
            last_run = compilation_run
        else:
            last_run = generation_run

        return RunAttempt(self, task, None, seed, last_run)

    @property
    def seeded(self):
        """
        Is the solver seeded?
        """

        return True

