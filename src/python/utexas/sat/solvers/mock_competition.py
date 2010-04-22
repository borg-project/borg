"""
utexas/sat/solvers/mock_competition.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.solvers.base import SAT_Solver

class SAT_MockSolver(SAT_Solver):
    """
    Fake solver behavior from data.
    """

    def solve(self, input_path, cutoff = None, seed = None):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # FIXME support no-cutoff operation

    def __get_outcome_matrix(self):
        """
        Build a matrix of outcome probabilities.
        """

        import numpy

        log.info("building task-action-outcome matrix")

        # hit the database
        from contextlib               import closing
        from sqlalchemy.sql.functions import count

        session = ResearchSession()

        with closing(session):
            counts = numpy.zeros((self.ntasks, self.nactions, self.noutcomes))

            for action in self.actions:
                run_case  = case([(SAT_SolverRun.proc_elapsed <= action.cutoff, SAT_SolverRun.satisfiable)])
                statement =                                                           \
                    select(
                        [
                            SAT_SolverRun.task_uuid,
                            run_case.label("result"),
                            count(),
                            ],
                        and_(
                            SAT_SolverRun.task_uuid.in_([t.task.uuid for t in self.tasks]),
                            SAT_SolverRun.solver        == action.solver,
                            SAT_SolverRun.cutoff        >= action.cutoff,
                            ),
                        )                                                             \
                        .group_by(SAT_SolverRun.task_uuid, "result")
                executed  = session.connection().execute(statement)

                # build the matrix
                world_tasks = dict((t.task.uuid, t) for t in self.tasks)
                total_count = 0

                for (task_uuid, result, nrows) in executed:
                    # map storage instances to world instances
                    world_task    = world_tasks[task_uuid]
                    world_outcome = SAT_Outcome.BY_VALUE[result]

                    # record the outcome count
                    counts[world_task.n, action.n, world_outcome.n]  = nrows
                    total_count                                     += nrows

                if total_count == 0:
                    log.warning("no rows found for action %s", action, t.task.uuid)

            norms = numpy.sum(counts, 2, dtype = numpy.float)

            return counts / norms[:, :, numpy.newaxis]

