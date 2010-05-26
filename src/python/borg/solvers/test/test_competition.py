"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    timed,
    assert_equal,
    )

def test_sat_competition_solver():
    """
    Test the SAT competition-binary solver.
    """

    # define various fake solvers
    finds_sat_solver     = \
"""
print "c comment"
print "s SATISFIABLE"
print "v 1 2 3 4"
print "v 5 6 7 8 0"
raise SystemExit(10)
"""
    finds_unsat_solver   = \
"""
print "c comment"
print "s UNSATISFIABLE"
raise SystemExit(20)
"""
    finds_unknown_solver = \
"""
print "c comment"
"""
    times_out_solver     = \
"""
print "c comment"
while True: pass
"""

    # each test instance is similar
    @timed(32.0)
    def test_solver(solver_code, answer):
        """
        Test the SAT competition-binary solver.
        """

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix = ".py") as code_file:
            # write the fake solver
            code_file.write(solver_code)
            code_file.flush()

            # build the solver interface
            from borg.solvers import CompetitionSolver

            solver = CompetitionSolver(["python", code_file.name])

            # run the solver
            from cargo.temporal import TimeDelta
            from borg.tasks     import FileTask

            task    = FileTask("/tmp/path_irrelevant.cnf")
            budget  = TimeDelta(seconds = 8.0)
            attempt = solver.solve(task, budget, None, None)

            # verify its response
            assert_equal(attempt.answer, answer)

    # run each test
    from borg.sat import Decision

    yield (test_solver, finds_sat_solver,     Decision(True,  range(1, 9) + [0]))
    yield (test_solver, finds_unsat_solver,   Decision(False, None))
    yield (test_solver, finds_unknown_solver, None)
    yield (test_solver, times_out_solver,     None)

