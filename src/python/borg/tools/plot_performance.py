"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.plot_performance import main

    raise SystemExit(main())

from cargo.log import get_logger

log = get_logger(__name__, default_level = "INFO")

def plot_trial(session, trial_row):
    """
    Plot the specified trial.
    """

    # get the relevant attempts
    from sqlalchemy import and_
    from borg.data  import RunAttemptRow

    attempt_rows =                                        \
        session                                           \
        .query(RunAttemptRow)                             \
        .filter(
            and_(
                RunAttemptRow.trials.contains(trial_row),
                RunAttemptRow.answer != None,
                ),
            )                                             \
        .order_by(RunAttemptRow.cost)

    # break them into series
    attempts = {}
    budget   = None

    for attempt_row in attempt_rows:

        solver_name     = attempt_row.solver_name
        solver_attempts = attempts.get(solver_name, [])

        solver_attempts.append(attempt_row.cost)

        attempts[solver_name] = solver_attempts

        # determine the budget
        if budget is None:
            budget = attempt_row.budget
        else:
            if budget != attempt_row.budget:
                raise RuntimeError("multiple budgets in trial")

    session.commit()

    # plot the series
    import pylab

    pylab.title("Solver Performance (Trial %s)" % trial_row.uuid)

    for (name, costs) in attempts.iteritems():
        x_values = [0.0] + [c.as_s for c in costs] + [budget.as_s]
        y_values = range(len(costs) + 1) + [len(costs)]

        pylab.plot(x_values, y_values, label = name)

    pylab.legend()
    pylab.show()

def main():
    """
    Run the script.
    """

    # get command line arguments
    import borg.data

    from cargo.flags import parse_given

    (trial_uuid,) = parse_given(usage = "%prog <trial_uuid> [options]")

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "DETAIL")

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            # get the trial
            from borg.data import TrialRow

            trial_row = session.query(TrialRow).get(trial_uuid)

            if trial_row is None:
                raise ValueError("no such trial")

            # and plot it
            plot_trial(session, trial_row)

