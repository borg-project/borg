"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.plot_performance import main

    raise SystemExit(main())

from cargo.log import get_logger

log = get_logger(__name__, default_level = "INFO")

def get_attempt_data(session, trial_row):
    """
    Get relevant attempt data from a trial.
    """

    from sqlalchemy import and_
    from borg.data  import (
        DecisionRow,
        RunAttemptRow,
        )

    return                                                       \
        session                                                  \
        .query(
            RunAttemptRow.solver_name,
            RunAttemptRow.budget,
            RunAttemptRow.cost,
            DecisionRow.satisfiable,
            )                                                    \
        .filter(
            and_(
                RunAttemptRow.trials.contains(trial_row),
                RunAttemptRow.answer_uuid == DecisionRow.uuid,
                ),
            )                                                    \
        .order_by(RunAttemptRow.cost)                            \
        .all()

def plot_trial(session, trial_rows):
    """
    Plot the specified trial.
    """

    # get attempts and break them into series
    rows       = sum((get_attempt_data(session, r) for r in trial_rows), [])
    costs      = {}
    the_budget = None

    for (solver_name, budget, cost, satisfiable) in rows:
        # store the cost and answer
        solver_costs = costs.get(solver_name, [])

        solver_costs.append((cost, satisfiable))

        costs[solver_name] = solver_costs

        # determine the budget
        if the_budget is None:
            the_budget = budget
        else:
            if the_budget != budget:
                raise RuntimeError("multiple budgets in trial")

    # plot the series
    import pylab

    from cargo.plot import get_color_list

    colors = get_color_list(len(costs))

    for (i, (name, costs)) in enumerate(costs.iteritems()):
        # set up the coordinates
        x_values      = [0.0] + [c.as_s for (c, _) in costs] + [budget.as_s]
        y_values      = range(len(costs) + 1) + [len(costs)]
        tick_x_values = {True: [], False: []}
        tick_y_values = {True: [], False: []}

        for (j, (_, satisfiable)) in enumerate(costs):
            tick_x_values[satisfiable].append(x_values[j + 1])
            tick_y_values[satisfiable].append(y_values[j + 1])

        # then plot them
        color = colors[i, :]

        pylab.plot(x_values, y_values, label = name, c = color)
        pylab.plot(tick_x_values[True], tick_y_values[True], marker = "+", c = color, ls = "None")
        pylab.plot(tick_x_values[False], tick_y_values[False], marker = "x", c = color, ls = "None")

    pylab.title("Solver Performance")
    pylab.legend(loc = "lower right")
    pylab.show()

def main():
    """
    Run the script.
    """

    # get command line arguments
    import borg.data

    from cargo.flags import parse_given

    trial_uuids = parse_given(usage = "%prog [options] <trial_uuid> [...]")

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

            trial_rows = [session.query(TrialRow).get(u) for u in trial_uuids]

            if None in trial_rows:
                raise ValueError("at least one trial could not be round")

            # and plot it
            plot_trial(session, trial_rows)

