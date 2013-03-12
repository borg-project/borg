"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import numpy
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def plan_to_start_end(category, planner_name, solver_names, plan):
    t = 0

    for (s, d) in plan:
        yield map(str, [category, planner_name, solver_names[s], t, t + d + 0.75])

        t += d + 1

def plans_to_per_bin(category, planner_name, solver_names, plans, B):
    #allocation = numpy.zeros((len(solver_names), B))

    #for plan in plans:
        #print plan
        #t = 0

        #for (s, d) in plan:
            #allocation[s, t:t + d + 1] += 1.0

            #t += d + 1

    #for (s, solver_name) in enumerate(solver_names):
        #for b in xrange(B):
            #yield [category, planner_name, solver_name, str(b), str(allocation[s, b])]

    for plan in plans:
        for (s, d) in plan:
            yield [category, planner_name, solver_names[s], str(d)]

def run_experiment(planner_name, bundle_path, category, individual):
    """Run a planning experiment."""

    logger.info("loading run data from %s", bundle_path)

    run_data = borg.RunData.from_bundle(bundle_path)

    logger.info("computing a plan over %i instances with %s", len(run_data), planner_name)

    if planner_name == "default":
        planner = borg.planners.default
    elif planner_name == "knapsack":
        planner = borg.planners.ReorderingPlanner(borg.planners.KnapsackPlanner())
    elif planner_name == "streeter":
        planner = borg.planners.StreeterPlanner()
    else:
        raise ValueError("unrecognized planner name: {0}".format(planner_name))

    B = 60
    bins = run_data.to_bins_array(run_data.solver_names, B).astype(numpy.double)
    bins[..., -2] += 1e-2 # if all else fails...
    rates = bins / numpy.sum(bins, axis = -1)[..., None]
    log_survival = numpy.log(1.0 + 1e-8 - numpy.cumsum(rates[..., :-1], axis = -1))

    if individual:
        plans = []

        for n in xrange(len(run_data)):
            plans.append(planner.plan(log_survival[n, :, :-1][None, ...]))

        rows = plans_to_per_bin(category, planner_name, run_data.solver_names, plans, B)
    else:
        plan = planner.plan(log_survival[..., :-1])
        rows = plan_to_start_end(category, planner_name, run_data.solver_names, plan)

    for row in rows:
        yield row

@borg.annotations(
    out_path = ("plan output path"),
    experiments = ("experiments to run", "positional", None, borg.util.load_json),
    individual = ("make individual plans?", "flag"),
    )
def main(out_path, experiments, individual = False):
    """Compute a plan as specified."""

    with borg.util.openz(out_path, "wb") as out_file:
        out_csv = csv.writer(out_file)

        if individual:
            #out_csv.writerow(["category", "planner", "solver", "bin", "count"])
            out_csv.writerow(["category", "planner", "solver", "length"])
        else:
            out_csv.writerow(["category", "planner", "solver", "start", "end"])

        for experiment in experiments:
            rows = \
                run_experiment(
                    experiment["planner"],
                    experiment["bundle"],
                    experiment["category"],
                    individual,
                    )

            out_csv.writerows(list(rows))

if __name__ == "__main__":
    borg.script(main)

