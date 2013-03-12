"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class PortfolioMaker(object):
    def __init__(self, portfolio_name, bins = 60):
        name_parts = portfolio_name.split(":")

        self.name = portfolio_name
        self.subname = name_parts[0]
        self.variants = name_parts[1:]
        self._bins = bins

    def __call__(self, suite, train_data, test_data = None, model_kwargs = {}):
        full_data = train_data

        logger.info("making portfolio %s; model_kwargs: %s", self.name, model_kwargs)

        if "knapsack" in self.variants:
            planner = borg.planners.ReorderingPlanner(borg.planners.KnapsackPlanner())
        elif "streeter" in self.variants:
            planner = borg.planners.StreeterPlanner()
        else:
            planner = borg.planners.default

        if self.subname == "random":
            portfolio = borg.portfolios.RandomPortfolio()
        elif self.subname == "uniform":
            portfolio = borg.portfolios.UniformPortfolio()
        elif self.subname == "baseline":
            portfolio = borg.portfolios.BaselinePortfolio(suite, train_data)
        elif self.subname == "oracle":
            portfolio = borg.portfolios.OraclePortfolio()
        else:
            bins = self._bins

            if self.subname.endswith("-mul"):
                estimator = borg.models.MulEstimator(**model_kwargs)
            elif self.subname.endswith("-dir"):
                estimator = borg.models.MulDirMatMixEstimator(**model_kwargs)
            elif self.subname.endswith("-log"):
                estimator = borg.models.DiscreteLogNormalMatMixEstimator(**model_kwargs)
            else:
                raise ValueError("unrecognized portfolio subname: {0}".format(self.subname))

            train_data = train_data.only_nontrivial(train_data.common_budget / bins) # XXX ?
            model = estimator(train_data, bins, full_data)

            if self.subname.startswith("preplanning-"):
                portfolio = borg.portfolios.PreplanningPortfolio(suite, model, planner = planner)
            elif self.subname.startswith("probabilistic-"):
                regress = borg.regression.NearestRTDRegression(model)
                portfolio = borg.portfolios.PureModelPortfolio(suite, model, regress)
            else:
                raise ValueError("unrecognized portfolio subname: {0}".format(self.subname))

        return borg.solver_io.RunningPortfolioFactory(portfolio, suite)

class SolverMaker(object):
    def __init__(self, solver_name):
        self.name = solver_name
        self.subname = solver_name

    def __call__(self, suite, train_data, test_data = None, model_kwargs = {}):
        return suite.solvers[self.name]

def simulate_run(run, maker, train_data, test_data):
    """Simulate portfolio execution on a train/test split."""

    split_id = uuid.uuid4()
    budget = test_data.common_budget
    #budget = test_data.common_budget / 4
    suite = borg.fake.FakeSuite(test_data)
    solver = maker(suite, train_data, test_data)
    rows = []

    for (i, instance_id) in enumerate(test_data.run_lists):
        logger.info("simulating run %i/%i on %s", i, len(test_data), instance_id)

        with suite.domain.task_from_path(instance_id) as instance:
            with borg.accounting() as accountant:
                answer = solver.start(instance).run_then_stop(budget)

            succeeded = suite.domain.is_final(instance, answer)
            cpu_cost = accountant.total.cpu_seconds

            logger.info(
                "%s %s on %s (%.2f CPU s)",
                maker.name,
                "succeeded" if succeeded else "failed",
                os.path.basename(instance),
                cpu_cost,
                )

            success_str = "TRUE" if succeeded else "FALSE"

            rows.append([run["category"], maker.name, budget, cpu_cost, success_str, split_id])

    return rows

@borg.annotations(
    out_path = ("results CSV output path"),
    runs = ("path to JSON runs specification", "positional", None, borg.util.load_json),
    repeats = ("number of times to repeat each run", "option", None, int),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, runs, repeats = 5, workers = 0, local = False):
    """Simulate portfolio and solver behavior."""

    logger.info("simulating %i runs", len(runs) * repeats)

    get_run_data = borg.util.memoize(borg.storage.RunData.from_bundle)

    def yield_jobs():
        for run in runs:
            train_data = get_run_data(run["train_bundle"])

            if run.get("only_nontrivial", False):
                train_data = train_data.only_nontrivial()

            if run["test_bundle"] == "-":
                validation = sklearn.cross_validation.KFold(len(train_data), repeats, indices = False)
                data_sets = [(train_data.masked(v), train_data.masked(e)) for (v, e) in validation]
            else:
                test_data = get_run_data(run["test_bundle"])

                if run.get("only_nontrivial", False):
                    test_data = test_data.only_nontrivial()

                data_sets = [(train_data, test_data)] * repeats

            if run["portfolio_name"] == "-":
                makers = map(SolverMaker, train_data.solver_names)
            else:
                makers = [PortfolioMaker(run["portfolio_name"])]

            for maker in makers:
                for (train_fold_data, test_fold_data) in data_sets:
                    yield (simulate_run, [run, maker, train_fold_data, test_fold_data])

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["category", "solver", "budget", "cost", "success", "split"])

        for (_, rows) in condor.do(yield_jobs(), workers, local):
            writer.writerows(rows)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

