"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import json
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class PortfolioMaker(object):
    def __init__(self, portfolio_name):
        self.name = portfolio_name

    def __call__(self, suite, train_data):
        if self.name == "random":
            portfolio = borg.portfolios.RandomPortfolio()
        elif self.name == "uniform":
            portfolio = borg.portfolios.UniformPortfolio()
        elif self.name == "baseline":
            portfolio = borg.portfolios.BaselinePortfolio(suite, train_data)
        elif self.name == "oracle":
            portfolio = borg.portfolios.OraclePortfolio()
        elif self.name == "preplanning":
            portfolio = borg.portfolios.PreplanningPortfolio(suite, train_data)
        elif self.name == "probabilistic":
            B = 60
            T = 1

            #model = borg.models.MulDirMixSampler().fit(train_data.solver_names, train_data, B, T)
            model = borg.models.MulSampler().fit(train_data.solver_names, train_data, B, T)
            regress = borg.regression.LinearRegression(train_data, model)
            portfolio = borg.portfolios.PureModelPortfolio(suite, model, regress)
        else:
            raise ValueError("unrecognized portfolio name: {0}".format(self.name))

        return borg.solver_io.RunningPortfolioFactory(portfolio, suite)

class SolverMaker(object):
    def __init__(self, solver_name):
        self.name = solver_name

    def __call__(self, suite, train_data):
        return suite.solvers[self.name]

def simulate_run(run, maker, train_data, test_data):
    """Simulate portfolio execution on a train/test split."""

    split_id = uuid.uuid4()
    budget = test_data.common_budget
    suite = borg.fake.FakeSuite(test_data)
    solver = maker(suite, train_data)
    rows = []

    for instance_id in test_data.run_lists:
        logger.info("simulating run on %s", instance_id)

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

            rows.append([run["category"], instance, maker.name, budget, cpu_cost, success_str, None, split_id])

    rows.append([run["category"], None, maker.name, budget, budget, "FALSE", None, split_id])

    return rows

@borg.annotations(
    out_path = ("results CSV output path"),
    runs_path = ("path to JSON runs specification"),
    repeats = ("number of times to repeat each run", "option", None, int),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, runs_path, repeats = 4, workers = 0, local = False):
    """Simulate portfolio and solver behavior."""

    with open(runs_path) as runs_file:
        runs = json.load(runs_file)

    logger.info("simulating %i runs", len(runs) * repeats)

    run_data_sets = {}

    def get_run_data(path):
        data = run_data_sets.get(path)

        if data is None:
            run_data_sets[path] = data = borg.storage.RunData.from_bundle(path)

        return data

    def yield_jobs():
        for run in runs:
            train_data = get_run_data(run["train_bundle"])
            test_data = get_run_data(run["test_bundle"])

            if run["portfolio_name"] == "-":
                makers = map(SolverMaker, train_data.solver_names)
            else:
                makers = [PortfolioMaker(run["portfolio_name"])]

            for maker in makers:
                for _ in xrange(repeats):
                    yield (simulate_run, [run, maker, train_data, test_data])

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["category", "path", "solver", "budget", "cost", "success", "answer", "split"])

        for (_, rows) in condor.do(yield_jobs(), workers, local):
            writer.writerows(rows)

if __name__ == "__main__":
    borg.script(main)

