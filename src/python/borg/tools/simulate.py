"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import csv
import uuid
import condor
import borg

logger = borg.get_logger(__name__)

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
        else:
            B = 10
            T = 1

            if self.name == "mul":
                model = borg.models.MulSampler().fit(train_data.solver_names, train_data, B, T)
            elif self.name == "mul-dir":
                model = borg.models.MulDirSampler().fit(train_data.solver_names, train_data, B, T)
            elif self.name == "mul-dirmix":
                model = borg.models.MulDirMixSampler().fit(train_data.solver_names, train_data, B, T)
            else:
                raise ValueError("unrecognized portfolio name")

            regress = borg.regression.LinearRegression(train_data, model)
            portfolio = borg.portfolios.PureModelPortfolio(suite, model, regress)

        return borg.solver_io.RunningPortfolioFactory(portfolio, suite)

class SolverMaker(object):
    def __init__(self, solver_name):
        self.name = solver_name

    def __call__(self, suite, train_data):
        return suite.solvers[self.name]

def simulate_split(maker, train_data, test_data):
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

            rows.append([instance, maker.name, budget, cpu_cost, success_str, None, split_id])

    rows.append([None, maker.name, budget, budget, "FALSE", None, split_id])

    return rows

@borg.annotations(
    out_path = ("results output path"),
    portfolio_name = ("name of the portfolio to train"),
    train_bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    test_bundle = ("path to pre-recorded runs", "positional", None, os.path.abspath),
    single = ("run a single solver", "option"),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(
    out_path,
    portfolio_name,
    train_bundle,
    test_bundle,
    single = None,
    workers = 0,
    local = False,
    ):
    """Simulate portfolio and solver behavior."""

    # generate jobs
    def yield_runs():
        train_data = borg.storage.RunData.from_bundle(train_bundle)
        test_data = borg.storage.RunData.from_bundle(test_bundle)

        if portfolio_name == "-":
            if single is None:
                makers = map(SolverMaker, train_data.solver_names)
            else:
                makers = map(SolverMaker, [single])
        else:
            makers = [PortfolioMaker(portfolio_name)]

        for maker in makers:
            for _ in xrange(4):
                yield (simulate_split, [maker, train_data, test_data])

    # and run them
    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["path", "solver", "budget", "cost", "success", "answer", "split"])

        condor.do(yield_runs(), workers, lambda _, r: writer.writerows(r), local)

if __name__ == "__main__":
    borg.script(main)

