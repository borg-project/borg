"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

# XXX feature computation time

import os.path
import csv
import uuid
import numpy
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class CustomEstimator(object):
    def __call__(self, run_data, bins, full_data):
        """Fit parameters of the log-normal linked mixture model."""

        # ...
        counts_NSD = run_data.to_bins_array(run_data.solver_names, bins)
        budget = run_data.get_common_budget()
        interval = budget / bins

        (N, S, D) = counts_NSD.shape

        # estimate training RTDs
        T = 10
        samples_TSD = numpy.zeros((T, S, D))
        h = 10

        samples_TSD[0, :, -1] = 1.0
        samples_TSD[1, 0, -1] = 1.0
        samples_TSD[1, 1:, 0] = 1.0
        samples_TSD[2, 0, -1] = 1.0
        samples_TSD[2, 1, -1] = 1.0
        samples_TSD[2, 2, 0] = 1.0
        samples_TSD[3, 0, 0] = 1.0
        samples_TSD[3, 1, -1] = 1.0
        samples_TSD[3, 2, -1] = 1.0
        samples_TSD[4, :, 0] = 1.0
        samples_TSD[5, 0, -1] = 1.0
        samples_TSD[5, 1, -1] = 1.0
        samples_TSD[5, 2, h] = 1.0
        samples_TSD[6, 0, -1] = 1.0
        samples_TSD[6, 1, h] = 1.0
        samples_TSD[6, 2, -1] = 1.0
        samples_TSD[7, 0, h] = 1.0
        samples_TSD[7, 1, -1] = 1.0
        samples_TSD[7, 2, -1] = 1.0
        samples_TSD[8, 0, -1] = 1.0
        samples_TSD[8, 1:, h] = 1.0
        samples_TSD[9, :2, h] = 1.0
        samples_TSD[9, 2, -1] = 1.0

        samples_TSD += 1e-4
        samples_TSD /= numpy.sum(samples_TSD, axis = -1)[..., None]

        weights_T = numpy.empty(T)

        weights_T[0] = 8
        weights_T[1] = 3
        weights_T[2] = 1
        weights_T[3] = 6
        weights_T[4] = 2
        weights_T[5] = 3
        weights_T[6] = 2
        weights_T[7] = 2
        weights_T[8] = 2
        weights_T[9] = 1

        weights_T /= numpy.sum(weights_T)

        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            #print samples_TSD
            #raise SystemExit()

        return \
            borg.models.MultinomialModel(
                interval,
                borg.statistics.to_log_survival(samples_TSD, axis = -1),
                log_masses = borg.statistics.floored_log(samples_TSD),
                log_weights = numpy.log(weights_T),
                )

class PortfolioMaker(object):
    def __init__(self, portfolio_name):
        name_parts = portfolio_name.split(":")

        self.name = portfolio_name
        self.subname = name_parts[0]
        self.variants = name_parts[1:]

    def __call__(self, suite, train_data, instances):
        full_data = train_data

        if instances is not None:
            ids = sorted(train_data.run_lists, key = lambda _: numpy.random.rand())[:instances]
            train_data = train_data.filter(*ids)

        if "10.sys" in self.variants:
            train_data = train_data.collect_systematic([1, 0]).only_nonempty()
        elif "10.ind" in self.variants:
            train_data = train_data.collect_independent([1, 0]).only_nonempty()

        #with borg.util.numpy_printing(precision = 2, suppress = True, linewidth = 200, threshold = 1000000):
            #print train_data.to_bins_array(train_data.solver_names, 30)

        #raise SystemExit()

        if self.subname == "random":
            portfolio = borg.portfolios.RandomPortfolio()
        elif self.subname == "uniform":
            portfolio = borg.portfolios.UniformPortfolio()
        elif self.subname == "baseline":
            portfolio = borg.portfolios.BaselinePortfolio(suite, train_data)
        elif self.subname == "oracle":
            portfolio = borg.portfolios.OraclePortfolio()
        else:
            bins = 30

            if self.subname.endswith("-mul"):
                model = borg.models.MulEstimator()(train_data, bins, full_data)
            elif self.subname.endswith("-dir"):
                model = borg.models.MulDirMatMixEstimator()(train_data, bins, full_data)
            elif self.subname.endswith("-log"):
                #model = borg.models.LogNormalMixEstimator()(train_data, bins)
                #model = borg.models.DiscreteLogNormalMixEstimator()(train_data, bins)
                model = borg.models.DiscreteLogNormalMatMixEstimator()(train_data, bins, full_data)
            elif self.subname.endswith("-custom"):
                model = CustomEstimator()(train_data, bins, full_data)
            else:
                raise ValueError("unrecognized portfolio subname: {0}".format(self.subname))

            if self.subname.startswith("preplanning-"):
                portfolio = borg.portfolios.PreplanningPortfolio(suite, model)
            elif self.subname.startswith("probabilistic-"):
                regress = borg.regression.ClusteredLogisticRegression(model)
                #regress = borg.regression.NeighborsRegression(model)
                portfolio = borg.portfolios.PureModelPortfolio(suite, model, regress)
            else:
                raise ValueError("unrecognized portfolio subname: {0}".format(self.subname))

        return borg.solver_io.RunningPortfolioFactory(portfolio, suite)

class SolverMaker(object):
    def __init__(self, solver_name):
        self.name = solver_name

    def __call__(self, suite, train_data, instances):
        return suite.solvers[self.name]

def simulate_run(run, maker, train_data, test_data, instances):
    """Simulate portfolio execution on a train/test split."""

    split_id = uuid.uuid4()
    budget = test_data.common_budget
    suite = borg.fake.FakeSuite(test_data)
    solver = maker(suite, train_data, instances)
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

            rows.append([run["category"], maker.name, budget, cpu_cost, success_str, split_id, instances])

    return rows

@borg.annotations(
    out_path = ("results CSV output path"),
    runs = ("path to JSON runs specification", "positional", None, borg.util.load_json),
    repeats = ("number of times to repeat each run", "option", None, int),
    workers = ("submit jobs?", "option", "w", int),
    local = ("workers are local?", "flag"),
    )
def main(out_path, runs, repeats = 4, workers = 0, local = False):
    """Simulate portfolio and solver behavior."""

    logger.info("simulating %i runs", len(runs) * repeats)

    get_run_data = borg.util.memoize(borg.storage.RunData.from_bundle)

    def yield_jobs():
        for run in runs:
            train_data = get_run_data(run["train_bundle"])

            if run["test_bundle"] is "-":
                validation = sklearn.cross_validation.KFold(len(train_data), repeats, indices = False)
                data_sets = [(train_data.masked(v), train_data.masked(e)) for (v, e) in validation]
            else:
                data_sets = [(train_data, get_run_data(run["test_bundle"]))] * repeats

            if run["portfolio_name"] == "-":
                makers = map(SolverMaker, train_data.solver_names)
            else:
                makers = [PortfolioMaker(run["portfolio_name"])]

            for maker in makers:
                for (train_data, test_data) in data_sets:
                    yield (simulate_run, [run, maker, train_data, test_data, None])

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["category", "solver", "budget", "cost", "success", "split", "instances"])

        for (_, rows) in condor.do(yield_jobs(), workers, local):
            writer.writerows(rows)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

