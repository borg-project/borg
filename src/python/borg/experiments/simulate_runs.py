"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

# XXX feature computation time

import os.path
import csv
import uuid
import random
import sklearn
import condor
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

class PortfolioMaker(object):
    def __init__(self, portfolio_name):
        name_parts = portfolio_name.split(":")

        self.name = portfolio_name
        self.subname = name_parts[0]
        self.variants = name_parts[1:]

    def __call__(self, suite, train_data, instances):
        #train_data = train_data.only_successful().collect_systematic([4])
        #train_data = train_data.only_successful()

        # XXX
        full_data = train_data

        if instances is not None:
            ids = sorted(train_data.run_lists, key = lambda _: random.random())[:instances]
            train_data = train_data.filter(*ids)

        if "10.sys" in self.variants:
            train_data = train_data.collect_systematic([1, 0]).only_nonempty()
        elif "2000.sys" in self.variants:
            train_data = train_data.collect_systematic([2, 0, 0, 0]).only_nonempty()
        elif "10.ind" in self.variants:
            train_data = train_data.collect_independent([1, 0]).only_nonempty()

        #if "1" in self.variants:
            #train_data = train_data.collect_systematic([1])
        #elif "2" in self.variants:
            #train_data = train_data.collect_systematic([2])
        #elif "3" in self.variants:
            #train_data = train_data.collect_systematic([3])
        #elif "4" in self.variants:
            #train_data = train_data.collect_systematic([4])
        #elif "5" in self.variants:
            #train_data = train_data.collect_systematic([5])
        #elif "6" in self.variants:
            #train_data = train_data.collect_systematic([6])

        if self.subname == "random":
            portfolio = borg.portfolios.RandomPortfolio()
        elif self.subname == "uniform":
            portfolio = borg.portfolios.UniformPortfolio()
        elif self.subname == "baseline":
            portfolio = borg.portfolios.BaselinePortfolio(suite, train_data)
        elif self.subname == "oracle":
            portfolio = borg.portfolios.OraclePortfolio()
        else:
            bins = 60

            if self.subname.endswith("-mul"):
                model = borg.models.MulEstimator()(train_data, bins, full_data)
            elif self.subname.endswith("-dir"):
                #sampler = borg.models.MulDirMixSampler()
                #model = \
                    #borg.models.mean_posterior(
                        #sampler,
                        #train_data.solver_names,
                        #train_data,
                        #bins = bins,
                        #)
                model = borg.models.MulDirMatMixEstimator()(train_data, bins, full_data)
            elif self.subname.endswith("-log"):
                #model = borg.models.LogNormalMixEstimator()(train_data, bins)
                #model = borg.models.DiscreteLogNormalMixEstimator()(train_data, bins)
                model = borg.models.DiscreteLogNormalMatMixEstimator()(train_data, bins)
            else:
                raise ValueError("unrecognized portfolio subname: {0}".format(self.subname))

            if self.subname.startswith("preplanning-"):
                portfolio = borg.portfolios.PreplanningPortfolio(suite, model)
                #portfolio = borg.portfolios.PureModelPortfolio(suite, model)
            elif self.subname.startswith("probabilistic-"):
                #regress = borg.regression.ClusteredLogisticRegression(model)
                regress = borg.regression.NeighborsRegression(model)
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

    # make explicit the number of test runs
    #test_data = test_data.collect_systematic([8])

    split_id = uuid.uuid4()
    budget = test_data.common_budget
    suite = borg.fake.FakeSuite(test_data)
    solver = maker(suite, train_data, instances)
    successes = 0
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

            #if succeeded:
                #successes += 1

    return rows

    #logger.info(
        #"%s had %i successes over %i instances",
        #maker.name,
        #successes,
        #len(test_data),
        #)

    #return [(run["category"], maker.name, instances, successes)]

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

                    #import numpy
                    #for instances in map(int, map(round, numpy.r_[10.0:150.0:32j])):
                    #for instances in [60]:
                        #for _ in xrange(16):
                        #for _ in [0]:
                            #yield (simulate_run, [run, maker, train_data, test_data, instances])

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["category", "solver", "budget", "cost", "success", "split", "instances"])
        #writer.writerow(["category", "solver", "instances", "successes"])

        for (_, rows) in condor.do(yield_jobs(), workers, local):
            writer.writerows(rows)

            out_file.flush()

if __name__ == "__main__":
    borg.script(main)

