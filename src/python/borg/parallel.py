"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import time
import random
import signal
import select
import datetime
import multiprocessing
import numpy
import cargo
import cargo.unix.sessions
import cargo.unix.accounting
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

class ParallelPortfolio(object):
    def __init__(self, solvers, train_paths):
        pass

    # XXX budget is wall-clock time, in this case?
    # XXX could just special-case the one-core timing / budget to be CPU time (for now)
    def __call__(self, cnf_path, budget):
        """Attempt to solve the instance."""

        cores = 4 # XXX
        queue = multiprocessing.Queue()
        processes = []
        taken = []
        cost = None

        while True:
            if len(processes) < cores:
                # XXX replan using taken as failures
                action = "XXX"
                seed = numpy.random.randint(2**31)
                process = SolverProcess(queue, action, seed)

                # XXX uniquely seed subprocesses
                process.start()
                taken.append(action)
                processes.append(process)
            else:
                # receive a message
                (run_cost, answer) = queue.get()

                print "got", run_cost, answer, "from subprocess"

                # XXX increment cost if measuring CPU time
                if answer is not None:
                    break

        # terminate in-flight actions
        # XXX in finally block
        for process in processes:
            print "killing process", process.pid
            os.kill(process.pid, signal.SIGUSR1)

            process.join()

        # ...
        return (cost, answer)

#borg.portfolios.named["bimixture-parallel"] = ParallelPortfolio

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    processes = [SolverProcess(queue, None, 42)]

    # XXX uniquely seed subprocesses
    for process in processes:
        process.start()

    time.sleep(16)

    queue.put(20)

    time.sleep(16)

    #for process in processes:
        #process.join()

    for process in processes:
        print "killing process", process.pid
        os.kill(process.pid, signal.SIGUSR1)
        process.join()

