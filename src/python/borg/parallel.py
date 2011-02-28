"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import random
import signal
import multiprocessing
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def handle_sigusr1(number, frame):
    """Kill the current process in response to a signal."""

    raise RuntimeError() # XXX use a better exception

class SolverProcess(multiprocessing.Process):
    """Attempt to solve the task in a subprocess."""

    def __init__(self, queue, action, seed):
        self._queue = queue

        numpy.random.seed(seed)
        random.seed(numpy.random.randint(2**31))

        multiprocessing.Process.__init__(self)

    def run(self):
        # XXX silence all logging
        try:
            print "in process", os.getpid()

            signal.signal(signal.SIGUSR1, handle_sigusr1)

            import time
            import numpy
            s = numpy.random.randint(16) + 4
            print "sleeping for", s
            time.sleep(s)

            self._queue.put((os.getpid(), [42]))
        except KeyboardInterrupt:
            print os.getpid(), "got keyboard interrupt"
            # XXX

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

