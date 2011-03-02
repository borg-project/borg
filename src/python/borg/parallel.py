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

class DeathRequestedError(Exception):
    pass

def handle_sigusr1(number, frame):
    """Kill the current process in response to a signal."""

    raise DeathRequestedError()

def timed_read(fd, timeout = -1):
    """Read with an optional timeout."""

    polling = select.poll()

    polling.register(fd, select.POLLIN)

    changed = polling.poll(timeout * 1000)

    if changed:
        ((fd, event),) = changed

        if event & select.POLLIN:
            return os.read(fd, 65536)
    else:
        return None

class SolverProcess(multiprocessing.Process):
    """Attempt to solve the task in a subprocess."""

    def __init__(self, command, stm_queue, mts_queue, limit, seed):
        self._mts_queue = mts_queue
        self._stm_queue = stm_queue
        self._solver = solver
        self._limit = limit
        self._seed = seed
        self._popened = None

        multiprocessing.Process.__init__(self)

    def run(self):
        numpy.random.seed(self._seed)
        random.seed(numpy.random.randint(2**31))

        try:
            signal.signal(signal.SIGUSR1, handle_sigusr1)

            self.handle_subsolver()
        except DeathRequestedError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            if self._popened is not None:
                self._popened.kill()

                os.kill(self._popened.pid, signal.SIGCONT)

                self._popened.wait()

    def handle_subsolver(self):
        self._popened = popened = cargo.unix.sessions.spawn_pipe_session(arguments, {})

        print "spawned child with pid", popened.pid

        accountant = cargo.unix.accounting.SessionTimeAccountant(popened.pid)
        all_output = ""

        while True:
            output = timed_read(popened.stdout.fileno(), 1)

            if output is not None:
                all_output += output

            accountant.audit()

            expenditure = accountant.total

            if popened.poll() is not None:
                # XXX parse output
                # XXX return answer
                break
            elif expenditure >= datetime.timedelta(seconds = self._limit):
                # pause the subsolver
                os.kill(popened.pid, signal.SIGSTOP)

                additional = self._queue.get()
                self._limit += additional

                os.kill(popened.pid, signal.SIGCONT)

                print "continued subproces"

            print "used", accountant.total, "cpu seconds"

class MonitoredSolver(object):
    def __init__(self, cnf_path, stm_queue, solver_id):
        command = [
            "/u/bsilvert/morgoth/sat-competition-2011/solvers/march_hi/march_hi",
            "/u/bsilvert/morgoth/sat-competition-2011/tasks/example/manol-pipe-unsat-g6bi.cnf",
            ]
        self._mts_queue = multiprocessing.Queue()
        self._process = SolverProcess(command, stm_queue, mts_queue, solver_id)

    def go(self, budget):
        self._process.start()
        mts_queue.put(budget)

    def die(self):
        os.kill(process.pid, signal.SIGUSR1)
        process.join()

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

