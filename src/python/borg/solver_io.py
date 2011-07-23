"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import uuid
import time
import signal
import select
import random
import datetime
import itertools
import multiprocessing
import numpy
import cargo
import cargo.unix.sessions
import cargo.unix.accounting
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

def random_seed():
    """Return a random solver seed."""

    return numpy.random.randint(0, 2**31)

@cargo.composed(list)
def timed_read(fds, timeout = -1):
    """Read from multiple descriptors with an optional timeout."""

    # poll for I/O events
    polling = select.poll()

    for fd in fds:
        polling.register(fd, select.POLLIN)

    changed = dict(polling.poll(timeout * 1000))

    # and interpret them
    for fd in fds:
        revents = changed.get(fd, 0)

        if revents & select.POLLIN:
            yield os.read(fd, 65536)
        elif revents & select.POLLHUP:
            yield ""
        else:
            yield None

class SolverProcess(multiprocessing.Process):
    """Attempt to solve the task in a subprocess."""

    def __init__(self, parse_output, arguments, stm_queue, mts_queue, solver_id):
        self._parse_output = parse_output
        self._arguments = arguments
        self._stm_queue = stm_queue
        self._mts_queue = mts_queue
        self._solver_id = solver_id
        self._seed = random_seed()
        self._popened = None

        logger.info("running %s", arguments)

        multiprocessing.Process.__init__(self)

    def run(self):
        numpy.random.seed(self._seed)
        random.seed(numpy.random.randint(2**31))

        try:
            class DeathRequestedError(Exception):
                pass

            try:
                def handle_sigusr1(number, frame):
                    raise DeathRequestedError()

                try:
                    signal.signal(signal.SIGUSR1, handle_sigusr1)

                    self.handle_subsolver()
                finally:
                    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
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
        # spawn solver
        limit = 0.0
        stdout = ""
        expenditure = datetime.timedelta(seconds = limit)
        last_expenditure = expenditure
        last_audit = time.time() - 1.0

        while limit == 0.0 or self._popened is not None:
            if expenditure >= datetime.timedelta(seconds = limit):
                if self._popened is not None:
                    os.kill(popened.pid, signal.SIGSTOP)

                    run_cost = cargo.seconds(expenditure - last_expenditure)
                    self._stm_queue.put((self._solver_id, run_cost, None, False))

                additional = self._mts_queue.get()
                limit += additional
                last_expenditure = expenditure

                if self._popened is None:
                    popened = cargo.unix.sessions.spawn_pipe_session(self._arguments, {})
                    self._popened = popened

                    descriptors = [popened.stdout.fileno(), popened.stderr.fileno()]
                    accountant = cargo.unix.accounting.SessionTimeAccountant(popened.pid)
                else:
                    os.kill(popened.pid, signal.SIGCONT)

            # spend some time waiting for output
            (chunk, _) = timed_read(descriptors, 1.0)

            if time.time() - last_audit > borg.defaults.proc_poll_period:
                accountant.audit()

                expenditure = accountant.total
                last_audit = time.time()

            # check for termination
            if chunk == "":
                self._popened = None
            elif chunk is not None:
                stdout += chunk

        # provide the outcome to the central planner
        answer = self._parse_output(stdout)
        run_cost = cargo.seconds(expenditure - last_expenditure)

        self._stm_queue.put((self._solver_id, run_cost, answer, True))

def prepare(command, root, cnf_path):
    """Format command for execution."""

    keywords = {
        "root": root,
        "task": cnf_path,
        "seed": random_seed(),
        }

    return [s.format(**keywords) for s in command]

class RunningSolver(object):
    """In-progress solver process."""

    def __init__(self, parse, command, root, task_path, stm_queue = None, solver_id = None):
        """Initialize."""

        if stm_queue is None:
            self._stm_queue = multiprocessing.Queue()
        else:
            self._stm_queue = stm_queue

        if solver_id is None:
            self._solver_id = uuid.uuid4()
        else:
            self._solver_id = solver_id

        self._mts_queue = multiprocessing.Queue()

        self._process = \
            SolverProcess(
                parse,
                prepare(command, root, task_path),
                self._stm_queue,
                self._mts_queue,
                self._solver_id,
                )

    def __call__(self, budget):
        """Unpause the solver, block for some limit, and terminate it."""

        self.unpause_for(budget)

        (solver_id, run_cpu_cost, answer, terminated) = self._stm_queue.get()

        assert solver_id == self._solver_id

        self.stop()

        borg.get_accountant().charge_cpu(run_cpu_cost)

        return answer

    def unpause_for(self, budget):
        """Unpause the solver for the specified duration."""

        if not self._process.is_alive():
            self._process.start()

        self._mts_queue.put(budget)

    def stop(self):
        """Terminate the solver."""

        if self._process.is_alive():
            os.kill(self._process.pid, signal.SIGUSR1)

            self._process.join()

class RunningPortfolio(object):
    """Portfolio running on a task."""

    def __init__(self, portfolio, suite, task):
        """Initialize."""

        self.portfolio = portfolio
        self.suite = suite
        self.task = task

    def __call__(self, budget):
        """Attempt to solve the associated task."""

        return self.portfolio(self.task, self.suite, borg.Cost(cpu_seconds = budget))

class RunningPortfolioFactory(object):
    """Run a portfolio on tasks."""

    def __init__(self, portfolio, suite):
        """Initialize."""

        self.portfolio = portfolio
        self.suite = suite

    def __call__(self, task):
        """Return an instance of this portfolio running on the task."""

        return RunningPortfolio(self.portfolio, self.suite, task)

