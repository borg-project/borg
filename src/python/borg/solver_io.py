"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import uuid
import time
import shutil
import signal
import select
import random
import tempfile
import datetime
import multiprocessing
import numpy
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

def random_seed():
    """Return a random solver seed."""

    return numpy.random.randint(0, 2**31)

def timed_read(fds, timeout = -1):
    """Read from multiple descriptors with an optional timeout."""

    # poll for I/O events
    polling = select.poll()

    for fd in fds:
        polling.register(fd, select.POLLIN)

    changed = dict(polling.poll(timeout * 1000))

    # and interpret them
    def make_read(fd):
        revents = changed.get(fd, 0)

        if revents & select.POLLIN:
            return os.read(fd, 65536)
        elif revents & select.POLLHUP:
            return ""
        else:
            return None

    return map(make_read, fds)

class SolverProcess(multiprocessing.Process):
    """Attempt to solve the task in a subprocess."""

    def __init__(self, parse_output, arguments, stm_queue, mts_queue, solver_id, tmpdir, cwd):
        self._parse_output = parse_output
        self._arguments = arguments
        self._stm_queue = stm_queue
        self._mts_queue = mts_queue
        self._solver_id = solver_id
        self._tmpdir = tmpdir
        self._seed = random_seed()
        self._popened = None
        self._cwd = cwd

        if self._cwd is None:
            logger.info("running %s", arguments)
        else:
            logger.info("running %s under %s", arguments, cwd)

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
            except Exception, error:
                self._stm_queue.put(error)
        except KeyboardInterrupt:
            pass
        finally:
            if self._popened is not None:
                self._popened.kill()

                os.kill(self._popened.pid, signal.SIGCONT)

                self._popened.wait()

            shutil.rmtree(self._tmpdir, ignore_errors = True)

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

                    run_cost = borg.util.seconds(expenditure - last_expenditure)
                    self._stm_queue.put((self._solver_id, run_cost, None, False))

                additional = self._mts_queue.get()
                limit += additional
                last_expenditure = expenditure

                if self._popened is None:
                    popened = borg.unix.sessions.spawn_pipe_session(self._arguments, cwd = self._cwd)
                    self._popened = popened

                    descriptors = [popened.stdout.fileno(), popened.stderr.fileno()]
                    accountant = borg.unix.accounting.SessionTimeAccountant(popened.pid)
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
        run_cost = borg.util.seconds(expenditure - last_expenditure)

        self._stm_queue.put((self._solver_id, run_cost, answer, True))

def prepare(command, root, cnf_path, tmpdir):
    """Format command for execution."""

    keywords = {
        "root": root,
        "task": cnf_path,
        "seed": random_seed(),
        "tmpdir": tmpdir,
        }

    return [s.format(**keywords) for s in command]

class RunningSolver(object):
    """In-progress solver process."""

    def __init__(
        self,
        parse,
        command,
        root,
        task_path,
        stm_queue = None,
        solver_id = None,
        cwd = None,
        ):
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
        self._tmpdir = tempfile.mkdtemp(prefix = "borg.")

        self._process = \
            SolverProcess(
                parse,
                prepare(command, root, task_path, self._tmpdir),
                self._stm_queue,
                self._mts_queue,
                self._solver_id,
                self._tmpdir,
                cwd,
                )

    def __call__(self, budget):
        """Unpause the solver, block for some limit, and terminate it."""

        self.unpause_for(budget)

        response = self._stm_queue.get()

        if isinstance(response, Exception):
            raise response
        else:
            (solver_id, run_cpu_cost, answer, terminated) = response

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

        shutil.rmtree(self._tmpdir, ignore_errors = True)

class RunningPortfolio(object):
    """Portfolio running on a task."""

    def __init__(self, portfolio, suite, task):
        """Initialize."""

        self.portfolio = portfolio
        self.suite = suite
        self.task = task

    def run_then_stop(self, budget):
        """Attempt to solve the associated task."""

        return self.portfolio(self.task, self.suite, borg.Cost(cpu_seconds = budget))

class RunningPortfolioFactory(object):
    """Run a portfolio on tasks."""

    def __init__(self, portfolio, suite):
        """Initialize."""

        self.portfolio = portfolio
        self.suite = suite

    def start(self, task):
        """Return an instance of this portfolio running on the task."""

        return RunningPortfolio(self.portfolio, self.suite, task)

class EmptySolver(object):
    """Immediately return the specified answer."""

    def __init__(self, answer):
        self._answer = answer

    def __call__(self, budget):
        return self._answer

    def unpause_for(self, budget):
        pass

    def stop(self):
        pass

