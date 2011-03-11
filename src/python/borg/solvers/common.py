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

logger = cargo.get_logger(__name__)

def random_seed():
    """Return a random solver seed."""

    return numpy.random.randint(0, 2**31)

def timed_read(fds, timeout = -1):
    """Read with an optional timeout."""

    polling = select.poll()

    for fd in fds:
        polling.register(fd, select.POLLIN)

    changed = polling.poll(timeout * 1000)
    chunks = {}

    for (fd, event) in changed:
        if event & select.POLLIN:
            chunks[fd] = os.read(fd, 65536)

    return chunks

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

                signal.signal(signal.SIGUSR1, handle_sigusr1)

                try:
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

        while True:
            if expenditure >= datetime.timedelta(seconds = limit):
                if self._popened is not None:
                    os.kill(popened.pid, signal.SIGSTOP)

                    run_cost = cargo.seconds(expenditure - last_expenditure)
                    self._stm_queue.put((self._solver_id, run_cost, None, False))

                additional = self._mts_queue.get()
                limit += additional
                last_expenditure = expenditure

                if self._popened is None:
                    self._popened = popened = cargo.unix.sessions.spawn_pty_session(self._arguments, {})
                    popened_fds = [popened.stdout.fileno(), popened.stderr.fileno()]
                    accountant = cargo.unix.accounting.SessionTimeAccountant(popened.pid)
                else:
                    os.kill(popened.pid, signal.SIGCONT)

            # spend some time waiting for output
            chunks = timed_read(popened_fds, 1.0)
            chunk = chunks.get(popened.stdout.fileno())

            if chunk is not None:
                stdout += chunk

            if time.time() - last_audit > borg.defaults.proc_poll_period:
                accountant.audit()

                expenditure = accountant.total
                last_audit = time.time()

            # check for termination
            if chunk is None and popened.poll() is not None:
                self._popened = None

                break

        # provide the outcome to the central planner
        answer = self._parse_output(stdout)
        run_cost = cargo.seconds(expenditure - last_expenditure)

        self._stm_queue.put((self._solver_id, run_cost, answer, True))

def prepare(command, cnf_path):
    """Format command for execution."""

    keywords = {
        "root": borg.defaults.solvers_root.rstrip("/"),
        "seed": random_seed(),
        "task": cnf_path,
        }

    return [s.format(**keywords) for s in command]

class MonitoredSolver(object):
    """Provide a standard interface to a solver process."""

    def __init__(self, parse_output, command, task_path, stm_queue = None, solver_id = None):
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
                parse_output,
                prepare(command, task_path),
                self._stm_queue,
                self._mts_queue,
                self._solver_id,
                )

    def __call__(self, budget):
        """Unpause the solver, block for some limit, and terminate it."""

        self.go(budget)

        (solver_id, run_cost, answer, terminated) = self._stm_queue.get()

        assert solver_id == self._solver_id

        self.die()

        return (run_cost, answer)

    def go(self, budget):
        """Unpause the solver for the specified duration."""

        if not self._process.is_alive():
            self._process.start()

        self._mts_queue.put(budget)

    def die(self):
        """Terminate the solver."""

        if self._process.pid is not None:
            os.kill(self._process.pid, signal.SIGUSR1)

            self._process.join()

def basic_command(relative):
    """Prepare a basic competition solver command."""

    return ["{{root}}/{0}".format(relative), "{task}", "{seed}"]

