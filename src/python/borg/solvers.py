"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re
import os.path
import itertools
import multiprocessing
import numpy
import cargo
import borg

logger = cargo.get_logger(__name__)

class DeathRequestedError(Exception):
    pass

def handle_sigusr1(number, frame):
    """Kill the current process in response to a signal."""

    raise DeathRequestedError()

def random_seed():
    """Return a random solver seed."""

    return numpy.random.randint(0, 2**31)

def prepare(command, cnf_path):
    """Format command for execution."""

    keywords = {
        "root": borg.defaults.solvers_root.rstrip("/"),
        "seed": random_seed(),
        "task": cnf_path,
        }

    return [s.format(**keywords) for s in command]

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
        self._command = command
        self._stm_queue = stm_queue
        self._mts_queue = mts_queue
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
        # spawn solver
        arguments = prepare(self._command, self._cnf_path)
        self._popened = popened = cargo.unix.sessions.spawn_pipe_session(arguments, {})

        accountant = cargo.unix.accounting.SessionTimeAccountant(popened.pid)
        stdout = ""

        while True:
            chunk = timed_read(popened.stdout.fileno(), 1)

            if chunk is not None:
                stdout += chunk

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

                print "continued subprocess"

            print "used", accountant.total, "cpu seconds"

        # parse the solver's output
        match = re.search(r"^s +(.+)$", stdout, re.M)

        if match:
            (answer_type,) = map(str.upper, match.groups())

            if answer_type == "SATISFIABLE":
                answer = []

                for line in re.findall(r"^v ([ \-0-9]*)$", stdout, re.M):
                    answer.extend(map(int, line.split()))

                with open(cnf_path) as cnf_file:
                    (_, N, _) = borg.dimacs.parse_cnf_header(cnf_file)

                if len(answer) == N + 1 and answer[-1] == 0:
                    answer = answer[:-1]
                elif len(answer) != N:
                    #logger.warning("subsolver cert has %i variables; CNF has %i", len(answer), N)

                    answer = None
            elif answer_type == "UNSATISFIABLE":
                answer = False
            else:
                answer = None
        else:
            answer = None

        #if answer is None and budget - cost > 1.0:
            #logger.warning("early subsolver termination (%.2f seconds remaining)", budget - cost)

class MonitoredSolver(object):
    def __init__(self, command, cnf_path, stm_queue, solver_id):
        self._mts_queue = multiprocessing.Queue()
        self._process = SolverProcess(command, stm_queue, self._mts_queue, solver_id)

    def go(self, budget):
        self._process.start()
        self._mts_queue.put(budget)

    def die(self):
        os.kill(process.pid, signal.SIGUSR1)
        process.join()

def basic_command(relative):
    """Prepare a basic competition solver command."""

    return ["{{root}}/{0}".format(relative), "{task}", "{seed}"]

core_commands = {
    # complete
    #"precosat-570": ["{root}/precosat-570-239dbbe-100801/precosat", "--seed={seed}", "{task}"],
    #"lingeling-276": ["{root}/lingeling-276-6264d55-100731/lingeling", "--seed={seed}", "{task}"],
    #"cryptominisat-2.9.0": ["{root}/cryptominisat-2.9.0/cryptominisat-2.9.0Linux64", "--randomize={seed}", "{task}"],
    #"glucosER": ["{root}/glucosER/glucoser_static", "{task}"],
    #"glucose": ["{root}/glucose/glucose_static", "{task}"],
    #"SApperloT": ["{root}/SApperloT/SApperloT-base", "-seed={seed}", "{task}"],
    "march_hi": basic_command("march_hi/march_hi"),
    "kcnfs-2006": ["{root}/kcnfs-2006/kcnfs-2006", "{task}"],
    # incomplete
    "TNM": basic_command("TNM/TNM"),
    "gnovelty+2": basic_command("gnovelty+2/gnovelty+2"),
    "hybridGM3": basic_command("hybridGM3/hybridGM3"),
    "adaptg2wsat2009++": basic_command("adaptg2wsat2009++/adaptg2wsat2009++"),
    "iPAWS": basic_command("iPAWS/iPAWS"),
    "FH": basic_command("FH/FH"),
    "NCVWr": basic_command("NCVWr/NCVWr"),
    }

def basic_solver(name, command):
    """Return a basic competition solver callable."""

    return cargo.curry(MonitoredSolver, command)

named = dict(zip(core_commands, itertools.starmap(basic_solver, core_commands.items())))

