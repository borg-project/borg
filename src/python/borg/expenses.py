"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import time
import operator
import resource
import contextlib
import borg

class Accountant(object):
    """Track resources used."""

    def __init__(self, parent = None):
        """Start tracking."""

        self._parent = parent
        self._past = Cost()

        self.start()

    def start(self):
        """Start or restart tracking."""

        self._start_cpu_seconds = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self._start_wall_seconds = time.time()

    def stop(self):
        """Stop tracking."""

        self._past = self.total

        self._start_cpu_seconds = None
        self._start_wall_seconds = None

    def charge(self, cost):
        """Add an external cost."""

        self._past += cost

        if self._parent is not None:
            self._parent.charge(cost)

    @property
    def total(self):
        """The total accumulated cost."""

        cpu_now = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        recent = \
            Cost(
                cpu_seconds = cpu_now - self._start_cpu_seconds,
                wall_seconds = time.time() - self._start_wall_seconds,
                )

        return self._past + recent

accountant_stack = []

def get_accountant():
    if accountant_stack:
        return accountant_stack[-1]
    else:
        return None

@contextlib.contextmanager
def accounting():
    accountant = Accountant(get_accountant())

    accountant_stack.append(accountant)

    yield accountant

    accountant.stop()
    accountant_stack.pop()

def normal_to_machine(machine_cpu_seconds):
    return machine_cpu_seconds * borg.defaults.machine_speed

def machine_to_normal(normal_cpu_seconds):
    return normal_cpu_seconds / borg.defaults.machine_speed

def none_op(op, x, y):
    if x is None:
        return y
    else:
        if y is None:
            return x
        else:
            return op(x, y)

class Cost(object):
    """Resources."""

    def __init__(self, cpu_seconds = None, wall_seconds = None):
        self.cpu_seconds = cpu_seconds
        self.wall_seconds = wall_seconds

    def __add__(self, other):
        none_add = lambda x, y: none_op(operator.add, x, y)

        return \
            Cost(
                cpu_seconds = none_add(self.cpu_seconds, other.cpu_seconds),
                wall_seconds = none_add(self.wall_seconds, other.wall_seconds),
                )

    def __sub__(self, other):
        none_sub = lambda x, y: none_op(operator.sub, x, y)

        return \
            Cost(
                cpu_seconds = none_sub(self.cpu_seconds, other.cpu_seconds),
                wall_seconds = none_sub(self.wall_seconds, other.wall_seconds),
                )

