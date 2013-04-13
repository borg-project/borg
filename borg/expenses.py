"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import time
import operator
import resource
import contextlib
import borg

class Cost(object):
    """Resources."""

    def __init__(self, cpu_seconds = None, wall_seconds = None):
        self.cpu_seconds = None if cpu_seconds is None else float(cpu_seconds)
        self.wall_seconds = None if wall_seconds is None else float(wall_seconds)

    def __str__(self):
        return "(CPU seconds: {0}; wall seconds: {1})".format(self.cpu_seconds, self.wall_seconds)

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

class Accountant(object):
    """Track resources used."""

    def __init__(self, parent = None, eve = False):
        """Start tracking."""

        self._parent = parent

        if eve:
            self._past = Cost(cpu_seconds = resource.getrusage(resource.RUSAGE_SELF).ru_utime, wall_seconds = 0.0)
        else:
            self._past = Cost(cpu_seconds = 0.0, wall_seconds = 0.0)

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

    def charge_cpu(self, cpu_seconds):
        """Add an external cost."""

        self.charge(Cost(cpu_seconds = float(cpu_seconds)))

    @property
    def total(self):
        """The total accumulated cost."""

        if self._start_cpu_seconds is None:
            recent = Cost()
        else:
            cpu_now = resource.getrusage(resource.RUSAGE_SELF).ru_utime
            recent = \
                Cost(
                    cpu_seconds = cpu_now - self._start_cpu_seconds,
                    wall_seconds = time.time() - self._start_wall_seconds,
                    )

        return self._past + recent

accountant_stack = [Accountant(eve = True)]

def get_accountant():
    if accountant_stack:
        return accountant_stack[-1]
    else:
        return None

@contextlib.contextmanager
def accounting():
    accountant = Accountant(get_accountant())

    accountant_stack.append(accountant)

    try:
        yield accountant
    except:
        raise
    finally:
        accountant.stop()
        accountant_stack.pop()

def normal_to_machine(machine_cpu_seconds):
    return machine_cpu_seconds * borg.defaults.machine_speed

def machine_to_normal(normal_cpu_seconds):
    return normal_cpu_seconds / borg.defaults.machine_speed

def none_op(op, x, y):
    if x is None:
        return None
    else:
        if y is None:
            return x
        else:
            return op(x, y)

def unicore_cpu_budget(budget):
    """The maximum single-core CPU budget."""

    if budget.cpu_seconds is None:
        if budget.wall_seconds is None:
            return 1e8
        else:
            return budget.wall_seconds
    elif budget.wall_seconds is None:
        return budget.cpu_seconds
    else:
        return min(budget.cpu_seconds, budget.wall_seconds)

