"""
cargo/ai/max_sat/solvers.py

Run max-satisfiability solvers.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re

from tempfile import NamedTemporaryFile
from itertools import count
from contextlib import closing
from cargo.io import (
    openz,
    )
from cargo.log import get_logger
from cargo.unix.proc import run_cpu_limited

log = get_logger(__name__)

class MAX_SAT_Solver(object):
    """
    A cutoff-standardized MAX-SAT solver.
    """

    __optimum_line_re = re.compile("o (?P<c>\\d+)")

    def __init__(self, command):
        """
        Initialize this solver.
        """

        self.command = command

    def solve(self, cutoff, cnf_path):
        """
        Execute the solver and return its outcome.
        """

        # attempt to autodetect the file type
        if cnf_path.endswith(".wcnf"):
            suffix = ".wcnf"
        else:
            suffix = ".cnf"

        # run the solver
        with closing(openz(cnf_path)) as source:
            with NamedTemporaryFile(suffix = suffix) as temporary:
                log.info("writing %s from %s" % (temporary.name, cnf_path))

                write_sanitized_cnf(temporary, source)

                temporary.flush()

                return self.__solve(cutoff, temporary.name)

    def __solve(self, cutoff, input_path):
        """
        Execute the solver and return its outcome, given a concrete input path.
        """

        # run the solver
        log.info("running %s on input" % self.command)

        chunks = \
            run_cpu_limited(
                [
                    self.command,
                    input_path,
                ],
                cutoff,
                )

        # grab the reported optima
        def yield_optima():
            for (cpu_time, chunk) in chunks:
                for line in chunk.split("\n"):
                    m = MAX_SAT_Solver.__optimum_line_re.match(line)

                    if m:
                        yield (cpu_time, int(m.group("c")))

        optima = list(yield_optima())

        if not optima:
            log.debug("oddly no optima found in output:\n%s\n", "\n".join(str(s) for (_, s) in chunks))

        return optima
 
