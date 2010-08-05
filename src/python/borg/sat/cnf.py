"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import re

from cargo.log import get_logger

log = get_logger(__name__)

__condense_spaces_re = re.compile(" +")

def write_sanitized_cnf(source, destination):
    """
    Filter a CNF file to make picky solvers happy.
    """

    for line in yield_sanitized_cnf(source):
        destination.write(line)

def yield_sanitized_cnf(source):
    """
    Filter a CNF file to make picky solvers happy.
    """

    for line in source:
        stripped = line.strip()

        if stripped == "%":
            break
        elif stripped.startswith("c "):
            continue
        else:
            yield __condense_spaces_re.sub(" ", stripped)
            yield "\n"

class DIMACS_ParseError(RuntimeError):
    """
    File was not in the expected format.
    """

    pass

class DIMACS_GraphFile(object):
    """
    Read metadata from the DIMACS satisfiability problem suggested format.
    """

    def __init__(self, file, weighted = False):
        """
        Initialize this problem.
        """

        # members
        self.__N = None
        self.__M = None
        self.weighted = weighted

        # parse
        self.__parse_file(file)

    def __parse_file(self, file):
        """
        Parse the specified file.
        """

        try:
            for (i, line) in enumerate(file):
                self.__parse_line(line)
        except DimacsParseError:
            log.error("failed to parse DIMACS file at line #%i:\n%s" % ((i + 1), line.rstrip()))

            raise

    def __parse_line(self, line):
        """
        Parse the specified line in the file.
        """

        def parser_check(boolean):
            if not boolean:
                raise DimacsParseError()

        s = line.split()

        if not s:
            return
        elif s[0] == "p":
            parser_check(len(s) == 4)

            if self.weighted:
                parser_check(s[1] == "wcnf")
            else:
                parser_check(s[1] == "cnf")

            parser_check(self.__N is None)
            parser_check(self.__M is None)

            self.__N = int(s[2])
            self.__M = int(s[3])

            if self.weighted:
                self.total_weight = 0
        elif self.__M is not None:
            if self.weighted:
                self.total_weight += int(s[0])

            parser_check(s[-1] == "0")

    # properties
    N = property(lambda self: self.__N)
    M = property(lambda self: self.__M)

def compute_features(path):
    """
    Compute relevant features of the specified DIMACS-format file.
    """

    # find the associated feature computation binary
    from os.path import (
        join,
        dirname,
        )

    features1s = join(dirname(__file__), "features1s")

    # execute the helper
    from cargo.io import check_call_capturing

    (output, _)     = check_call_capturing([features1s, path])
    (names, values) = [l.split(",") for l in output.splitlines()]

    return dict((n, float(v)) for (n, v) in zip(names, values))

