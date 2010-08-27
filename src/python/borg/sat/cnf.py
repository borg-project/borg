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

feature_names = [
    "nvars",
    "nclauses",
    "vars-clauses-ratio",
    "VCG-VAR-mean",
    "VCG-VAR-coeff-variation",
    "VCG-VAR-min",
    "VCG-VAR-max",
    "VCG-VAR-entropy",
    "VCG-CLAUSE-mean",
    "VCG-CLAUSE-coeff-variation",
    "VCG-CLAUSE-min",
    "VCG-CLAUSE-max",
    "VCG-CLAUSE-entropy",
    "POSNEG-RATIO-CLAUSE-mean",
    "POSNEG-RATIO-CLAUSE-coeff-variation",
    "POSNEG-RATIO-CLAUSE-min",
    "POSNEG-RATIO-CLAUSE-max",
    "POSNEG-RATIO-CLAUSE-entropy",
    "POSNEG-RATIO-VAR-mean",
    "POSNEG-RATIO-VAR-stdev",
    "POSNEG-RATIO-VAR-min",
    "POSNEG-RATIO-VAR-max",
    "POSNEG-RATIO-VAR-entropy",
    "UNARY",
    "BINARY+",
    "TRINARY+",
    "HORNY-VAR-mean",
    "HORNY-VAR-coeff-variation",
    "HORNY-VAR-min",
    "HORNY-VAR-max",
    "HORNY-VAR-entropy",
    "horn-clauses-fraction",
    "VG-mean",
    "VG-coeff-variation",
    "VG-min",
    "VG-max",
    "KLB-featuretime",
    "CG-mean",
    "CG-coeff-variation",
    "CG-min",
    "CG-max",
    "CG-entropy",
    "cluster-coeff-mean",
    "cluster-coeff-coeff-variation",
    "cluster-coeff-min",
    "cluster-coeff-max",
    "cluster-coeff-entropy",
    "CG-featuretime",
    ]

def compute_raw_features(path):
    """
    Compute relevant features of the specified DIMACS-format file.
    """

    # find the associated feature computation binary
    from borg import get_support_path

    features1s = get_support_path("features1s")

    # execute the helper
    from cargo.io import check_call_capturing

    log.detail("executing %s %s", features1s, path)

    (output, _)     = check_call_capturing([features1s, path])
    (names, values) = [l.split(",") for l in output.splitlines()]

    if names != feature_names:
        raise RuntimeError("unexpected or missing feature names from features1s")
    else:
        return dict((n, float(v)) for (n, v) in zip(names, values))

