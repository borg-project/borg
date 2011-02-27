"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

cimport numpy
cimport libc.stdlib

class DIMACS_ParseError(RuntimeError):
    """
    File was not in the expected format.
    """

    def __init__(self, line = None):
        """
        Initialize.
        """

        if line is None:
            RuntimeError.__init__(self)
        else:
            RuntimeError.__init__(self, "parse error at: %s" % line)

class DIMACS_GraphFile(object):
    """
    Read the DIMACS satisfiability problem suggested format.
    """

    def __init__(self, comments, clauses, N):
        """
        Initialize.
        """

        self.comments = comments
        self.clauses = clauses
        self.N = N

    @staticmethod
    def parse_header(file_):
        """Parse the header of the specified CNF file."""

        comments = []

        for line in file_:
            if line.startswith("c"):
                comments.append(line[1:])
            elif line.startswith("p"):
                fields = line.split()

                if len(fields) != 4  or fields[1] != "cnf":
                    raise DIMACS_ParseError(line)
                else:
                    (N, M) = map(int, fields[2:])

                    break
            else:
                raise DIMACS_ParseError(line)

        return (comments, N, M)

    @staticmethod
    def parse(file_):
        """Parse the specified CNF file."""

        (comments, N, M) = DIMACS_GraphFile.parse_header(file_)

        clauses = []
        clause = []

        for line in file_:
            for field in line.split():
                literal = int(field)

                if literal == 0:
                    clauses.append(clause)

                    clause = []
                elif abs(literal) <= N:
                    clause.append(literal)
                else:
                    raise DIMACS_ParseError(line)

        if len(clauses) != M:
            raise DIMACS_ParseError()

        return DIMACS_GraphFile(comments, clauses, N)

parse_cnf = DIMACS_GraphFile.parse
parse_cnf_header = DIMACS_GraphFile.parse_header

