"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy

cimport numpy

class DIMACS_ParseError(RuntimeError):
    """File was not in the expected format."""

    def __init__(self, line = None):
        if line is None:
            RuntimeError.__init__(self)
        else:
            RuntimeError.__init__(self, "parse error at: %s" % line)

class DIMACS_GraphFile(object):
    """
    Read the DIMACS satisfiability problem suggested format.
    """

    def __init__(self, comments, clauses, N):
        self.comments = comments
        self.clauses = clauses
        self.N = N

    def write(self, out_file):
        """Write this CNF to a file, in DIMACS format."""

        out_file.write("p cnf {0} {1}\n".format(self.N, len(self.clauses)))

        for clause in self.clauses:
            out_file.write(" ".join(clause))
            out_file.write(" 0\n")

    def satisfied(self, certificate):
        """Verify that the certificate satisfies this expression."""

        assignment = numpy.ones(self.N, bool)

        for literal in certificate:
            variable = abs(literal) - 1

            if literal == 0:
                break
            elif variable >= assignment.size:
                return None

            assignment[variable] = literal > 0

        for clause in self.clauses:
            satisfied = False

            for literal in clause:
                if assignment[abs(literal)] == literal > 0:
                    satisfied = True

                    break

            if not satisfied:
                return None

        return assignment

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

