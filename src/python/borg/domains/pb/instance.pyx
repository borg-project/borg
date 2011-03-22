"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import cargo

logger = cargo.get_logger(__name__)

class PseudoBooleanInstance(object):
    def __init__(self, N, constraints, objective = None, nonlinear = False):
        self.N = N
        self.constraints = constraints
        self.objective = objective
        self.nonlinear = nonlinear

def parse_opb_file_linear(opb_file):
    """Parse a pseudo-Boolean satisfiability instance."""

    # prepare constraint parsing
    def parse_terms(parts):
        terms = []

        assert len(parts) % 2 == 0

        for i in xrange(0, len(parts), 2):
            weight = int(parts[i][1:])
            literal = int(parts[i + 1][1:])

            terms.append((weight, literal))

        return terms

    # parse all lines
    constraints = []
    objective = None

    for (i, line) in enumerate(opb_file):
        if i == 0:
            (M, N, nonlinear) = parse_opb_file_header(line)
        elif not line.startswith("*"):
            if line.startswith("min:"):
                # objective line
                if objective is not None:
                    raise RuntimeError("multiple objectives in PB input")

                objective = parse_terms(line[4:-2].split())
            else:
                # constraint line
                parts = line[:-2].split()
                terms = parse_terms(parts[:-2])
                relation = parts[-2]
                value = int(parts[-1])

                constraints.append((terms, relation, value))

    # ...
    return PseudoBooleanInstance(N, constraints, objective, nonlinear)

def parse_opb_file_header(line):
    (N,) = map(int, re.findall("#variable= *([0-9]+)", line))
    (M,) = map(int, re.findall("#constraint= *([0-9]+)", line))
    nonlinear = re.search("#product=", line) is not None

    return (M, N, nonlinear)

#def parse_terms(parts):
    #weight = None
    #literals = []
    #relation = None
    #terms = []

    #for part in parts:
        #if part[0] == "~":
            #literals.append(-int(part[2:]))
        #elif part[0] == "x":
            #literals.append(int(part[1:]))
        #else:
            #if weight is not None:
                #terms.append((weight, literals))

                #literals = []

            #weight = int(part)

    #terms.append((weight, literals))

    #return terms

#def parse_comment(char* position):
    #pass

#def parse_objective(char* position):
    #pass

#def parse_constraint(char* constraint):
    #pass

#def parse_opb_file_rd(opb_file):
    #pass

