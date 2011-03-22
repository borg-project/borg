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

def parse_opb_file(opb_file):
    """Parse a pseudo-Boolean satisfiability instance."""

    # prepare constraint parsing
    def parse_terms(parts):
        weight = None
        literals = []
        relation = None
        terms = []

        for part in parts:
            if part[0] == "~":
                literals.append(-int(part[2:]))
            elif part[0] == "x":
                literals.append(int(part[1:]))
            else:
                if weight is not None:
                    terms.append((weight, literals))

                    literals = []

                weight = int(part)

        terms.append((weight, literals))

        return terms

    # parse all lines
    N = None
    constraints = []
    objective = None

    for line in opb_file:
        if line.startswith("*"):
            # comment line
            if N is None:
                (N,) = map(int, re.findall("#variable= *([0-9]+)", line))
                nonlinear = re.search("#product=", line) is not None
        elif line.startswith("min:"):
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

    assert N is not None

    # ...
    return PseudoBooleanInstance(N, constraints, objective, nonlinear)

