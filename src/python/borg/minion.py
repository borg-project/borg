"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def write_min_from_cnf(out_file, cnf):
    """Convert a CNF file to the Minion input format."""

    out_file.write("MINION 3\n")
    out_file.write("**VARIABLES**\n")
    out_file.write("BOOL b[{0}]\n".format(cnf.N))
    out_file.write("**CONSTRAINTS**\n")

    for clause in cnf.clauses:
        clause_string = ",".join("{0}b[{1}]".format("!" if l < 0 else "", abs(l) - 1) for l in clause)

        out_file.write("watchsumgeq([{0}],1)\n".format(clause_string))

    out_file.write("**EOF**\n")

