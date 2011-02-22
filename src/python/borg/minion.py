"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def write_minion_from_cnf(out_file, cnf):
    """Convert a CNF file to the Minion input format."""

    out_file.write("**VARIABLES**\n")
    out_file.write("BOOL b[{0}]\n".format(cnf.N))
    out_file.write("**CONSTRAINTS**\n")

    for clause in cnf.clauses:
        out_file.write("watchsumgeq(b, 1)\n")

    out_file.write("**EOF**\n")

