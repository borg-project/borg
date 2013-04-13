"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import cStringIO as StringIO
import nose.tools
import borg

def path_to(name):
    return os.path.join(os.path.dirname(__file__), name)

def test_cnf_parse_simple():
    """Test simple CNF input."""

    with open(path_to("example.simple.cnf")) as cnf_file:
        instance = borg.domains.sat.instance.parse_sat_file(cnf_file)

    nose.tools.assert_equal(instance.N, 4)
    nose.tools.assert_equal(instance.M, 2)
    nose.tools.assert_equal(
        instance.to_clauses(),
        [[4, -1], [3, 2]],
        )

def test_cnf_write_simple():
    """Test simple CNF output."""

    clauses = [[-1, 4], [2, 3]]
    cnf_out = borg.domains.sat.instance.SAT_Instance.from_clauses(clauses, 4)
    file_out = StringIO.StringIO()

    cnf_out.write(file_out)

    file_in = StringIO.StringIO(file_out.getvalue())
    cnf_in = borg.domains.sat.instance.parse_sat_file(file_in)

    print cnf_out.to_clauses()
    print file_out.getvalue()

    nose.tools.assert_equal(cnf_in.to_clauses(), clauses)

