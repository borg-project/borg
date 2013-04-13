"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import borg

def test_max_sat_parser():
    task_path = os.path.join(os.path.dirname(__file__), "s2v120c1200-2.cnf")
    parser = borg.domains.max_sat.instance.DIMACS_Parser()

    with open(task_path) as task_file:
        print parser.parse(task_file.read())

