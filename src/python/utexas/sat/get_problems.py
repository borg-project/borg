"""
utexas/sat/get_problems.py

Get SAT problem set metadata.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from __future__ import with_statement

import os.path
import logging
import optparse
import numpy
import tables
import utexas.io

from contextlib import closing
from utexas.io import openz
from utexas.sat.dimacs import (
    DimacsGraphFile,
    DimacsParseError,
    )
from cargo.log import get_logger

log = get_logger(__name__)

class SAT_TasksTableDescription(tables.IsDescription):
    """
    Structure of an HDF5 SAT tasks table.
    """

    path = tables.StringCol(255, pos = 0)
    variables = tables.Int32Col(pos = 1)
    clauses = tables.Int32Col(pos = 2)

def get_task_row_key(row):
    """
    Get the unique key for a task table row.

    @sa: SAT_TasksTableDescription
    """

    return row["path"]

class GetProblems(object):
    """
    Collect a list of DIMACS CNF files in some location.
    """

    def __init__(self):
        """
        Initialize this experiment.
        """

        # parse command line arguments
        parser = optparse.OptionParser()

        parser.add_option(
            "-s",
            "--satisfiable",
            action = "store_true",
            help = "ignore unsat instances?")
        parser.add_option(
            "-u",
            "--unsatisfiable",
            action = "store_true",
            help = "ignore sat instances?")
        parser.add_option(
            "-w",
            "--weighted",
            action = "store_true",
            help = "find weighted CNF files")
        parser.add_option(
            "-d",
            "--directory",
            help = "read problems under directory PATH",
            metavar = "PATH")
        parser.add_option(
            "-o",
            "--out",
            dest = "out",
            help = "write metadata to h5 file PATH",
            metavar = "PATH")

        (self.__options, arguments) = parser.parse_args()

        assert self.__options.directory
        assert self.__options.out
        assert not self.__options.satisfiable or not self.__options.unsatisfiable

        self.__h5 = None

    def __collect(self):
        """
        Collect distributions for a solver.
        """

        # set up storage
        self.__h5 = tables.openFile(self.__options.out, "w")
        problems = self.__h5.createTable(self.__h5.root, "problems", SAT_TasksTableDescription)

        # collect and write metadata
        if self.__options.weighted:
            paths = list(utexas.io.files_under(self.__options.directory, "*.wcnf"))
            paths += list(utexas.io.files_under(self.__options.directory, "*.wcnf.gz"))
            paths += list(utexas.io.files_under(self.__options.directory, "*.wcnf.bz2"))
        else:
            paths = list(utexas.io.files_under(self.__options.directory, "*.cnf"))
            paths += list(utexas.io.files_under(self.__options.directory, "*.cnf.gz"))
            paths += list(utexas.io.files_under(self.__options.directory, "*.cnf.bz2"))

        for path in paths:
            # ignore as necessary
            if self.__options.satisfiable and "UNSAT" in path:
                continue
            elif self.__options.unsatisfiable and "UNSAT" not in path:
                continue

            # parse the file
            with closing(openz(path)) as f:
                try:
                    dimacs = DimacsGraphFile(f, weighted = self.__options.weighted)
                except DimacsParseError:
                    log.error("failed to parse %s" % path)

                    raise

                n = dimacs.N

                # FIXME a bloody hack
                try:
                    m = dimacs.total_weight
                except AttributeError:
                    m = dimacs.M

            # execute the solver
            log.info("%s: %i variables and %i clauses" % (os.path.basename(path), n, m))

            # store its behavior
            problems.append([(os.path.abspath(path), n, m)])

    def run(self):
        """
        Run.
        """

        log.info("metadata collection startup")

        self.__collect()

        self.close()

    def close(self):
        """
        Clean up.
        """

        if self.__h5:
            self.__h5.close()

        del self.__h5

# command line invocation
if __name__ == "__main__":
    e = GetProblems()

    try:
        e.run()
    except KeyboardInterrupt:
        log.warning("received keyboard interrupt; cleaning up")

        e.close()

        del e

