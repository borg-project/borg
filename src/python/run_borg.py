#!/usr/bin/env python
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def run_borg():
    """
    Run the module specified on the command line.
    """

    import sys

    from runpy import run_module

    name     = sys.argv[1]
    sys.argv = sys.argv[1:]

    run_module(name, run_name = "__main__")

if __name__ == "__main__":
    run_borg()

