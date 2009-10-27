#!/usr/bin/env python2.6

import os
import sys
import signal

from os.path import (
    join,
    normpath,
    )
from subprocess import Popen

def main():
    """
    Exit a subprocess in an acridid environment.
    """

    # collect the relevant paths
    our_path       = sys.path[0]
    git_root       = join(our_path, "..")
    git_third_root = join(git_root, "dep/third")
    git_cargo_root = join(git_root, "dep/cargo")
    python_paths   = (
        join(git_third_root, "sqlalchemy/lib"),
        join(git_third_root, "psycopg2/build/built"),
        join(git_cargo_root, "kit/src/python"),
        join(git_cargo_root, "ai-sat/src/python"),
        join(git_cargo_root, "unix/src/python"),
        join(git_root, "src/python"),
        )

    # build the environment variable
    paths_prefix = ":".join(normpath(p) for p in python_paths)
    current_path = os.environ.get("PYTHONPATH", None)
    environment  = dict(os.environ)

    if current_path:
        environment["PYTHONPATH"] = "%s:%s" % (paths_prefix, current_path)
    else:
        environment["PYTHONPATH"] = paths_prefix

    environment["ACRIDID_ROOT"] = normpath(git_root)

    # make the call
    child = None

    try:
        child = Popen(sys.argv[1:], env = environment)

        child.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if child is not None:
            try:
                child.send_signal(signal.SIGINT)
                child.wait()
            except:
                pass

if __name__ == "__main__":
    main()

