#!/usr/bin/env python2.6
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import sys

def addenv(variable, value, separator = ":"):
    """
    Set the environment variable, prepending if necessary.
    """

    from os import (
        getenv,
        putenv,
        )

    old = getenv(variable)

    if old is None:
        putenv(variable, value)
    else:
        putenv(variable, "%s%s%s" % (value, separator, old))

def exec_main(as_module):
    """
    Launch the solver in a custom environment.
    """

    from os      import (
        getenv,
        putenv,
        execvp,
        )
    from os.path import (
        join,
        abspath,
        normpath,
        basename,
        )

    # provide the borg root path
    borg_root = abspath(sys.path[0])

    putenv("BORG_ROOT", borg_root)

    # build relevant custom environment variables
    putnv("PYTHONHOME", join(borg_root, "ext/prefix"))

    python_paths   = (
        join(borg_root, "src/dep/cargo/src/python"),
        join(borg_root, "src/dep/borg/src/python"),
        join(borg_root, "ext/prefix/lib/python2.6"),
        )

    addenv("PATH", join(borg_root, "ext/prefix/bin"))
    addenv("PYTHONPATH", ":".join(map(normpath, python_paths)))
    addenv("CMAKE_PREFIX_PATH", join(borg_root, "ext/prefix"))
    addenv("LD_LIBRARY_PATH", join(borg_root, "ext/prefix/lib"))
    addenv("CARGO_FLAGS_EXTRA_FILE", join(borg_root, "ext/flags.json"))

    # including a default tmpdir
    tmpdir_name = "TMPDIR"

    if getenv(tmpdir_name) is None:
        putenv(tmpdir_name, join(borg_root, "ext/tmp"))

    # make the call
    if as_module:
        program   = join(borg_root, "ext/prefix/bin/python")
        arguments = [program, "-m"] + sys.argv[1:]
    else:
        program   = sys.argv[1]
        arguments = sys.argv[1:]

    execvp(program, arguments)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec_main()
    else:
        print "usage: ./solver <module>"
        print "(see README)"

