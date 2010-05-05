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

def get_default_root():
    """
    The computed root directory of this package.
    """

    from os.path import abspath

    return abspath(sys.path[0])

def execute_borg(
    root      = get_default_root(),
    support   = "",
    submodule = "",
    as_module = False,
    argv      = sys.argv,
    ):
    """
    Set up the solver environment, and run.
    """

    from os      import (
        getenv,
        putenv,
        execvp,
        )
    from os.path import (
        join,
        isdir,
        exists,
        normpath,
        )

    root_path      = normpath(root)
    support_path   = join(root, support)
    submodule_path = join(root, submodule)

    # provide the borg root path
    putenv("BORG_ROOT", root_path)

    # set up submodule paths for python
    python_paths = [
        join(root_path, "src/python"),
        join(submodule_path, "cargo/src/python"),
        join(submodule_path, "borg/src/python"),
        ]

    addenv("PYTHONPATH", ":".join(python_paths))

    # set up the local prefix if it exists
    prefix_path = join(support_path, "prefix")

    if isdir(prefix_path):
        # common
        addenv("CMAKE_PREFIX_PATH", prefix_path)
        addenv("PATH", join(prefix_path, "bin"))
        addenv("LD_LIBRARY_PATH", join(prefix_path, "lib"))

        # only if python is installed locally
        local_python_binary = join(prefix_path, "bin/python")

        if exists(local_python_binary):
            local_python = local_python_binary

            putenv("PYTHONHOME", prefix_path)
            addenv("PYTHONPATH", join(prefix_path, "lib/python2.6"))
        else:
            local_python = None
    else:
        local_python = None

    # set up standard configuration file(s)
    flags_path = join(support_path, "flags.json")

    if exists(flags_path):
        addenv("CARGO_FLAGS_EXTRA_FILE", flags_path)

    # set up a default (local) tmpdir, if one exists
    tmpdir_name = "TMPDIR"

    if getenv(tmpdir_name) is None:
        tmpdir_path = join(support_path, "tmp")

        if isdir(tmpdir_path):
            putenv(tmpdir_name, tmpdir_path)

    # execute the command
    if as_module:
        program   = local_python if local_python else "python"
        arguments = [program, "-m"] + sys.argv[1:]
    else:
        program   = sys.argv[1]
        arguments = sys.argv[1:]

    execvp(program, arguments)

