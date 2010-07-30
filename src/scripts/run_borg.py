#!/usr/bin/env python
"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def patch_environment(path):
    """
    Load an environment modification and apply it.
    """

    # load the configuration
    from json import load

    with open(path) as f:
        loaded = load(f)

    # apply it
    from os      import environ
    from os.path import (
        abspath,
        dirname,
        expandvars,
        )

    if environ.has_key("HERE"):
        raise RuntimeError("$HERE already exists")
    else:
        environ["HERE"] = abspath(dirname(path))

    for (name, command) in loaded.items():
        if isinstance(command, str):
            action    = "set"
            new_value = command
        elif isinstance(command, dict):
            action    = command.get("action", "set")
            new_value = command["value"]
        else:
            raise ValueError("unexpected command type in configuration")

        old_value = environ.get(name)

        if action == "replace":
            update = new_value
        if action == "set":
            if old_value is not None:
                raise RuntimeError("variable to set is already set")
            else:
                update = new_value
        else:
            if old_value is None:
                old_values = []
            else:
                old_values = [old_value]

            if action == "append":
                unjoined = old_values + new_value
            elif action == "prepend":
                unjoined = new_value + old_values
            else:
                raise ValueError("unrecognized action")

            update = ":".join(unjoined)

        environ[name] = expandvars(update)

    del environ["HERE"]

def run_borg():
    """
    Run the module specified on the command line.
    """

    # modify our environment
    from os.path import join
    from sys     import path

    patch_environment(join(path[0], "environment.json"))

    # then replace this process image
    from os  import execvp
    from sys import argv

    execvp("python", ["python", "-m"] + argv[1:])

if __name__ == "__main__":
    run_borg()

