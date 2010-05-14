"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def clean_up_environment():
    """
    Set up the process environment for a forked script.
    """

    from os import unsetenv

    unsetenv("CARGO_FLAGS_EXTRA_FILE")

