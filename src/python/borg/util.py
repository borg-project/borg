"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import bz2
import gzip
import fnmatch
import contextlib

def files_under(path, pattern = "*"):
    """Iterate over the set of paths to files in the specified directory tree."""

    # walk the directory tree
    if os.path.isfile(path):
        walked = [path]
    else:
        def walk_path():
            for (p, _, f) in os.walk(path, followlinks = True):
                for n in f:
                    yield os.path.join(p, n)

        walked = walk_path()

    # filter names
    if isinstance(pattern, str):
        pattern = [pattern]

    for name in walked:
        if any(fnmatch.fnmatch(name, p) for p in pattern):
            yield name

def openz(path, mode = "rb", closing = True):
    """Open a file, transparently [de]compressing it if a known extension is present."""

    (_, extension) = os.path.splitext(path)

    if extension == ".bz2":
        file_ = bz2.BZ2File(path, mode)
    elif extension == ".gz":
        file_ = gzip.GzipFile(path, mode)
    elif extension == ".xz":
        raise NotImplementedError()
    else:
        return open(path, mode)

    if closing:
        return contextlib.closing(file_)

