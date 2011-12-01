"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import bz2
import gzip
import json
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

def memoize(call):
    """Automatically memoize a callable."""

    # XXX use the fancy wrapper convenience functions in 2.7

    results = {}

    def wrapper(*args, **kwargs):
        key = (tuple(args), tuple(sorted(kwargs.iteritems())))

        try:
            return results[key]
        except KeyError:
            results[key] = result = call(*args, **kwargs)

            return result

    return wrapper

def load_json(path_or_file):
    """Load JSON from a path or file."""

    if isinstance(path_or_file, str):
        with openz(path_or_file, "rb") as json_file:
            return json.load(json_file)
    else:
        return json.load(path_or_file)

