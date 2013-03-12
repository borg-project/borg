"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import bz2
import sys
import pwd
import gzip
import shutil
import tempfile
import json
import traceback
import contextlib
import subprocess
import numpy

def files_under(path, extensions = None):
    """Iterate over paths in the specified directory tree."""

    assert not isinstance(extensions, str)

    if os.path.isfile(path):
        walked = [path]
    else:
        def walk_path():
            for (p, _, f) in os.walk(path, followlinks = True):
                for n in f:
                    yield os.path.join(p, n)

        walked = walk_path()

    if extensions is None:
        for name in walked:
            yield name
    else:
        for name in walked:
            if any(name.endswith(e) for e in extensions):
                yield name

@contextlib.contextmanager
def mkdtemp_scoped(prefix = None):
    """Create, and then delete, a temporary directory."""

    # provide a reasonable default prefix
    if prefix is None:
        prefix = "%s." % pwd.getpwuid(os.getuid())[0]

    # create the context
    path = None

    try:
        path = tempfile.mkdtemp(prefix = prefix)

        yield path
    finally:
        if path is not None:
            shutil.rmtree(path, ignore_errors = True)

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

@contextlib.contextmanager
def numpy_printing(**kwargs):
    """Temporarily modify numpy printing options."""

    old = numpy.get_printoptions()

    numpy.set_printoptions(**kwargs)

    try:
        yield
    except:
        raise
    finally:
        numpy.set_printoptions(**old)

@contextlib.contextmanager
def numpy_errors(**kwargs):
    """Temporarily modify numpy error options."""

    old = numpy.seterr(**kwargs)

    try:
        yield
    except:
        raise
    finally:
        numpy.seterr(**old)

def seconds(value):
    """Return the equivalent number of seconds, floating-point."""

    return value.days * 8.64e4 + value.seconds + value.microseconds / 1e6

def call_capturing(arguments, input = None, preexec_fn = None):
    """Spawn a process and return its output and status code."""

    popened = None

    try:
        # launch the subprocess
        popened = \
            subprocess.Popen(
                arguments,
                stdin      = subprocess.PIPE,
                stdout     = subprocess.PIPE,
                stderr     = subprocess.PIPE,
                preexec_fn = preexec_fn,
                )

        # wait for its natural death
        (stdout, stderr) = popened.communicate(input)
    except:
        #raised = Raised()

        if popened is not None and popened.poll() is None:
            #try:
            popened.kill()
            popened.wait()
            #except:
                #Raised().print_ignored()

        #raised.re_raise()
    else:
        return (stdout, stderr, popened.returncode)

def check_call_capturing(arguments, input = None, preexec_fn = None):
    """Spawn a process and return its output."""

    (stdout, stderr, code) = call_capturing(arguments, input, preexec_fn)

    if code == 0:
        return (stdout, stderr)
    else:
        from subprocess import CalledProcessError

        error = CalledProcessError(code, arguments)

        error.stdout = stdout
        error.stderr = stderr

        raise error

class Raised(object):
    """
    Store the currently-handled exception.

    The current exception must be saved before errors during error handling are
    handled, so that the original exception can be re-raised with its context
    information intact.
    """

    def __init__(self):
        (self.type, self.value, self.traceback) = traceback.exc_info()

    def format(self):
        """Return a list of lines describing the exception."""

        return traceback.format_exception(self.type, self.value, self.traceback)

    def re_raise(self):
        """Re-raise the stored exception."""

        raise (self.type, self.value, self.traceback)

    def print_ignored(self, message = "An error was unavoidably ignored:", file_ = sys.stderr):
        """Print an exception-was-ignored message."""

        file_.write("\n%s\n" % message)
        file_.write("".join(self.format()))
        file_.write("\n")

