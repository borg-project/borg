"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import pty
import subprocess
import borg

log = borg.get_logger(__name__)

def _child_preexec(environment):
    """Run in the child code prior to execution."""

    # update the environment
    for (key, value) in environment.iteritems():
        os.putenv(key, str(value))

    # start our own session
    os.setsid()

def spawn_pipe_session(arguments, environment = {}, cwd = None):
    """Spawn a subprocess in its own session."""

    popened = \
        subprocess.Popen(
            arguments,
            close_fds = True,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            preexec_fn = lambda: _child_preexec(environment),
            cwd = cwd,
            )

    popened.stdin.close()

    return popened

def spawn_pty_session(arguments, environment = {}, cwd = None):
    """Spawn a subprocess in its own session, with stdout routed through a pty."""

    # build a pty
    (master_fd, slave_fd) = pty.openpty()

    log.debug("opened pty %s", os.ttyname(slave_fd))

    # launch the subprocess
    try:
        popened        = \
            subprocess.Popen(
                arguments,
                close_fds = True,
                stdin = slave_fd,
                stdout = slave_fd,
                stderr = subprocess.PIPE,
                preexec_fn = lambda: _child_preexec(environment),
                cwd = cwd,
                )
        popened.stdout = os.fdopen(master_fd)

        os.close(slave_fd)

        return popened
    except:
        raised = borg.util.Raised()

        try:
            if master_fd is not None:
                os.close(master_fd)
            if slave_fd is not None:
                os.close(slave_fd)
        except:
            borg.util.Raised().print_ignored()

        raised.re_raise()

def kill_session(sid, number):
    """
    Send signal C{number} to all processes in session C{sid}.

    Theoretically imperfect, but should be consistently effective---almost
    certainly paranoid overkill---in practice.
    """

    # why do we pkill multiple times? because we're crazy.
    for i in xrange(2):
        exit_code = subprocess.call(["pkill", "-%i" % number, "-s", "%i" % sid])

        if exit_code not in (0, 1):
            raise RuntimeError("pkill failure")

