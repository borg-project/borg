"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import select
import signal
import datetime
import collections
import borg

log = borg.get_logger(__name__)

class SessionTimeAccountant(object):
    """
    Track the total CPU (user) time for members of a session.

    Process accounting under Linux is a giant pain, especially without root
    access. In the general case, it is literally impossible (without patching
    the kernel or some such craziness). Whatever. We do our best. Slightly
    fancier schemes are available, but they're not much fancier---they're
    mostly good only at making it harder for processes to actively evade being
    charged. For primarily long-running processes that act in good faith, we
    should do ok.
    """

    def __init__(self, sid):
        """
        Initialize.
        """

        self.sid     = sid
        self.charged = {}

    def audit(self):
        """
        Update estimates.
        """

        for p in borg.unix.proc.ProcessStat.in_session(self.sid):
            self.charged[p.pid] = p.user_time

    @property
    def total(self):
        """
        Return estimated total.
        """

        return sum(self.charged.values(), datetime.timedelta())

class PollingReader(object):
    """
    Read from file descriptors with timeout.
    """

    def __init__(self, fds):
        """
        Initialize.
        """

        self.fds     = fds
        self.polling = select.poll()

        for fd in fds:
            self.polling.register(fd, select.POLLIN)

    def unregister(self, fds):
        """
        Unregister descriptors.
        """

        for fd in fds:
            self.polling.unregister(fd)
            self.fds.remove(fd)

    def read(self, timeout = -1):
        """
        Read with an optional timeout.
        """

        changed = self.polling.poll(timeout * 1000)

        for (fd, event) in changed:
            log.debug("event on fd %i is %#o", fd, event)

            if event & select.POLLIN:
                # POLLHUP is level-triggered; we'll be back if it was missed
                return (fd, os.read(fd, 65536))
            elif event & select.POLLHUP:
                return (fd, "")
            else:
                raise IOError("unexpected poll response %#o from file descriptor" % event)

        return (None, None)

CPU_LimitedRun = \
    collections.namedtuple(
        "CPU_LimitedRun",
        [
            "started",
            "limit",
            "out_chunks",
            "err_chunks",
            "usage_elapsed",
            "proc_elapsed",
            "exit_status",
            "exit_signal",
            ],
        )

def run_cpu_limited(
    arguments,
    limit,
    pty         = True,
    environment = {},
    resolution  = 0.5,
    ):
    """
    Spawn a subprocess whose process tree is granted limited CPU (user) time.

    @param environment Override specific existing environment variables.

    The subprocess must not expect input. This method is best suited to
    processes which may run for a reasonable amount of time (eg, at least
    several seconds); it will be fairly inefficient (and ineffective) at
    fine-grained limiting of CPU allocation to short-duration processes.

    We run the process and read its output. Every time we receive a chunk of
    data, or every C{resolution} seconds, we estimate the total CPU time used
    by the session---and store that information with the chunk of output, if
    any. After at least C{limit} of CPU time has been used by the spawned
    session, or after the session leader terminates, whichever is first, the
    session is (sig)killed, the session leader waited on, and any data
    remaining in the pipe is read.

    Note that the use of SIGKILL means that child processes *cannot* perform
    their own cleanup.

    If C{pty} is specified, process stdout is piped through a pty, which makes
    process output less likely to be buffered. This behavior is the default.

    Kernel-reported resource usage includes the sum of all directly and
    indirectly waited-on children. It will be accurate in the common case where
    processes terminate after correctly waiting on their children, and
    inaccurate in cases where zombies are reparented to init. Elapsed CPU time
    taken from the /proc accounting mechanism is used to do CPU time limiting,
    and will always be at least the specified limit.
    """

    log.detail("running %s for %s", arguments, limit)

    # sanity
    if not arguments:
        raise ValueError()

    # start the run
    popened   = None
    fd_chunks = {}
    exit_pid  = None
    started   = datetime.datetime.utcnow()

    try:
        # start running the child process
        if pty:
            popened = borg.unix.sessions.spawn_pty_session(arguments, environment)
        else:
            popened = borg.unix.sessions.spawn_pipe_session(arguments, environment)

        fd_chunks = {
            popened.stdout.fileno(): [],
            popened.stderr.fileno(): [],
            }

        log.debug("spawned child with pid %i", popened.pid)

        # read the child's output while accounting (note that the session id
        # is, under Linux, the pid of the session leader)
        accountant = SessionTimeAccountant(popened.pid)
        reader     = PollingReader(fd_chunks.keys())

        while reader.fds:
            # nuke if we're past cutoff
            if accountant.total >= limit:
                popened.kill()

                break

            # read from and audit the child process
            (chunk_fd, chunk) = reader.read(resolution)

            accountant.audit()

            if chunk is not None:
                log.debug(
                    "got %i bytes at %s (user time) on fd %i; chunk follows:\n%s",
                    len(chunk),
                    accountant.total,
                    chunk_fd,
                    chunk,
                    )

                if chunk != "":
                    fd_chunks[chunk_fd].append((accountant.total, chunk))
                else:
                    reader.unregister([chunk_fd])

        # wait for our child to die
        (exit_pid, termination, usage) = os.wait4(popened.pid, 0)

        # nuke the session from orbit (it's the only way to be sure)
        borg.unix.sessions.kill_session(popened.pid, signal.SIGKILL)
    except:
        # something has gone awry, so we need to kill our children
        log.warning("something went awry! (our pid is %i)", os.getpid())

        raised = borg.util.Raised()

        if exit_pid is None and popened is not None:
            try:
                # nuke the entire session
                borg.unix.sessions.kill_session(popened.pid, signal.SIGKILL)

                # and don't leave the child as a zombie
                os.waitpid(popened.pid, 0)
            except:
                borg.util.Raised().print_ignored()

        raised.re_raise()
    else:
        # grab any output left in the kernel buffers
        while reader.fds:
            (chunk_fd, chunk) = reader.read(128.0)

            if chunk:
                fd_chunks[chunk_fd].append((accountant.total, chunk))
            elif chunk_fd:
                reader.unregister([chunk_fd])
            else:
                raise RuntimeError("final read from child timed out; undead child?")

        # done
        from datetime import timedelta

        return \
            CPU_LimitedRun(
                started,
                limit,
                fd_chunks[popened.stdout.fileno()],
                fd_chunks[popened.stderr.fileno()],
                timedelta(seconds = usage.ru_utime),
                accountant.total,
                os.WEXITSTATUS(termination) if os.WIFEXITED(termination) else None,
                os.WTERMSIG(termination) if os.WIFSIGNALED(termination) else None,
                )
    finally:
        # let's not leak file descriptors
        if popened is not None:
            popened.stdout.close()
            popened.stderr.close()

