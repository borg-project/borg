"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import re
import datetime

class ProcFileParseError(RuntimeError):
    """A file in /proc could not be parsed."""

class ProcessStat(object):
    """
    Information about a specific process.

    Merely a crude wrapper around the information in the /proc/<pid>/stat file.
    Read the man pages! Read the kernel source! Nothing in /proc is ever quite
    as it seems.
    """

    __ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    __entry_re         = re.compile("\\d+")
    __stat_re_strings  = [
        # signedness decisions were made by examining the kernel source, and in some
        # cases (eg pid) don't make much sense---but who are we in userland to judge?
        "(?P<pid>-?\\d+)",      # process pid
        "(?P<name>\\(.+\\))",   # executable name
        "(?P<state>[RSDZTWX])", # process state
        "(?P<ppid>-?\\d+)",     # parent's pid
        "(?P<pgid>-?\\d+)",     # process group id
        "(?P<sid>-?\\d+)",      # session id
        "(?P<tty>-?\\d+)",      # tty number
        "(?P<ttyg>-?\\d+)",     # group id of the process which owns the associated tty
        "(?P<flags>\\d+)",      # kernel flags word (kernel-version-dependent)
        "(?P<min>\\d+)",        # minor faults count
        "(?P<cmin>\\d+)",       # waited-for-children minor faults count
        "(?P<maj>\\d+)",        # major faults count
        "(?P<cmaj>\\d+)",       # waited-for-children major faults count
        "(?P<utime>\\d+)",      # user mode jiffies count
        "(?P<stime>\\d+)",      # kernel mode jiffies count
        "(?P<cutime>-?\\d+)",   # waited-for-children user mode jiffies count
        "(?P<cstime>-?\\d+)",   # waited-for-children kernel mode jiffies count
        "(?P<priority>-?\\d+)", # real-time priority or raw nice value
        "(?P<nice>-?\\d+)",     # signed nice value in [-19, 19]
        "(?P<nthreads>-?\\d+)", # number of threads in the process (replaced removed field)
        "0",                    # removed-field placeholder
        "(?P<start>\\d+)",      # process start time in jiffies
        "(?P<vsize>\\d+)",      # bytes of process virtual memory
        "(?P<rss>-?\\d+)",      # resident set size minus three
        "(?P<rlim>\\d+)",       # rss limit in bytes
        "(?P<pbot>\\d+)",       # program text bottom address
        "(?P<ptop>\\d+)",       # program text top address
        "(?P<stack>\\d+)",      # stack start address
        "(?P<esp>\\d+)",        # stack pointer address
        "(?P<eip>\\d+)",        # instruction pointer address
        "(?P<pending>\\d+)",    # pending signals bitmap
        "(?P<blocked>\\d+)",    # blocked signals bitmap
        "(?P<ignored>\\d+)",    # ignored signals bitmap
        "(?P<caught>\\d+)",     # caught signals bitmap
        "(?P<wchan>\\d+)",      # process wait channel
        "\\d+",                 # zero (in the past, pages swapped)
        "\\d+",                 # zero (in the past, childrens' pages swapped)
        "(?P<dsig>-?\\d+)",     # death signal to parent
        "(?P<cpu>-?\\d+)",      # last CPU of execution
        "(?P<rtprio>\\d+)",     # real-time scheduling priority
        "(?P<policy>\\d+)",     # scheduling policy
        "(?P<blkio>\\d+)",      # clock ticks of block I/O delays
        "(?P<gtime>\\d+)",      # process guest time in clock ticks
        "(?P<cgtime>\\d+)",     # waited-for-children's guest time in clock ticks
        ]
    __stat_res = [re.compile(s) for s in __stat_re_strings]

    def __init__(self, pid):
        """Read and parse /proc/<pid>/stat."""

        with open("/proc/%i/stat" % pid) as file:
            stat = file.read()

        strings  = stat.split()
        self.__d = {
            "pid"   : strings[0],
            "sid"   : strings[5],
            "utime" : strings[13],
            }

#         for i in fields:
#             m = ProcessStat.__stat_res[i].match(strings[i])

#             self.__d.update(m.groupdict())

    @staticmethod
    def all():
        """
        Iterate over all processes on the system.

        Grabs a list of pids from /proc and iterates over them, skipping any
        processes which have terminated by the time they are reached in the
        iteration. The returned information is therefore not a perfect snapshot
        of system state, but we have no alternative.
        """

        for name in os.listdir("/proc"):
            m = ProcessStat.__entry_re.match(name)

            if m:
                try:
                    yield ProcessStat(int(name))
                except IOError:
                    pass

    @staticmethod
    def in_session(sid):
        """Iterate over all processes in a session."""

        for process in ProcessStat.all():
            if process.sid == sid:
                yield process

    def __ticks_to_timedelta(self, ticks):
        """Convert kernel clock ticks to a Python timedelta value."""

        return datetime.timedelta(seconds = float(ticks) / self.__ticks_per_second)

    # expose the relevant fields
    pid                 = property(lambda self: int(self.__d["pid"]))
    name                = property(lambda self: self.__d["name"])
    state               = property(lambda self: self.__d["state"])
    ppid                = property(lambda self: int(self.__d["ppid"]))
    pgid                = property(lambda self: int(self.__d["pgid"]))
    sid                 = property(lambda self: int(self.__d["sid"]))
    tty                 = property(lambda self: int(self.__d["tty"]))
    tty_owner_group     = property(lambda self: int(self.__d["ttyg"]))
    flags               = property(lambda self: long(self.__d["flags"]))
    minor_faults        = property(lambda self: long(self.__d["min"]))
    child_minor_faults  = property(lambda self: long(self.__d["cmin"]))
    major_faults        = property(lambda self: long(self.__d["maj"]))
    child_major_faults  = property(lambda self: long(self.__d["cmaj"]))
    user_time           = property(lambda self: self.__ticks_to_timedelta(self.__d["utime"]))
    kernel_time         = property(lambda self: self.__ticks_to_timedelta(self.__d["stime"]))
    child_user_time     = property(lambda self: self.__ticks_to_timedelta(self.__d["cutime"]))
    child_kernel_time   = property(lambda self: self.__ticks_to_timedelta(self.__d["cstime"]))
    priority            = property(lambda self: int(self.__d["priority"]))
    nice                = property(lambda self: int(self.__d["nice"]))
    threads             = property(lambda self: int(self.__d["nthreads"]))
    start_time          = property(lambda self: self.__ticks_to_timedelta(self.__d["start"]))
    virtual_size        = property(lambda self: long(self.__d["vsize"]))
    resident_set_size   = property(lambda self: int(self.__d["rss"]))
    resident_set_limit  = property(lambda self: long(self.__d["rlim"]))
    text_bottom         = property(lambda self: long(self.__d["pbot"]))
    text_top            = property(lambda self: long(self.__d["ptop"]))
    stack_start         = property(lambda self: long(self.__d["stack"]))
    stack_pointer       = property(lambda self: long(self.__d["esp"]))
    instruction_pointer = property(lambda self: long(self.__d["eip"]))
    pending_signals     = property(lambda self: long(self.__d["pending"]))
    blocked_signals     = property(lambda self: long(self.__d["blocked"]))
    ignored_signals     = property(lambda self: long(self.__d["ignored"]))
    caught_signals      = property(lambda self: long(self.__d["caught"]))
    wait_channel        = property(lambda self: long(self.__d["wchan"]))
    exit_signal         = property(lambda self: int(self.__d["dsig"]))
    last_cpu            = property(lambda self: int(self.__d["cpu"]))
    priority            = property(lambda self: long(self.__d["rtprio"]))
    policy              = property(lambda self: long(self.__d["policy"]))
    io_delay            = property(lambda self: long(self.__d["blkio"]))
    guest_time          = property(lambda self: self.__ticks_to_timedelta(self.__d["gtime"]))
    child_guest_time    = property(lambda self: self.__ticks_to_timedelta(self.__d["cgtime"]))

def get_pid_utime(pid):
    return ProcessStat(pid).user_time

def get_sid_utime(sid):
    return sum(p.user_time for p in ProcessStat.in_session(sid))

