import sys
import logging

class DefaultLogger(logging.getLoggerClass()):
    """Simple standard logging."""

    def __init__(self, name, level = logging.NOTSET):
        logging.Logger.__init__(self, name, level)

        self.is_squeaky_clean = True

    def detail(self, message, *args, **kwargs):
        """Write a log message at the DETAIL level."""

        return self.log(logging.DETAIL, message, *args, **kwargs)

    def note(self, message, *args, **kwargs):
        """Write a log message at the NOTE level."""

        return self.log(logging.NOTE, message, *args, **kwargs)

# global customization (unfortunate, but whatever)
logging.setLoggerClass(DefaultLogger)

logging.DETAIL = 15
logging.NOTE = 25

logging.addLevelName(logging.DETAIL, "DETAIL")
logging.addLevelName(logging.NOTE, "NOTE")

class TTY_Formatter(logging.Formatter):
    """A log formatter for console output."""

    _DATE_FORMAT = "%H:%M:%S"
    _TIME_COLOR  = "\x1b[34m"
    _NAME_COLOR  = "\x1b[35m"
    _LEVEL_COLOR = "\x1b[33m"
    _COLOR_END   = "\x1b[00m"

    def __init__(self, stream = None):
        """
        Construct this formatter.

        Provides colored output if the stream parameter is specified and is an acceptable TTY.
        We print hardwired escape sequences, which will probably break in some circumstances;
        for this unfortunate shortcoming, we apologize.
        """

        # select and construct format string
        import curses

        format = None

        if stream and hasattr(stream, "isatty") and stream.isatty():
            curses.setupterm()

            # FIXME do nice block formatting, increasing column sizes as necessary
            if curses.tigetnum("colors") > 2:
                format = \
                    "%s%%(asctime)s%s %%(message)s" % (
                        TTY_Formatter._NAME_COLOR,
                        TTY_Formatter._COLOR_END,
                        )

        if format is None:
            format = "%(name)s - %(levelname)s - %(message)s"

        # initialize this formatter
        logging.Formatter.__init__(self, format, TTY_Formatter._DATE_FORMAT)

def log_level_to_number(level):
    """Convert a level name to a level number, if necessary."""

    if type(level) is str:
        return logging._levelNames[level]
    else:
        return level

def get_logger(name = None, level = None, default_level = logging.INFO):
    """Get or create a logger."""

    if name is None:
        logger = logging.root
    else:
        logger = logging.getLogger(name)

    # set the default level, if the logger is new
    try:
        clean = logger.is_squeaky_clean
    except AttributeError:
        pass
    else:
        if clean and default_level is not None:
            logger.setLevel(log_level_to_number(default_level))

    # unconditionally set the logger level, if requested
    if level is not None:
        logger.setLevel(log_level_to_number(level))

        logger.is_squeaky_clean = False

    return logger

def enable_default_logging():
    """Set up logging in the typical way."""

    if not enable_default_logging.enabled:
        get_logger(level = "NOTSET")

        # build a handler
        handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(TTY_Formatter(sys.stdout))

        # add it
        logging.root.addHandler(handler)

        enable_default_logging.enabled = True

enable_default_logging.enabled = False

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

get_logger("borg").addHandler(NullHandler())

