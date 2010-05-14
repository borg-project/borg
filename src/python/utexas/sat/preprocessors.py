"""
utexas/sat/preprocessors.py

Run satisfiability preprocessors.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from os.path          import join
from abc              import (
    abstractmethod,
    abstractproperty,
    )
from cargo.io         import mkdtemp_scoped
from cargo.log        import get_logger
from cargo.sugar      import ABC
from cargo.flags      import (
    Flag,
    Flags,
    )
from cargo.errors     import Raised

log = get_logger(__name__)

class SAT_PreprocessorError(RuntimeError):
    """
    The preprocessor failed in an unexpected way.
    """

class SAT_PreprocessorOutput(ABC):
    """
    A preprocessed instance with associated variable maps.
    """

    @abstractproperty
    def preprocessed(self):
        """
        Did the preprocessor sucessfully preprocess the instance?
        """

    @abstractproperty
    def elapsed(self):
        """
        Time elapsed in preprocessor execution.
        """

    @abstractproperty
    def cnf_path(self):
        """
        The path to the post-preprocessed CNF, if any.
        """

    @abstractproperty
    def solver_result(self):
        """
        The result of the integrated solver, if any.
        """

    @abstractmethod
    def extend(self, certificate):
        """
        Extend the specified certificate.

        Translates a solution to the preprocessed CNF expression back into a
        solution to the unprocessed CNF expression.
        """

class SAT_Preprocessor(ABC):
    """
    Preprocess SAT instances.
    """

    @abstractproperty
    def name(self):
        """
        Canonical name of the preprocessor.
        """

    @abstractmethod
    def preprocess(self, input_path, output_dir, cutoff = None):
        """
        Preprocess an instance.
        """

class SAT_UncompressingPreprocessor(SAT_Preprocessor):
    """
    Uncompress and then preprocess SAT instances.
    """

    def __init__(self, preprocesor):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        self.preprocessor = preprocessor

    def preprocess(self, input_path, output_dir, cutoff = None):
        """
        Preprocess an instance.
        """

        log.info("starting to preprocess %s", input_path)

        # FIXME use a temp file, not directory

        with mkdtemp_scoped(prefix = "sat_preprocessor.") as sandbox_path:
            # decompress the instance, if necessary
            from cargo.io import decompress_if

            uncompressed_path = \
                decompress_if(
                    input_path,
                    join(sandbox_path, "uncompressed.cnf"),
                    )

            log.info("uncompressed task is %s", uncompressed_path)

            # then preprocess it
            return self.prepreocessor.preprocess(uncompressed_path, output_dir, cutoff)

class SAT_LookupPreprocessor(SAT_Preprocessor):
    """
    Use a named SAT preprocessor.
    """

    def __init__(self, name):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        self.name = name

    def preprocess(self, input_path, output_dir, cutoff = None):
        """
        Preprocess an instance.
        """

        raise NotImplementedError()

class SatELiteOutput(SAT_PreprocessorOutput):
    """
    Result of the SatELite preprocessor.
    """

    def __init__(self, run, binary_path, output_dir, solver_result):
        """
        Initialize.
        """

        SAT_PreprocessorOutput.__init__(self)

        self.run            = run
        self.binary_path    = binary_path
        self.output_dir     = output_dir
        self._solver_result = solver_result

    @property
    def preprocessed(self):
        """
        Did the preprocessor sucessfully preprocess the instance?
        """

        return bool(self.cnf_path)

    @property
    def elapsed(self):
        """
        Time elapsed in preprocessor execution.
        """

        return self.run.proc_elapsed

    @property
    def cnf_path(self):
        """
        The path to the preprocessed CNF.
        """

        if self.output_dir is None:
            return None
        else:
            return join(self.output_dir, "preprocessed.cnf")

    @property
    def solver_result(self):
        """
        The result of the integrated solver, if any.
        """

        return self._solver_result

    def extend(self, certificate):
        """
        Extend the specified certificate.
        """

        if self.cnf_path is None:
            raise RuntimeError("extend() on SatELite output that has no CNF")

        # write the certificate to a file
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile("w", prefix = "sat_certificate.") as certificate_file:
            certificate_file.write("SAT\n")
            certificate_file.write(" ".join(str(l) for l in certificate))
            certificate_file.write("\n")
            certificate_file.flush()

            with mkdtemp_scoped(prefix = "tmp.satelite.") as tmpdir:
                # run the solver
                command = [
                    self.binary_path,
                    "+ext",
                    self.cnf_path,
                    certificate_file.name,
                    join(self.output_dir, "variable_map"),
                    join(self.output_dir, "eliminated_clauses"),
                    ]

                log.note("extending model certificate with %s", command)

                popened = None

                try:
                    # launch SatELite
                    import subprocess

                    from os         import putenv
                    from subprocess import Popen

                    popened = \
                        Popen(
                            command,
                            stdin      = None,
                            stdout     = subprocess.PIPE,
                            stderr     = subprocess.STDOUT,
                            preexec_fn = lambda: putenv("TMPDIR", tmpdir),
                            )

                    # parse the extended certificate from its output
                    from itertools import imap

                    extended = None

                    for line in popened.stdout:
                        if line.startswith("v "):
                            literals = imap(int, line[2:].split())

                            if extended is None:
                                extended = list(literals)
                            else:
                                extended.extend(literals)

                    # wait for its natural death
                    popened.wait()
                except:
                    raised = Raised()

                    if popened is not None and popened.poll() is None:
                        try:
                            popened.kill()
                            popened.wait()
                        except:
                            Raised().print_ignored()

                    raised.re_raise()
                else:
                    if popened.returncode != 10:
                        raise SAT_PreprocessorError("model extension failed")

                    return extended

class SatELitePreprocessor(SAT_Preprocessor):
    """
    The standard SatELite preprocessor.
    """

    class_flags = \
        Flags(
            "SatELite Configuration",
            Flag(
                "--satelite-file",
                default = "./satelite",
                metavar = "FILE",
                help    = "binary is FILE [%default]",
                ),
            )

    def __init__(self, flags = {}):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        flags = SatELitePreprocessor.class_flags.merged(flags)

        self.binary_path = flags.satelite_file

    def preprocess(self, input_path, output_dir, cutoff = None):
        """
        Preprocess the instance.
        """

        # FIXME better support for the no-cutoff case

        from cargo.temporal     import TimeDelta
        from utexas.sat.solvers import scan_competition_output

        if cutoff is None:
            cutoff = TimeDelta(seconds = 1e6)

        with mkdtemp_scoped(prefix = "satelite.") as tmpdir:
            # run the solver
            from cargo.unix.accounting import run_cpu_limited

            command = [
                self.binary_path,
                input_path,
                join(output_dir, "preprocessed.cnf"),
                join(output_dir, "variable_map"),
                join(output_dir, "eliminated_clauses"),
                ]

            run = \
                run_cpu_limited(
                    command,
                    cutoff,
                    pty         = True,
                    environment = {
                        "TMPDIR": tmpdir,
                        },
                    )

        # interpret its behavior
        if run.exit_status in (10, 20):
            from utexas.sat.solvers import SAT_BareResult

            out_lines                  = "".join(c for (t, c) in run.out_chunks).split("\n")
            (satisfiable, certificate) = scan_competition_output(out_lines)
            result                     = SAT_BareResult(satisfiable, certificate)

            return SatELiteOutput(run, self.binary_path, None, result)
        elif run.exit_status == 0:
            return SatELiteOutput(run, self.binary_path, output_dir, None)
        else:
            return SatELiteOutput(run, self.binary_path, None, None)

    @property
    def name(self): return "SatELite"

