"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from utexas.sat.preprocessors import (
    SAT_Preprocessor,
    PreprocessorResult,
    )

class SatELiteOutput(PreprocessorResult):
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

        from os.path import join

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

        from cargo.errors import Raised

        # sanity
        if self.cnf_path is None:
            raise RuntimeError("extend() on SatELite output that has no CNF")

        # write the certificate to a file
        from tempfile import NamedTemporaryFile
        from cargo.io import mkdtemp_scoped

        with NamedTemporaryFile("w", prefix = "sat_certificate.") as certificate_file:
            certificate_file.write("SAT\n")
            certificate_file.write(" ".join(str(l) for l in certificate))
            certificate_file.write("\n")
            certificate_file.flush()

            with mkdtemp_scoped(prefix = "tmp.satelite.") as tmpdir:
                # run the solver
                from os.path import join

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
                    from utexas.sat.solvers import scan_competition_output

                    (_, certificate) = scan_competition_output(popened.stdout)

                    # wait for its natural death
                    popened.wait()
                except:
                    # something went wrong; make sure it's dead
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

    def __init__(self, command):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        self._command = command

    def preprocess(self, task, budget, output_dir, random, environment):
        """
        Preprocess the instance.
        """

        # argument sanity
        from utexas.sat import SAT_FileTask

        if not isinstance(task, SAT_FileTask):
            raise TypeError("SatELite requires a file-backed task")

        # preprocess the task
        from cargo.io           import mkdtemp_scoped

        with mkdtemp_scoped(prefix = "tmp.satelite.") as tmpdir:
            # run the solver
            from os.path               import join
            from cargo.unix.accounting import run_cpu_limited

            arguments = [
                task.path,
                join(output_dir, "preprocessed.cnf"),
                join(output_dir, "variables_map"),
                join(output_dir, "eliminated_clauses"),
                ]
            run       = \
                run_cpu_limited(
                    self._command + arguments,
                    budget,
                    pty         = True,
                    environment = {
                        "TMPDIR": tmpdir,
                        },
                    )

        # interpret its behavior
        from utexas.sat.preprocessors import BarePreprocessorRunResult

        if run.exit_status in (10, 20):
            from utexas.sat         import SAT_Answer
            from utexas.sat.solvers import scan_competition_output

            out_lines                  = "".join(c for (t, c) in run.out_chunks).split("\n")
            (satisfiable, certificate) = scan_competition_output(out_lines)
            answer                     = SAT_Answer(satisfiable, certificate)

            return BarePreprocessorRunResult(self, task, task, answer, run)
        elif run.exit_status == 0:
            return BarePreprocessorRunResult(self, task, "FIXME", None, run)
        else:
            return BarePreprocessorRunResult(self, task, task, None, run)

