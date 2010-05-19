"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log    import get_logger
from borg.rowed   import Rowed
from borg.tasks   import AbstractPreprocessedTask
from borg.solvers import AbstractPreprocessor

log = get_logger(__name__)

class SatELitePreprocessor(Rowed, AbstractPreprocessor):
    """
    The standard SatELite preprocessor.
    """

    def __init__(self, command, relative_to = None):
        """
        Initialize.
        """

        Rowed.__init__(self)

        if relative_to is None:
            self._command = command
        else:
            self._command = [s.replace("$HERE", relative_to) for s in command]

    def solve(self, task, budget, random, environment):
        """
        Attempt to solve the specified instance.
        """

        from cargo.io              import mkdtemp_scoped
        from borg.solvers.attempts import RunAttempt

        with mkdtemp_scoped() as output_path:
            attempt = self.preprocess(task, budget, output_path, random, environment)

        return RunAttempt(self, task, attempt.answer, attempt.seed, attempt.run)

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess the instance.
        """

        # argument sanity
        from borg.tasks import AbstractFileTask

        if not isinstance(task, AbstractFileTask):
            raise TypeError("SatELite requires a file-backed task")

        # preprocess the task
        from cargo.io import mkdtemp_scoped

        with mkdtemp_scoped(prefix = "tmp.satelite.") as tmpdir:
            # run the solver
            from os.path               import join
            from cargo.unix.accounting import run_cpu_limited

            arguments = [
                task.path,
                join(output_path, "preprocessed.cnf"),
                join(output_path, "variables_map"),
                join(output_path, "eliminated_clauses"),
                ]

            log.debug("SatELite arguments are %s", arguments)

            run = \
                run_cpu_limited(
                    self._command + arguments,
                    budget,
                    pty         = True,
                    environment = {
                        "TMPDIR": tmpdir,
                        },
                    )

        # interpret its behavior
        from borg.solvers.attempts import PreprocessorAttempt

        if run.exit_status in (10, 20):
            from borg.sat                 import SAT_Answer
            from borg.solvers.competition import scan_competition_output

            out_lines = "".join(c for (t, c) in run.out_chunks).split("\n")
            answer    = scan_competition_output(out_lines)

            return PreprocessorAttempt(self, task, answer, None, run, task)
        elif run.exit_status == 0:
            from borg.tasks import PreprocessedDirectoryTask

            output_task = PreprocessedDirectoryTask(self, None, task, output_path, "preprocessed.cnf")

            return PreprocessorAttempt(self, task, None, None, run, output_task)
        else:
            return PreprocessorAttempt(self, task, None, None, run, task)

    def extend(self, task, answer, environment):
        """
        Extend an answer to a preprocessed task to its parent task.
        """

        # sanity
        from borg.tasks import AbstractPreprocessedTask

        assert isinstance(task, AbstractPreprocessedTask)

        # trivial cases
        if answer.certificate is None:
            return answer

        # typical case
        from tempfile     import NamedTemporaryFile
        from cargo.errors import Raised

        with NamedTemporaryFile("w", prefix = "sat_certificate.") as certificate_file:
            # write the certificate to a file
            certificate_file.write("SAT\n")
            certificate_file.write(" ".join(str(l) for l in answer.certificate))
            certificate_file.write("\n")
            certificate_file.flush()

            # prepare to invoke the solver
            from cargo.io import mkdtemp_scoped

            with mkdtemp_scoped(prefix = "tmp.satelite.") as tmpdir:
                # run the solver
                from os.path import join

                arguments = [
                    "+ext",
                    task.path,
                    certificate_file.name,
                    join(task.output_path, "variable_map"),
                    join(task.output_path, "eliminated_clauses"),
                    ]

                log.note("model extension arguments are %s", arguments)

                popened = None

                try:
                    # launch SatELite
                    import subprocess

                    from os         import putenv
                    from subprocess import Popen

                    popened = \
                        Popen(
                            self._command + arguments,
                            stdin      = None,
                            stdout     = subprocess.PIPE,
                            stderr     = subprocess.STDOUT,
                            preexec_fn = lambda: putenv("TMPDIR", tmpdir),
                            )

                    # parse the extended certificate from its output
                    from borg.solvers.competition import scan_competition_output

                    extended_answer = \
                        scan_competition_output(
                            popened.stdout,
                            satisfiable = answer.satisfiable,
                            )

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

                    return extended_answer

    def make_task(self, seed, input_task, output_path, environment, row = None):
        """
        Construct an appropriate preprocessed task from its output directory.
        """

        from borg.tasks import PreprocessedDirectoryTask

        return PreprocessedDirectoryTask(self, seed, input_task, output_path, "preprocessed.cnf")

