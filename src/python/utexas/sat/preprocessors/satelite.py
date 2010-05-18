"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log                import get_logger
from utexas.rowed             import Rowed
from utexas.sat.tasks         import AbstractPreprocessedFileTask
from utexas.sat.preprocessors import SAT_Preprocessor
from utexas.rowed             import Rowed

log = get_logger(__name__)

class SatELitePreprocessor(Rowed, SAT_Preprocessor):
    """
    The standard SatELite preprocessor.
    """

    def __init__(self, command, relative_to):
        """
        Initialize.
        """

        SAT_Preprocessor.__init__(self)

        self._command = [s.replace("$HERE", relative_to) for s in command]

    def preprocess(self, task, budget, output_path, random, environment):
        """
        Preprocess the instance.
        """

        # argument sanity
        from utexas.sat import AbstractFileTask

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
        from utexas.sat.preprocessors import BarePreprocessorRunResult

        if run.exit_status in (10, 20):
            from utexas.sat         import SAT_Answer
            from utexas.sat.solvers import scan_competition_output

            out_lines                  = "".join(c for (t, c) in run.out_chunks).split("\n")
            (satisfiable, certificate) = scan_competition_output(out_lines)
            answer                     = SAT_Answer(satisfiable, certificate)

            return BarePreprocessorRunResult(self, None, task, task, answer, run)
        elif run.exit_status == 0:
            output_task = SatELitePreprocessedTask(self, None, task, output_path)

            return BarePreprocessorRunResult(self, None, task, output_task, None, run)
        else:
            return BarePreprocessorRunResult(self, None, task, task, None, run)

    def extend(self, task, answer, environment):
        """
        Extend an answer to a preprocessed task to its parent task.
        """

        # sanity
        from utexas.sat.tasks import AbstractPreprocessedFileTask

        if not isinstance(task, AbstractPreprocessedFileTask):
            raise TypeError("tasks to extend must be preprocessed file-backed tasks")

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
                    from utexas.sat         import SAT_Answer
                    from utexas.sat.solvers import scan_competition_output

                    (_, extended)   = scan_competition_output(popened.stdout)
                    extended_answer = SAT_Answer(answer.satisfiable, extended)

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

class SatELitePreprocessedTask(Rowed, AbstractPreprocessedFileTask):
    """
    A task preprocessed by SatELite.
    """

    def __init__(self, preprocessor, seed, input_task, output_path, row = None):
        """
        Initialize.
        """

        Rowed.__init__(self, row)

        self._preprocessor = preprocessor
        self._seed         = seed
        self._input_task   = input_task
        self._output_path  = output_path

    def get_new_row(self, session, preprocessor_row = None, **kwargs):
        """
        Get or create the ORM row associated with this object.
        """

        from sqlalchemy  import and_
        from utexas.data import PreprocessedTaskRow as PT

        if preprocessor_row is None:
            preprocessor_row = self.preprocessor.get_row(session)

        input_task_row         = self.input_task.get_row(session)
        preprocessed_task_row  =                         \
            session                                      \
            .query(PT)                                   \
            .filter(
                and_(
                    PT.preprocessor == preprocessor_row,
                    PT.seed         == self.seed,
                    PT.input_task   == input_task_row,
                ),
            )                                            \
            .first()

        if preprocessed_task_row is None:
            preprocessed_task_row = \
                PT(
                    preprocessor = preprocessor_row,
                    seed         = self.seed,
                    input_task   = input_task_row,
                    )

            session.add(preprocessed_task_row)

        return preprocessed_task_row

    @property
    def preprocessor(self):
        """
        The preprocessor that yielded this task.
        """

        return self._preprocessor

    @property
    def seed(self):
        """
        The preprocessor seed on the run that yielded this task.
        """

        return self._seed

    @property
    def input_task(self):
        """
        The preprocessor input task that yielded this task.
        """

        return self._input_task

    @property
    def path(self):
        """
        The path to the associated task file.
        """

        from os.path import join

        return join(self._output_path, "preprocessed.cnf")

    @property
    def output_path(self):
        """
        The path to the directory of preprocessor output files.
        """

        return self._output_path

