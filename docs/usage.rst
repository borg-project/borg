Using borg
==========

Assembling a portfolio of subsolvers
------------------------------------

The first step in building a portfolio solver is to assemble its constituent
solvers (referred to as "subsolvers") for the problem domain. The borg project
makes several subsolver collections available. Let's assume that you are
interested in solving instances of pseudo-Boolean satisfiability (PB), and
download a PB subsolver collection:

http://nn.cs.utexas.edu/pages/research/borg-bulk/solvers-pb.tar.gz

Collecting solver performance data
----------------------------------

The second step is to collect subsolver performance data for use in training.
Each subsolver in the portfolio is run on each problem instance in the training
set, often multiple times. Since this process requires an (unsurprisingly)
enormous amount of computational time, and the borg project makes sets of
training data publicly available, let's download one of those sets:

http://nn.cs.utexas.edu/pages/research/borg-bulk/tasks-pb-pre11.tar.xz

File format: subsolver run records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run records are stored in CSV files with the suffix ``.runs.csv``. The
following columns are expected in the following order:

``solver``
    Unique name of the solver.

``budget``
    Budget allotted to the run, in CPU seconds.

``cost``
    Computational cost of the run, in CPU seconds.

``succeeded``
    Did the solver succeed on this run?

``answer``
    Base64-encoded gzipped pickled answer returned by the solver on this run,
    if any.

File format: instance features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instance features are stored in CSV files with the suffix ``.features.csv``.
The first column must be ``cost``, the computational cost of feature
computation, in CPU seconds. The remaining columns are domain-specific, one per
feature.

Training a portfolio solver
---------------------------

This section will walk you through the process of training a borg portfolio.

At this point we have both a suite of subsolvers and a set of training data.
The third and penultimate step in constructing a borg portfolio is fitting a
predictive model to these data. With subsolvers and training data downloaded
and unpacked, after activating the virtualenv in which borg is installed, use
the "train" tool to fit a model:

.. code-block:: bash

    $ python -m borg.tools.train borg-pb.model.pickle borg-mix+class solvers/pb/portfolio.py tasks/pb/categorized/dec-smallint-lin

This process can take ten minutes or more, depending on the amount of training
data and the nubmer of subsolvers. It will write a portfolio model (of type
"borg-mix+class") to the file "pb-model.pickle".

Finally, we need to calibrate the solver to the local execution environment,
since the machines used to collect training data may be faster or slower than
the machine on which you're running the portfolio. Every collection of training
data includes a "calibration" directory, which contains a problem instance, a
SOLVER_NAME file, and a runs file with the ".runs.train.csv" suffix. This
directory stores runs made by a single solver (that named in SOLVER_NAME), on a
single instance, on the machine(s) used for training data.

To collect runs using the same solver on the local machine, run

.. code-block:: bash

    $ python -m borg.tools.run_solvers solvers/pb/portfolio.py tasks/pb/calibration/ 120 -r 9 -suffix .local.runs.csv -only_solver $(cat tasks/pb/calibration/SOLVER_NAME)

which will make 9 runs and store them in the corresponding "<instance>.local.runs.csv" file.

Then compute the local machine calibration factor with:

.. code-block:: bash

    $ python -m borg.tools.get_calibration tasks/pb/calibration/normalized-cache-ibm-q-unbounded.Icl2arity.ucl.opb.{local,train}.runs.csv

The ratio that it prints can be used as a value for the "--speed" parameter to
the "solve" tool discussed below.

Running a portfolio solver
--------------------------

Now let's solve the same calibration using the full portfolio, with

.. code-block:: bash

    $ python -m borg.tools.solve --speed 1.0 borg-pb.model.pickle solvers/pb/portfolio.py tasks/pb/calibration/normalized-cache-ibm-q-unbounded.Icl2arity.ucl.opb

changing the speed parameter given the output of the "get_calibration" tool
above.

Borg will parse the instance, compute instance features, condition its internal
model, and run a sequence of solvers---replanning as necessary. In this case,
it should quickly solve the instance with its first solver run.

