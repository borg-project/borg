Using borg
**********

This chapter walks through the basic steps in using borg:

#. assembling solvers for some problem domain together into a "portfolio";
#. collecting performance data for each solver in that portfolio;
#. training a single portfolio solver to make solver execution decisions; and
#. running that portfolio solver on instances of the domain.

Assembling a portfolio of subsolvers
====================================

The first step in building a portfolio solver is to assemble its constituent
solvers for the problem domain. We will often refer to these constituent
solvers as "subsolvers", since they are wrapped by an outer "portfolio solver".

These subsolvers must be selected and prepared for execution, and then borg
must be configured to execute them and to interpret their output. Here, we will
build a portfolio solver for the `SAT problem
<http://www.satisfiability.org/>`_.

Fetching SAT solver binaries
----------------------------

In this example, we will build a simple portfolio consisting of several solvers
from the `2011 SAT competition <http://www.satcompetition.org/2011/>`_. Much of
the data from these competitions can be accessed at

http://www.satcompetition.org/

including static binaries of the solvers entered into the competition. Let's
download these and unpack them:

.. code-block:: bash

    $ wget http://www.cril.univ-artois.fr/SAT11/solvers/SAT2011-static-binaries.tar.gz
    $ tar zxvf SAT2011-static-binaries.tar.gz

.. warning::

    This tarball is 125MiB compressed, and may take some time to download.

After unpacking, we find the solvers under ``SAT2011/bin/static`` and
information about how to execute them inside
``SAT2011/bin/static/CommandLines.txt``.

Selecting solvers to use
------------------------

While a portfolio solver can be a useful tool, it still requires significant
domain knowledge to select the solvers to include in the portfolio. In this
example, we will simply select three of the best-performing solvers from the
`2009 SAT competition <http://www.satcompetition.org/2009/>`_ and the `2010 SAT
race <http://baldur.iti.uka.de/sat-race-2010/>`_:

- ``cryptominisat`` (Mate Soos),
- ``clasp`` (Martin Gebser, Benjamin Kaufmann, and Torsten Schaub), and
- ``TNM`` (Wanxia Wei and Chu Min Li).

This selection follows the lead of the ``ppfolio`` tool, a basic parallel
portfolio that performed well in the 2011 competition. More information is
available in `its documentation
<http://www.cril.univ-artois.fr/~roussel/ppfolio/>`_. 

Since ``ppfolio`` has helpfully packaged these solvers for the 2011
competition, we can access them inside the
``SAT2011/bin/static/main/sat11-11-roussel/bin`` directory. Let's make a
symlink to that directory with

.. code-block:: bash

    $ ln -s SAT2011/bin/static/main/sat11-11-roussel/bin ppfolio-bin

so that we can access it more easily later.

.. TODO provide an example trivial satisfiable instance
.. TODO and demonstrate executing these solvers on that instance

Constructing a solver suite
---------------------------

We will now write a configuration file that allows borg to execute these SAT
solvers by name. In borg, a collection of named solvers is referred to as a
"suite".

A suite configuration file is simply a Python module that establishes how to
execute each solver. Here is a configuration file for the suite of solvers
selected above::

    import borg

    domain = borg.get_domain("sat")
    commands = {
        "cryptominisat": ["{root}/ppfolio-bin/cryptominisat", "--randomize={seed}", "{task}"],
        "clasp": ["{root}/ppfolio-bin/clasp", "--seed={seed}", "{task}"],
        "TNM": ["{root}/ppfolio-bin/TNM", "{task}", "{seed}"],
        }
    solvers = borg.make_solvers(borg.domains.sat.solvers.SAT_SolverFactory, __file__, commands)

We will call this file ``suite_sat_ppfolio.py``.

A solver suite is required to provide two top-level variables:

- ``domain``, which must be an instance of a class such as
  :class:`borg.domains.sat.Satisfiability` that allows ``borg`` to parse
  instances of the problem domain, compute features on those instances, and
  determine basic properties of "answers" to instances of the domain; and
- ``solvers``, a dictionary that maps arbitrary solver names (e.g., "minisat")
  to instances of a solver factory class, such as
  :class:`borg.domains.sat.solvers.SAT_SolverFactory`, that allow ``borg`` to
  initiate solver runs on problem instances and to understand their output.

``borg`` includes support for the output formats of various common solver
types. In this case, the class
:class:`borg.domains.sat.solvers.SAT_SolverFactory` supports the typical output
format of SAT competition entries.

.. TODO link to output format documentation

Collecting solver performance data
==================================

The second step is to collect subsolver performance data for use in training.
Each subsolver in the portfolio is run on each problem instance in the training
set, often multiple times.

Gathering training instances
----------------------------

In this example, we will collect such data for the ppfolio suite on a small set
of instances from the SAT2011 competition. Unfortunately, doing so requires
downloading the entire set of instances from the competition:

.. code-block:: bash

    $ mkdir benchmarks
    $ cd benchmarks
    $ wget http://www.cril.univ-artois.fr/SAT11/bench/SAT11-Competition-SelectedBenchmarks.tar
    $ tar xvf SAT11-Competition-SelectedBenchmarks.tar

.. warning::

    This tarball is 1.7GiB compressed, and may take some time to download.

The archive contains a huge number of individually compressed instances. For
now, we will train our portfolio on a small subset of those instances. The
easiest way to create such a subset is simply to symlink or copy the relevant
instances into a common location---here, into a new directory named "selected":

.. code-block:: bash

    $ mkdir selected
    $ cd selected
    $ cp ../SAT11/random/large/unif-k3-r4.2-v10000* .
    $ cp ../SAT11/application/fuhs/AProVE11/* .
    $ bunzip2 \*.bz2

We have now pulled together an arbitrarily selected set of 20 instances to use
as our training set.

Executing solvers repeatedly
----------------------------

Borg can use an [HTCondor](http://research.cs.wisc.edu/htcondor/) or [IPython
cluster](http://ipython.org/ipython-doc/dev/parallel/index.html) to execute
solvers repeatedly and collect training data. For this experiment, set up a
local IPython cluster by running:

.. code-block:: bash

    $ ipcluster start -n 2

In this invocation, the ipcluster script will launch two engines for parallel
processing. The value of the "-n" argument can be bumped up if you have more
cores.

The borg run_solvers tool collects run duration data. By default, it uses the
local IPython cluster. To invoke it, specify the portfolio configuration above
(in this case, "suite_sat_ppfolio.py"), the directory containing the set of
benchmarks on which to execute the solver suite (in this case,
"benchmarks/selected"), and the run duration cutoff in seconds (in this case,
300 seconds).

.. code-block:: bash

    $ python -m borg.tools.run_solvers suite_sat_ppfolio.py benchmarks/selected/ 300

.. warning::

    Solver run data collection is extremely expensive in general. For example,
    even though this suite of solvers and collection of benchmark instances are
    both quite restricted, this set of runs will take a substantial amount of
    time---12 hours or more---to complete using one or two cores.

File format: subsolver run records
----------------------------------

Run records are typically stored in CSV files with the suffix ``.runs.csv``.
The full suffix can be modified through the "-suffix" flag to run_solvers,
among other borg tools.

The following columns are expected in the following order:

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

Generating instance feature information
=======================================

Portfolios typically use domain-specific information about a given problem
instance to make better solver execution decisions.

Borg's "get_features" tool collects such information for the domains that
it supports. The set of features collected is built into borg.

We can use this tool to collect feature information from the set of
selected benchmarks:

.. code-block:: bash

    python -m borg.tools.get_features sat benchmarks/selected/

Like the run_solvers tool, get_features uses the local IPython cluster by
default.

File format: instance features
------------------------------

Instance features are stored in CSV files with the suffix ``.features.csv``.
The first column must be ``cost``, the computational cost of feature
computation, in CPU seconds. The remaining columns are domain-specific, one per
feature.

Training a portfolio solver
===========================

This section will walk you through the process of training a borg portfolio.

At this point we have both a suite of subsolvers and a set of training data.
The third and penultimate step in constructing a borg portfolio is fitting a
predictive model to these data. Use the "train" tool to fit a model:

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

Running the trained portfolio solver
====================================

Now let's solve the same calibration using the full portfolio, with

.. code-block:: bash

    $ python -m borg.tools.solve --speed 1.0 borg-pb.model.pickle solvers/pb/portfolio.py tasks/pb/calibration/normalized-cache-ibm-q-unbounded.Icl2arity.ucl.opb

changing the speed parameter given the output of the "get_calibration" tool
above.

Borg will parse the instance, compute instance features, condition its internal
model, and run a sequence of solvers---replanning as necessary. In this case,
it should quickly solve the instance with its first solver run.

