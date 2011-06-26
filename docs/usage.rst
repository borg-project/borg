Using borg
==========

Training a portfolio solver
---------------------------

Command line:::

    $ python -m borg.tools.train <TRAINED_PORTFOLIO> <PORTFOLIO_TYPE> <PATH_TO_SUBSOLVERS_MODULE> <PATH_TO_TRAINING_INSTANCES>

XXX.

File format: subsolver run records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run records are stored in CSV files with the suffix ``.rtd.csv``. The following
columns are expected, in the following order:

``solver``
    Unique name of the solver.

``seed``
    Random seed associated with the run. *Not used.*

``budget``
    Budget allotted to the run, in CPU seconds.

``cost``
    Computational cost of the run, in CPU seconds.

``answer``
    Short-form answer returned by the run. Any non-empty answer is considered a
    successful run.

File format: instance features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instance features are stored in CSV files with the suffix ``.features.csv``.
The column must be ``cost``, the computational cost of feature computation, in
CPU seconds. The remaining columns are domain-specific, one per feature.

Running a portfolio solver
--------------------------

XXX.

