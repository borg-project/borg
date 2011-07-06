Using borg
==========

Training a portfolio solver
---------------------------

Command line:::

    $ python -m borg.tools.train <TRAINED_PORTFOLIO> <PORTFOLIO_TYPE> <PATH_TO_SUBSOLVERS_MODULE> <PATH_TO_TRAINING_INSTANCES>

XXX.

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

Running a portfolio solver
--------------------------

XXX.

