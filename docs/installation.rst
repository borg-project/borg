Installing borg
===============

Compilation and setup for borg involves four major steps, described in detail
in the other sections below:

#. installing any missing system-level dependencies;
#. installing the required Python dependencies;
#. installing borg and its required "cargo" library; and
#. calibrating borg to the local machine.

Installing system dependencies
------------------------------

A reasonably complete development environment is required to compile borg and
its dependencies. That includes at least:

* Python >= 2.6
* compilers: gcc, gfortran, and g++
* devel packages for

  * Python
  * linear algebra libraries: BLAS, LAPACK, and/or ATLAS

Most, if not all, of these packages are common dependencies, and should already
be installed on a typical development machine.

Verifying Python >= 2.6
^^^^^^^^^^^^^^^^^^^^^^^

Make sure that you're using a recent Python version by running:

.. code-block:: bash

    $ python --version

and checking that it reports at least "Python 2.6".

Building Python on CentOS 5.4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CentOS 5.4, unfortunately, does not provide a modern version of Python. The
recommended solution is to install one into a user-owned local directory,
assumed to be ``~/local`` in the instructions that follow.

Download, unpack, build, and install Python 2.6:

.. code-block:: bash

    $ wget http://www.python.org/ftp/python/2.6.6/Python-2.6.6.tar.bz2
    $ tar jxvf Python-2.6.6.tar.bz2
    $ cd Python-2.6.6
    $ ./configure --prefix=$HOME/local/
    $ make
    $ make install

Add ``~/local/bin`` to the beginning of your path:

.. code-block:: bash

    $ export PATH=~/local/bin:$PATH

Note that compiling a full Python system may require additional system
dependencies, e.g., development packages for ncurses and zlib.

Installing Python dependencies
------------------------------

The recommended approach to installing borg and its dependencies is to do so
inside a "virtualenv", a self-contained local Python environment. A copy of
the ``virtualenv`` tool is packaged with borg.

Start by creating a virtualenv in some directory (we will assume
``~/virtualenv``):

.. code-block:: bash

    $ python <path_to_borg>/virtualenv --no-site-packages ~/virtualenv

The "--no-site-packages" flag isolates the virtualenv from Python modules
installed globally. This isolation makes it easier to understand and debug
problems, but you may opt to use globally-installed packages instead, depending
on your system.

Next, install any missing Python packages:

.. code-block:: bash

    $ pip install plac
    $ pip install nose
    $ pip install cython
    $ pip install numpy
    $ pip install scipy
    $ pip install scikits.learn

.. Final calibration
.. -----------------

.. Borg requires simple calibration to its local execution environment. Run:::

    .. $ python <path_to_borg>/calibrate <path_to_borg>/etc/local_speed

.. The process should take no longer than several minutes. It must be performed on
.. whatever machine that borg will be running on, since its purpose is to measure
.. the speed of local execution.

.. Running borg
.. ------------

.. Finally, make sure that borg is able to solve a basic instance:::

    .. $ python DIR/solve DIR/etc/borg-mix+class.random.pickle DIR/solvers/calibration/unif-k7-r89-v75-c6675-S342542912-045.cnf

.. where DIR is the path to the borg directory.

.. The recommended command line differs for each of our solver entries. For the
.. random category, use:::

    .. $ python DIR/solve DIR/etc/borg-mix+class.random.pickle BENCHNAME -seed RANDOMSEED -budget TIMEOUT -cores NBCORE

