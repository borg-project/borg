Installing borg
***************

.. _PyPI: https://pypi.python.org/pypi

The installation process for borg is similar to that for other Python-based
projects. Its installation involves three major steps, described in detail
below:

#. installing any missing system-level dependencies;
#. installing virtualenv and creating an installation environment; and
#. installing borg and its required Python dependencies.

These instructions assume a ``bash`` shell on a Linux system.

Installing system dependencies
==============================

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
-----------------------

Make sure that you're using a recent Python version by running:

.. code-block:: bash

    $ python --version

and checking that it reports at least "Python 2.6". Users on ancient platforms
will likely need to install a local version of a more recent Python, as in the
instructions for CentOS below.

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

Creating an installation environment
====================================

The recommended approach to installing borg and its dependencies is to do so
inside a "virtualenv", a self-contained local Python environment constructed
with the `virtualenv tool <http://www.virtualenv.org/>`_.

Obtaining virtualenv
--------------------

The virtualenv tool may already be installed (try running "virtualenv" in your
shell). If not, you may be able to install it using the system package manager.
If you are using Ubuntu, for example, install it by performing:

.. code-block:: bash

    $ sudo apt-get install python-virtualenv

If your system package manager does not include it, or you do not have system
root access, you will need to download and use a local copy according to the
`instructions <http://www.virtualenv.org/en/latest/#installation>`_ in the
virtualenv documentation.

Creating an environment
-----------------------

Start by creating a virtual environment ("virtualenv") in some directory; we
will assume ``~/borg-venv``:

.. code-block:: bash

    $ virtualenv --no-site-packages ~/borg-venv

The ``--no-site-packages`` flag isolates the virtualenv from Python packages
installed globally.

Using the environment
---------------------

Next, "activate" the virtualenv to use its Python installation in the current
shell session:

.. code-block:: bash

    $ source ~/borg-venv/bin/activate

Running ``python`` with this environment activated will use the local
interpreter ``~/borg-venv/bin/python``.

.. note::

    The rest of the documentation assumes that you are operating with this
    environment activated.

Leaving the environment
-----------------------

The virtualenv can be later deactivated with:

.. code-block:: bash

    $ deactivate

Installing borg
===============

We can now install borg and its dependencies into this environment.

Installing the numpy dependency
-------------------------------

Due to limitations in Python packaging, the numpy package must be installed
first. Use

.. code-block:: bash

    $ pip install numpy

to download, compile, and install numpy in the local environment. This may
take a few minutes.

Installing borg and other dependencies
--------------------------------------

You should now be able to run

.. code-block:: bash

    $ pip install borg

to download and install the latest release of borg from `PyPI`_, as well its
dependencies.

.. note::

    Some of the borg dependencies, especially ``cython``, ``numpy``, ``scipy``,
    and ``scikit-learn``, are complex libraries that may take ten minutes or more
    to install from source using ``pip``.

