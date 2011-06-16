Installing borg
===============

Outline of setup process
------------------------

Setting up borg involves four major steps, described in detail in the other
sections below:

#. installing any missing system-level dependencies;
#. installing a local version of Python 2.6;
#. installing the required Python packages; and
#. running a final calibration script.

This process will install local copies of:

* Python 2.6.6
* numpy 1.5.1
* scipy 0.9.0
* scikits.learn 0.6
* plac 0.8.0

Installing system dependencies
------------------------------

The system packages (CentOS repository names)

* gcc
* gcc-gfortran
* gcc-c++
* blas-devel
* lapack-devel
* ncurses-devel

are needed to build Python and the other package dependencies.

Installing Python 2.6
---------------------

This version of borg has been pre-built for CentOS 5.4, but does not include
its Python dependencies. CentOS, unfortunately, does not provide a modern
version of Python. The recommended solution is to install one into a user-owned
local directory, assumed to be "~/local" in the instructions that follow.

Download, unpack, build, and install Python 2.6:

    wget http://www.python.org/ftp/python/2.6.6/Python-2.6.6.tar.bz2
    tar jxvf Python-2.6.6.tar.bz2
    cd Python-2.6.6
    ./configure --prefix=$HOME/local/
    make
    make install

Add ~/local/bin to the beginning of your path:

``$ export PATH=~/local/bin:$PATH``

Verify that you're using that python version by running

``$ python --version``

and checking that it reports "Python 2.6.6".

Installing other dependencies
-----------------------------

Back up a directory and download the setuptools egg:

$ cd ..
$ wget http://pypi.python.org/packages/2.6/s/setuptools/setuptools-0.6c11-py2.6.egg

It will install itself when run as a shell script:

$ sh setuptools-0.6c11-py2.6.egg

Now we build and install the final set of dependencies (pay no attention to
complaints about numpy.distutils in easy_install output):

$ easy_install-2.6 numpy
$ easy_install-2.6 scipy
$ easy_install-2.6 scikits.learn==0.6
$ easy_install-2.6 plac

Final calibration
-----------------

Borg requires simple calibration to its local execution environment. Run:

``$ python <path_to_borg>/calibrate <path_to_borg>/etc/local_speed``

The process should take no longer than several minutes. It must be performed
*on a competition cluster node*, since it is (crudely) measuring the speed of
local execution.

Running borg
------------

Finally, make sure that borg is able to solve a basic instance:

``$ python DIR/solve DIR/etc/borg-mix+class.random.pickle DIR/solvers/calibration/unif-k7-r89-v75-c6675-S342542912-045.cnf``

where DIR is the path to the borg directory.

The recommended command line differs for each of our solver entries. For the
random category, use:

``$ python DIR/solve DIR/etc/borg-mix+class.random.pickle BENCHNAME -seed RANDOMSEED -budget TIMEOUT -cores NBCORE``

For the industrial category, use:

``$ python DIR/solve DIR/etc/borg-mix+class.industrial.pickle BENCHNAME -seed RANDOMSEED -budget TIMEOUT -cores NBCORE``

The solver will adjust for sequential or parallel operation depending on the
value of NBCORE.

