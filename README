BORG
====

ABOUT
-----

Modern heuristic solvers can tackle difficult computational problems, but each
solver performs well only on certain tasks. An algorithm portfolio uses
empirical knowledge---past experience of each solver's behavior---to run the
best solvers on each task.

The borg project includes a practical algorithm portfolio for decision
problems, a set of tools for algorithm portfolio development, and a research
platform for the application of statistical models and decision-theoretic
reasoning to the algorithm portfolio setting.

The project web site is:

http://nn.cs.utexas.edu/pages/research/borg/

Its technical documentation can be found at:

http://borg.readthedocs.org/

REQUIREMENTS
------------

The project also depends on cargo, a general-purpose Python support library:

http://github.com/borg-project/cargo

Its installation procedure is the same as that of borg, below.

COMPILATION
-----------

# Overview

Most of the project is written in Python, but important subsets are written in
Cython and compiled. CMake is used to handle compilation and installation. The
procedure roughly follows:

1. create an out-of-source build directory, eg, `mkdir build`
2. run cmake from the build directory, eg:

`cmake -DCMAKE_INSTALL_PREFIX=<prefix> <path_to_source_tree>`

3. `make`
4. `make install`

# Dependencies

To provide the location of libraries in nonstandard installation prefixes to
cmake, define `CMAKE_PREFIX_PATH`; eg, if required libraries are installed under
`/opt/foo/include` and `/opt/foo/lib`:

`cmake -DCMAKE_PREFIX_PATH=/opt/foo <other_options>`

LICENSE
-------

This software package is provided under the non-copyleft open-source "MIT"
license. The complete legal notice can be found in the included LICENSE file.

