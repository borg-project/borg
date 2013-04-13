Contributions and development
*****************************

If you are interested in modifying, contributing to, or basing your own project
on top of ``borg``, this chapter provides a bit of documentation on doing so.

Obtaining the project source code
=================================

The suggested way to download the project source is to use `git
<http://git-scm.com/>`_, which will simplify acquiring future updates and
making local changes. Assuming that git is installed, clone the two
relevant repositories from `github <https://github.com/>`_:

.. code-block:: bash

    $ git clone git@github.com:borg-project/cargo.git
    $ git clone git@github.com:borg-project/borg.git

If git is not installed and cannot be installed, tarball snapshots of the
source trees can be downloaded from github's web interface; see the borg and
cargo repositories under the `borg project github page
<https://github.com/borg-project>`_.

Building documentation
======================

The borg documentation can be generated in various formats from its
reStructuredText source. The `sphinx` tool is used to drive this process.
It can be installed as usual via `pip`, with:

.. code-block:: bash

    $ pip install sphinx

.. code-block:: bash

    $ cd borg/docs
    $ make html

