"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_get_named_solvers():
    """
    Test loading of named solvers.
    """

    solvers_json = """
{
    "solvers" : {
        "foo" : {
			"type"    : "sat_competition",
            "command" : ["HERE/foo", "BENCHNAME", "RANDOMSEED"]
            }
        },
    "includes" : ["bar/solvers.json"]
    }
"""
    bar_solvers_json = """
{
    "solvers" : {
        "bar" : {
			"type"    : "satelite",
            "command" : ["HERE/foo"]
            }
        }
    }
"""

    # test loading
    from cargo.io import mkdtemp_scoped

    with mkdtemp_scoped() as sandbox_path:
        # write the configuration files
        from os      import mkdir
        from os.path import join

        solvers_json_path = join(sandbox_path, "solvers.json")

        with open(solvers_json_path, "w") as file:
            file.write(solvers_json)

        bar_path              = join(sandbox_path, "bar")
        bar_solvers_json_path = join(bar_path, "solvers.json")

        mkdir(bar_path)

        with open(bar_solvers_json_path, "w") as file:
            file.write(bar_solvers_json)

        # load the solvers
        from borg.solvers import get_named_solvers

        solvers = \
            get_named_solvers(
                flags = {
                    "solvers_file"      : [solvers_json_path],
                    "use_recycled_runs" : False,
                    },
                )

        # assert expectations
        from nose.tools   import (
            assert_true,
            assert_equal,
            )
        from borg.solvers import (
            StandardSolver,
            SatELitePreprocessor,
            )

        assert_equal(len(solvers), 2)
        assert_true(isinstance(solvers["foo"], StandardSolver))
        assert_true(isinstance(solvers["bar"], SatELitePreprocessor))

