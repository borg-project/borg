"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.plot_validation import main

    raise SystemExit(main())

from cargo.log import get_logger

log = get_logger(__name__, default_level = "INFO")

import matplotlib

markers = [
    "+",
    ",",
    ".",
    "1",
    "2",
    "3",
    "4",
    "<",
    ">",
    "D",
    "H",
    "^",
    "_",
    "d",
    "h",
    "o",
    "p",
    "s",
    "v",
    "x",
    "|",
    matplotlib.lines.TICKUP,
    matplotlib.lines.TICKDOWN,
    matplotlib.lines.TICKLEFT,
    matplotlib.lines.TICKRIGHT,
    matplotlib.lines.CARETLEFT,
    matplotlib.lines.CARETRIGHT,
    matplotlib.lines.CARETUP,
    matplotlib.lines.CARETDOWN,
    ]

def get_mean_score(session, solver_name, group, components, model_type):
    """
    Get relevant attempt data from a trial.
    """

    from sqlalchemy import (
        and_,
        func,
        )
    from borg.data  import ValidationRunRow as VRR

    return                                      \
        session                                 \
        .query(func.avg(VRR.score))             \
        .filter(
            and_(
                VRR.solver_name == solver_name,
                VRR.group       == group,
                VRR.components  == components,
                VRR.model_type  == model_type,
                ),
            )                                   \
        .one()

def pt_to_in(pt):
    """
    Return the number of inches corresponding to the specified size in pt.
    """

    return pt / 72.27

def set_up_for_beamer():
    """
    Set up plot configuration.
    """

    params = {
#         "backend":         "ps",
        "font.family":     "sans-serif",
#         "font.weight":     "normal",
        "axes.labelsize":  10,
        "text.fontsize":   10,
        "text.titlesize":  10,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 0.5,
        "patch.linewidth": 0.5,
        "axes.linewidth":  1.0,
        "text.usetex":     True,
        "text.latex.preamble": [open(os.path.join(os.path.dirname(__file__), "beamer_sfmath.tex")).read()],
#         "ps.usedistiller": "xpdf",
    }

    pylab.rcParams.update(params)

def plot_validation(session, output_path):
    """
    Plot the specified trial.
    """

    x_values = range(1, 65)
#     y_values = [
#         get_mean_score(session, "portfolio", "sat/competition_2009/random", k, "dcm")
#         for k in x_values
#         ]
    y_values = [x**2 for x in x_values]

    # plot the series
    import pylab

    pylab.rcParams.update({'figure.figsize': (4.75, 3.0)})

    from cargo.plot import get_color_list

    colors = get_color_list(1)

    pylab.axes(pylab.axes([0.12, 0.15, 0.85, 0.80]))

    for i in [0]:
        pylab.plot(x_values, y_values, label = "H + DCM", marker = "+", c = colors[i, :])

    pylab.xlim(0, 65)
    pylab.ylim(200, 360)
    pylab.xticks(range(0, 70, 5))
    pylab.xlabel("Number of mixture components $(K)$")
    pylab.ylabel("SAT instances solved")

    legend = pylab.legend(loc = "lower right", ncol = 2)

    legend.get_frame().set_alpha(0.75)
    legend.get_frame().set_edgecolor("none")

    pylab.draw()
    pylab.savefig(output_path)

def main():
    """
    Run the script.
    """

    # get command line arguments
    import borg.data

    from cargo.flags import parse_given

    (output,) = parse_given(usage = "%prog [options] <output>")

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "DETAIL")

    # connect to the database and go
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        with ResearchSession() as session:
            plot_validation(session, output)

