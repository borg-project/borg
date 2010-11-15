"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                       import call
    from borg.tools.plot_validation import main

    call(main)

from cargo.log import get_logger

log = get_logger(__name__, default_level = "INFO")

import matplotlib.lines

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

def get_mean_scores(session, solver_name, group, model_type = None):
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
        .query(
            VRR.components,
            func.avg(VRR.score),
            )                                  \
        .filter(
            and_(
                VRR.solver_name == solver_name,
                VRR.group       == group,
                VRR.model_type  == model_type,
                ),
            )                                   \
        .group_by(VRR.components)               \
        .order_by(VRR.components)               \
        .all()

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

    # prepare series properties
    from cargo.plot import get_color_list

    colors = get_color_list(4)

    # set up the plot
    import pylab

    pylab.rcParams.update({'figure.figsize': (4.75, 3.0)})
    pylab.axes(pylab.axes([0.12, 0.15, 0.85, 0.80]))

    # plot the series
    def plot_mean_scores(label, n, solver_name, model_type = None):
        """
        Plot a specific series.
        """

        xy_values = get_mean_scores(session, solver_name, "sat/competition_2009/random", model_type)
        (x_values, y_values) = zip(*xy_values)

        if x_values == (None,):
            pylab.plot([0, 65], y_values * 2, label = label, marker = markers[n], c = colors[n])
        else:
            pylab.plot(x_values, y_values, label = label, marker = markers[n], c = colors[n])

    plot_mean_scores("DCM", 0, "portfolio", "dcm")
    plot_mean_scores("Multinomial", 1, "portfolio", "multinomial")
    plot_mean_scores("Best Single", 2, "sat/2009/TNM")
    plot_mean_scores("SATzilla", 3, "sat/2009/SATzilla2009_R")

    # final plot formatting
    pylab.xlim(0, 65)
#     pylab.ylim(200, 360)
    pylab.xticks(range(0, 70, 5))
    pylab.xlabel("Number of latent classes $(K)$")
    pylab.ylabel("SAT instances solved")

    legend = pylab.legend(loc = "lower right", ncol = 1)

    legend.get_frame().set_alpha(0.75)
    legend.get_frame().set_edgecolor("none")

    # at least, plot the plot
    pylab.draw()
    pylab.savefig(output_path)

def main(output):
    """
    Run the script.
    """

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

