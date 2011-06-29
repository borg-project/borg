"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import cargo
import borg

logger = cargo.get_logger(__name__)

def basic_command(relative):
    """Prepare a basic competition solver command."""

    return ["{{root}}/{0}".format(relative), "{task}", "{seed}"]

#commands = {
    ## complete
    #"precosat-570": ["{root}/precosat-570-239dbbe-100801/precosat", "--seed={seed}", "{task}"],
    #"lingeling-276": ["{root}/lingeling-276-6264d55-100731/lingeling", "--seed={seed}", "{task}"],
    #"cryptominisat-2.9.0": ["{root}/cryptominisat-2.9.0/cryptominisat-2.9.0Linux64", "--randomize={seed}", "{task}"],
    #"glucosER": ["{root}/glucosER/glucoser_static", "{task}"],
    #"glucose": ["{root}/glucose/glucose_static", "{task}"],
    #"SApperloT": ["{root}/SApperloT/SApperloT-base", "-seed={seed}", "{task}"],
    #"march_hi": basic_command("march_hi/march_hi"),
    #"kcnfs-2006": ["{root}/kcnfs-2006/kcnfs-2006", "{task}"],
    ## incomplete
    #"TNM": basic_command("TNM/TNM"),
    #"gnovelty+2": basic_command("gnovelty+2/gnovelty+2"),
    #"hybridGM3": basic_command("hybridGM3/hybridGM3"),
    #"adaptg2wsat2009++": basic_command("adaptg2wsat2009++/adaptg2wsat2009++"),
    #"iPAWS": basic_command("iPAWS/iPAWS"),
    #"FH": basic_command("FH/FH"),
    #"NCVWr": basic_command("NCVWr/NCVWr"),
    #}

def parse_sat_output(stdout):
    """Parse a solver's standard competition-format output."""

    match = re.search(r"^s +(.+)$", stdout, re.M)

    if match:
        (answer_type,) = map(str.upper, match.groups())

        if answer_type == "SATISFIABLE":
            answer = []

            for line in re.findall(r"^v ([ \-0-9]*)$", stdout, re.M):
                answer.extend(map(int, line.split()))

            if answer[-1] == 0:
                return answer[:-1]
        elif answer_type == "UNSATISFIABLE":
            return False

    return None

class SAT_SolverFactory(object):
    """Construct a basic competition solver callable."""

    def __init__(self, root, command):
        """Initialize."""

        self._root = root
        self._command = command

    def __call__(self, task, stm_queue = None, solver_id = None):
        return \
            borg.solver_io.RunningSolver(
                parse_sat_output,
                self._command,
                self._root(),
                task.path,
                stm_queue = stm_queue,
                solver_id = solver_id,
                )

