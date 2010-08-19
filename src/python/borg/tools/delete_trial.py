"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                    import call
    from borg.tools.delete_trial import main

    call(main)

from cargo.log import get_logger

log = get_logger(__name__, default_level = "INFO")

def get_trial_family(session, trial_uuids):
    """
    Get the specified trials and all of their descendants.
    """

    from borg.data import TrialRow

    filter      = TrialRow.parent_uuid.in_(trial_uuids)
    child_uuids = session.query(TrialRow.uuid).filter_by(filter).all()

    return trial_uuids + get_trial_family(session, child_uuids)

def show_trials(session, trial_uuids):
    """
    Print the specified trials.
    """

    from borg.data import TrialRow

    for trial_uuid in trial_uuids:
        trial_row = session.query(TrialRow).get(trial_uuid)

        for attempt in trial_row.attempts:
            task_names = attempt.task.names

            if task_names:
                if len(task_names) > 1:
                    task_name_string = "%s (...)" % task_names[0].name
                else:
                    task_name_string = task_names[0].name
            else:
                task_name_string = str(attempt.task_uuid)

            if attempt.type == "run":
                print attempt.solver_name, attempt.budget, attempt.cost, task_name_string
            else:
                print attempt.type, attempt.budget, attempt.cost, task_name_string

def delete_trials(session, trial_uuids):
    """
    Delete the specified trials.
    """

    from borg.data import TrialRow

    for trial_uuid in trial_uuids:
        session.query(TrialRow).get(trial_uuid).delete(session)
        session.commit()

def main(mode, *trial_uuids):
    """
    Run the script.
    """

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "WARNING")

    # set up the mode functions
    mode_functions = {
        "show"   : show_trials,
        "delete" : delete_trials,
        }

    # connect to the database
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        # perform the action on the trials
        with ResearchSession() as session:
            mode_functions[mode](session, trial_uuids)

