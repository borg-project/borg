"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from borg.tools.delete_trial import main

    raise SystemExit(main())

from cargo.log import get_logger

log = get_logger(__name__, default_level = "INFO")

def main():
    """
    Run the script.
    """

    # get command line arguments
    import borg.data

    from cargo.flags import parse_given

    trial_uuids = parse_given(usage = "%prog [options]")

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "DETAIL")

    # connect to the database
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        from cargo.sql.alchemy import make_session
        from borg.data         import research_connect

        ResearchSession = make_session(bind = research_connect())

        # delete every trial
        with ResearchSession() as session:
            from borg.data import TrialRow

            for trial_uuid in trial_uuids:
                trial_row = session.query(TrialRow).get(trial_uuid)

                trial_row.delete(session)

                session.commit()

