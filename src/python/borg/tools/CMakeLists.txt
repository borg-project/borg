#
# source
#

install(
	FILES
        __init__.py
        calibrate.py
        delete_trial.py
        generate_ddl.py
        plot_performance.py
        preprocess_tasks.py
        print_run.py
        run_solvers.py
	DESTINATION borg/tools
	)

#
# descend
#

add_subdirectory(portfolio)
add_subdirectory(sat)

