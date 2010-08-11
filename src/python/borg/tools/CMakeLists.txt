#
# source
#

install(
	FILES
        __init__.py
		analyze.py
		bench_bellman.py
        calibrate.py
        delete_trial.py
        generate_ddl.py
        plot_performance.py
        preprocess_tasks.py
        print_run.py
        run_solvers.py
        run_validation.py
        plot_validation.py
	DESTINATION lib/borg/tools
	)

#
# descend
#

add_subdirectory(portfolio)
add_subdirectory(sat)
add_subdirectory(test)

