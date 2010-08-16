#
# source
#

install(
	FILES
        __init__.py
		analyze.py
		bench_bellman.py
        cache_data.py
        calibrate.py
        delete_trial.py
        generate_ddl.py
        plot_performance.py
        preprocess_tasks.py
        print_run.py
        run_solvers.py
        run_validation.py
        plot_validation.py
        solve.py
        build_solver.py
	DESTINATION lib/borg/tools
	)

#
# descend
#

add_subdirectory(sat)
add_subdirectory(test)

