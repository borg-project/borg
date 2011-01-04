#
# source
#

install(
	FILES
        __init__.py
		analyze.py
		bench_bellman.py
        calibrate.py
        plot_performance.py
        preprocess_tasks.py
        print_run.py
        run_solvers.py
        run_validation.py
        plot_validation.py
        solve.py
        build_solver.py
        get_tasks.py
        check_solvers.py
        sanitize_cnfs.py
	DESTINATION lib/borg/tools
	)

#
# descend
#

add_subdirectory(test)

