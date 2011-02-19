#
# source
#

install(
	FILES
        __init__.py
		run_solvers.py
		run_validation.py
		get_features.py
	DESTINATION lib/borg/tools
	)

#
# descend
#

add_subdirectory(test)

