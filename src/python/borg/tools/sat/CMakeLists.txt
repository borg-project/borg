#
# source
#

install(
	FILES
        __init__.py
        get_tasks.py
        test_solvers.py
	DESTINATION lib/borg/tools/sat
	)

#
# descend
#

add_subdirectory(test)

