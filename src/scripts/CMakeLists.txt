#
# source
#

install(
	FILES
        __init__.py
        cache.py
        data.py
        rowed.py
        tasks.py
	DESTINATION borg
	)

#
# descend
#

add_subdirectory(pb)
add_subdirectory(portfolio)
add_subdirectory(sat)
add_subdirectory(solvers)
add_subdirectory(test)
add_subdirectory(tools)

