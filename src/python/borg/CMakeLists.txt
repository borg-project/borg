#
# source
#

install(
    FILES
        defaults.py
        solvers.py
        features.py
        portfolios.py
        __init__.py
	DESTINATION lib/borg
    )

#cython_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# descend
#

add_subdirectory(tools)

