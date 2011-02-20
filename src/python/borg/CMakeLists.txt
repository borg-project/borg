#
# source
#

install(
    FILES
        defaults.py
        portfolios.py
        solvers.py
        __init__.py
    )

#cython_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# descend
#

add_subdirectory(tools)

