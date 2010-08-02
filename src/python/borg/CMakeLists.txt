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
    DESTINATION lib/borg
    )

#cython_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# descend
#

add_subdirectory(pb)
add_subdirectory(portfolio)
add_subdirectory(sat)
add_subdirectory(solvers)
add_subdirectory(test)
add_subdirectory(tools)

