#
# install
#

install(
    FILES
        __init__.py
        pb.py
        sat.py
    DESTINATION lib/borg/domains
    )

#
# descend
#

add_subdirectory(features)
add_subdirectory(solvers)

