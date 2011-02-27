#
# build
#

add_cython_module(borg_dimacs dimacs)

#
# install
#

install(
    TARGETS
        borg_dimacs
    DESTINATION lib/borg
    )

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

#
# descend
#

add_subdirectory(tools)

