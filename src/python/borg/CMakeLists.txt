#
# build
#

add_cython_module(borg_dimacs dimacs)
add_cython_module(borg_models models)

#
# install
#

install(
    TARGETS
        borg_dimacs
        borg_models
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
        minion.py
        bilevel.py
        parallel.py
        __init__.py
    DESTINATION lib/borg
    )

#
# descend
#

add_subdirectory(tools)

