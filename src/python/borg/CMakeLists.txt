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
        features.py
        portfolios.py
        bilevel.py
        __init__.py
    DESTINATION lib/borg
    )

#
# descend
#

add_subdirectory(tools)
add_subdirectory(solvers)

