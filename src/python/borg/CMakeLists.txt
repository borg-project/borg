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
install(
    FILES
        defaults.py
        portfolios.py
        bilevel.py
        opb.py
        expenses.py
        __init__.py
    DESTINATION lib/borg
    )

#
# descend
#

add_subdirectory(tools)
add_subdirectory(solvers)
add_subdirectory(features)
add_subdirectory(test)

