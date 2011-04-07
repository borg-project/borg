#
# build
#

add_cython_module(borg_models models)

#
# install
#

install(
    TARGETS
        borg_models
    DESTINATION lib/borg
    )
install(
    FILES
        defaults.py
        portfolios.py
        bilevel.py
        expenses.py
        solver_io.py
        __init__.py
    DESTINATION lib/borg
    )

#
# descend
#

add_subdirectory(tools)
add_subdirectory(domains)
add_subdirectory(visual)
add_subdirectory(test)

