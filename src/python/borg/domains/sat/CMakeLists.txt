#
# build
#

add_cython_module(borg_domains_sat_dimacs dimacs)

#
# install
#

install(
    TARGETS
        borg_domains_sat_dimacs
    DESTINATION lib/borg/domains/pb
    )
install(
    FILES
        __init__.py
        solvers.py
        features.py
    DESTINATION lib/borg/domains/sat
    )

