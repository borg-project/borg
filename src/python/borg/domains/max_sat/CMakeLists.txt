#
# build
#

add_cython_module(borg_domains_max_sat_features features)
add_cython_module(borg_domains_max_sat_instance instance)

#
# install
#

install(
    TARGETS
        borg_domains_max_sat_features
        borg_domains_max_sat_instance
    DESTINATION lib/borg/domains/max_sat
    )
install(
    FILES
        __init__.py
        solvers.py
    DESTINATION lib/borg/domains/max_sat
    )

##
## descend
##

add_subdirectory(test)

