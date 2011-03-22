#
# build
#

add_cython_module(borg_domains_pb_features features)
add_cython_module(borg_domains_pb_instance instance)

#
# install
#

install(
    TARGETS
        borg_domains_pb_features
        borg_domains_pb_instance
    DESTINATION lib/borg/domains/pb
    )
install(
    FILES
        __init__.py
        solvers.py
    DESTINATION lib/borg/domains/pb
    )

#
# descend
#

add_subdirectory(test)

