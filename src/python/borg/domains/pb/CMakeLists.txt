#
# build
#

add_cython_module(borg_domains_pb_features features)

#
# install
#

install(
    TARGETS
        borg_domains_pb_features
    DESTINATION lib/borg/domains/pb
    )
install(
    FILES
        __init__.py
        solvers.py
        opb.py
    DESTINATION lib/borg/domains/pb
    )

