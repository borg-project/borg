#
# build
#

add_cython_module(borg_domains_features_pb pb)

#
# install
#

install(
    TARGETS
        borg_domains_features_pb
    DESTINATION lib/borg/domains/features
    )
install(
    FILES
        __init__.py
        common.py
        sat.py
    DESTINATION lib/borg/domains/features
    )

