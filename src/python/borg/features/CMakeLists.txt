#
# build
#

add_cython_module(borg_features_pb pb)

#
# source
#

install(
    TARGETS
        borg_features_pb
    DESTINATION lib/borg/features
    )
install(
    FILES
        __init__.py
        common.py
        sat.py
    DESTINATION lib/borg/features
    )

