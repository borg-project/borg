#
# install
#

install(
    FILES
        __init__.py
    DESTINATION lib/borg/domains
    )

#
# descend
#

add_subdirectory(pb)
add_subdirectory(sat)

