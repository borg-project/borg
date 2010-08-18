#
# install
#

install(
    FILES
        foo_solver.py
        s57-100.cnf
        s57-100.cnf.xz
    DESTINATION lib/borg/_files/for_tests
    )

#
# descend
#

add_subdirectory(reset_defaults)

