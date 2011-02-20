#
# install
#

install(
    PROGRAMS
        features1s
    DESTINATION lib/borg/_files
    )
install(
    FILES
        beamer_sfmath.tex
    DESTINATION lib/borg/_files
    )

#
# descend
#

add_subdirectory(for_tests)

