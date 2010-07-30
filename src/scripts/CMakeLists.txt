install(
    PROGRAMS
        run_borg.py
    DESTINATION bin/borg_
    )
install(
    CODE
    "
    execute_process(
        COMMAND
            ${CMAKE_COMMAND}
            -E
            create_symlink
            ${CMAKE_INSTALL_PREFIX}/bin/borg_/run_borg.py
            ${CMAKE_INSTALL_PREFIX}/bin/borg
        )
    "
    )

