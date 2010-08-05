#
# source
#

install(
	FILES
        __init__.py
        answer.py
        cnf.py
	DESTINATION lib/borg/sat
	)
install(
	PROGRAMS
		features1s
	DESTINATION lib/borg/sat
	)

#
# descend
#

add_subdirectory(test)

