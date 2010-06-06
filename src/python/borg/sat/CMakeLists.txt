#
# source
#

install(
	FILES
        __init__.py
        answer.py
        cnf.py
	DESTINATION borg/sat
	)

#
# descend
#

add_subdirectory(test)

