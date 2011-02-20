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

#
# descend
#

add_subdirectory(test)

