#
# source
#

install(
	FILES
        __init__.py
        learn.py
        practice.py
        solve.py
        split.py
	DESTINATION lib/borg/tools/portfolio
	)

#
# descend
#

add_subdirectory(test)

