#
# source
#

install(
	FILES
        __init__.py
        cache_data.py
        learn.py
        practice.py
        solve.py
        split.py
	DESTINATION borg/tools/portfolio
	)

#
# descend
#

add_subdirectory(test)

