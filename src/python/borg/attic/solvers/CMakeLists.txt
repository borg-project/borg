#
# source
#

install(
	FILES
        __init__.py
        attempts.py
        base.py
        competition.py
        lookup.py
        portfolio.py
        preprocessing.py
        recycling.py
        sanitizing.py
        satelite.py
        uncompressing.py
        zefram.py
	DESTINATION lib/borg/solvers
	)

#
# descend
#

add_subdirectory(test)

