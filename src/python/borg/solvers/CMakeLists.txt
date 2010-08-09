#
# source
#

install(
	FILES
        __init__.py
		analyzers.py
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
	DESTINATION lib/borg/solvers
	)

#
# descend
#

add_subdirectory(test)

