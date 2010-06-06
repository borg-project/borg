#
# build
#

add_cython_module(borg_portfolio_base    _base)
add_cython_module(borg_portfolio_bellman _bellman)
add_cython_module(borg_portfolio_models  _models)

#
# install
#

install(
    TARGETS
        borg_portfolio_base
        borg_portfolio_bellman
    DESTINATION borg/portfolio
    )
install(
	FILES
        __init__.py
        bellman.py
        decision_world.py
        models.py
        planners.py
        strategies.py
        world.py
	DESTINATION borg/portfolio
	)

#
# descend
#

add_subdirectory(test)

