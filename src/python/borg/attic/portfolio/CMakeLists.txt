#
# build
#

add_cython_module(borg_portfolio_base   _base)
add_cython_module(borg_portfolio_models _models)

target_link_libraries(borg_portfolio_models ${GSL_LIBRARIES})

#
# install
#

install(
    TARGETS
        borg_portfolio_base
        borg_portfolio_models
    DESTINATION lib/borg/portfolio
    )
install(
    FILES
		__init__.py
		bellman.py
		models.py
		planners.py
		world.py
    DESTINATION lib/borg/portfolio
    )

#
# descend
#

add_subdirectory(test)
add_subdirectory(strategies)

