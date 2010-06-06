/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include <boost/python.hpp>
#include <utexas/python/get_shared_ptr.h>
#include <borg/python/abstract_model.h>
#include <borg/python/abstract_action.h>
#include <borg/python/abstract_outcome.h>

using boost::python::list;
using borg::AbstractAction;


//
// SUPPORT
//

//! Wrap AbstractAction::get_outcomes().
list abstract_action_get_outcomes(const AbstractAction& action)
{
    list outcomes;

    BOOST_FOREACH(auto outcome, action.get_outcomes())
    {
        outcomes.append(outcome);
    }

    return outcomes;
}


//
// MODULE
//

//! Module initialization.
BOOST_PYTHON_MODULE(_borg)
{
    using std::shared_ptr;
    using boost::noncopyable;
    using boost::python::class_;
    using boost::python::no_init;
    using boost::python::register_ptr_to_python;

    // bind AbstractOutcome
    using borg::AbstractOutcome;
    using borg::details::AbstractOutcomeWrapper;

    register_ptr_to_python<shared_ptr<AbstractOutcome>>();

    class_<AbstractOutcomeWrapper, noncopyable>("AbstractOutcome", no_init)
        // properties
        .add_property("utility", &AbstractOutcome::get_utility);

    // bind AbstractAction
    using borg::details::AbstractActionWrapper;

    register_ptr_to_python<shared_ptr<AbstractAction>>();

    class_<AbstractActionWrapper, noncopyable>("AbstractAction", no_init)
        // properties
        .add_property("description", &AbstractAction::get_description)
        .add_property("cost", &AbstractAction::get_cost)
        .add_property("outcomes", &abstract_action_get_outcomes);

    // bind AbstractModel
    using borg::details::AbstractModelWrapper;

    register_ptr_to_python<shared_ptr<AbstractModel>>();

    class_<AbstractModelWrapper, noncopyable>("AbstractModel", no_init)
        // properties
        .add_property("description", &AbstractAction::get_description)
        .add_property("cost", &AbstractAction::get_cost)
        .add_property("actions", &abstract_model_get_actions);
}

