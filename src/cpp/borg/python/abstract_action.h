/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef F1B03557_6EBB_45D6_B894_5EB55C0D3541
#define F1B03557_6EBB_45D6_B894_5EB55C0D3541

#include <string>
#include <vector>
#include <memory>
#include <boost/foreach.hpp>
#include <borg/python/abstract_outcome.h>

namespace borg
{

//! An action in the world.
class AbstractAction
{
    public:
        virtual ~AbstractAction();

    public:
        //! A human-readable description of this action.
        virtual std::string get_description() const = 0;

        //! The typical cost of taking this action.
        virtual double get_cost() const = 0;

        //! The possible outcomes of this action.
        virtual std::vector<std::shared_ptr<AbstractOutcome>> get_outcomes() const = 0;
};

namespace details
{

//! Permit derivation from AbstractAction in Python.
class AbstractActionWrapper :
    public AbstractAction,
    public boost::python::wrapper<AbstractAction>
{
    public:
        //! A human-readable description of this action.
        virtual std::string get_description() const
        {
            return this->get_override("get_description")();
        }

        //! The typical cost of taking this action.
        virtual double get_cost() const
        {
            return this->get_override("get_cost")();
        }

        //! The possible outcomes of this action.
        virtual std::vector<std::shared_ptr<AbstractOutcome>> get_outcomes() const
        {
            using std::vector;
            using std::shared_ptr;
            using boost::python::len;
            using boost::python::object;
            using boost::python::extract;

            vector<shared_ptr<AbstractOutcome>> outcomes;
            object python_outcomes = this->get_override("get_outcomes")();

            for(long i = 0; i < len(python_outcomes); ++i)
            {
                auto python_outcome = python_outcomes[i];

                outcomes.push_back(extract<shared_ptr<AbstractOutcome>>(python_outcome));
            }

            return outcomes;
        }
};

}
}

#endif

