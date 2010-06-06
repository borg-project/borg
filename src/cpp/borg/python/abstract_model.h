/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef HEADER_6D300C0C_138A_45F6_8175_8E30D4247DEE
#define HEADER_6D300C0C_138A_45F6_8175_8E30D4247DEE

#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <utexas/python/numpy_array.h>
#include <borg/python/abstract_action.h>

namespace borg
{

//! Model the portfolio domain.
class AbstractModel
{
    public:
        //! Destruct.
        virtual ~AbstractModel();

    public:
        //! Return a map between actions and (normalized) predicted probabilities.
        virtual
        void
        predict
        (
                const utexas::NumpyArrayFY<unsigned int, 2>& history,
                boost::mt19937& generator
        )
        const = 0;

        //! The actions associated with this model.
        virtual std::vector<std::shared_ptr<AbstractAction>> get_actions() const = 0;
};

namespace details
{

//! Permit derivation from AbstractAction in Python.
class AbstractModelWrapper :
    public AbstractModel,
    public boost::python::wrapper<AbstractModel>
{
    public:
        //! Return a map between actions and (normalized) predicted probabilities.
        virtual
        void
        predict
        (
                const utexas::NumpyArrayFY<unsigned int, 2>& history,
                boost::mt19937& generator
        )
        const
        {
            raise runtime_error();
        }

        //! The actions associated with this model.
        virtual std::vector<std::shared_ptr<AbstractAction>> get_actions() const
        {
            using std::vector;
            using std::shared_ptr;
            using boost::python::len;
            using boost::python::object;
            using boost::python::extract;

            vector<shared_ptr<AbstractAction>> actions;
            object python_actions = this->get_override("get_actions")();

            for(long i = 0; i < len(python_outcomes); ++i)
            {
                actions.push_back(extract<shared_ptr<AbstractAction>>(python_actions[i]));
            }

            return actions;
        }
};

}
}

#endif

