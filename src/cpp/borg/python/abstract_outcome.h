/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef HEADER_E6A9E96B_409A_4620_8DDA_73817375A36D
#define HEADER_E6A9E96B_409A_4620_8DDA_73817375A36D

#include <boost/python.hpp>

namespace borg
{

//! An outcome of an action in the world.
class AbstractOutcome
{
    public:
        //! Destruct.
        virtual ~AbstractOutcome();

    public:
        //! The utility of this outcome.
        virtual double get_utility() const = 0;
};

namespace details
{

//! Permit derivation from AbstractOutcome in Python.
class AbstractOutcomeWrapper :
    public AbstractOutcome,
    public boost::python::wrapper<AbstractOutcome>
{
    public:
        //! The utility of this outcome.
        virtual double get_utility() const
        {
            return this->get_override("get_utility")();
        }
};

}
}

#endif

