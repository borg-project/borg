/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef 32EBDD2C_CB21_4BF0_80B3_2C57612598ED
#define 32EBDD2C_CB21_4BF0_80B3_2C57612598ED

namespace borg
{

//! Make random predictions.
class RandomModel
{
    public:
        //! Initialize.
        RandomModel(actions)
        {
            self._actions = actions
        }

    public:
        //! Return the predicted probability of each outcome, given history.
        def predict(self, history, random):
        {
            return dict((a, self._random_prediction(a, random)) for a in self._actions)
        }

        //! The actions associated with this model.
        def actions(self):
        {
            return self._actions;
        }

        //! Build a model as requested.
        def build(request, trainer)
        {
            return RandomModel(trainer.build_actions(request["actions"]));
        }

    private:
        //! Return a random prediction.
        def _random_prediction(self, action, random):
        {
            predictions  = random.rand(len(action.outcomes))
            predictions /= numpy.sum(predictions)

            return predictions
        }
};

}

#endif

