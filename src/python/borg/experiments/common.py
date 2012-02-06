"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import borg

def train_model(model_name, training, bins = 10):
    if model_name == "mul_alpha=0.1":
        estimator = borg.models.MulEstimator(alpha = 0.1)
    elif model_name == "mul-dir":
        estimator = borg.models.MulDirEstimator()
    elif model_name == "mul-dirmix":
        estimator = borg.models.MulDirMixEstimator(samples_per = 8)
    elif model_name == "mul-dirmatmix":
        estimator = borg.models.MulDirMatMixEstimator()
    else:
        raise Exception("unrecognized model name \"{0}\"".format(model_name))

    return estimator(training, bins, training)

