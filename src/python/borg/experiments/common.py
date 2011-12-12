"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import borg

def train_model(model_name, training, bins = 10, samples_per_chain = 1, chains = 1):
    if model_name == "mul_alpha=0.1":
        sampler = borg.models.MulSampler(alpha = 0.1)
    elif model_name == "mul-dir":
        sampler = borg.models.MulDirSampler()
    elif model_name == "mul-dirmix":
        sampler = borg.models.MulDirMixSampler()
    elif model_name == "mul-dirmatmix":
        sampler = borg.models.MulDirMatMixSampler()
    else:
        raise Exception("unrecognized model name \"{0}\"".format(model_name))

    model = \
        borg.models.mixture_from_posterior(
            sampler,
            training.solver_names,
            training,
            bins = bins,
            samples_per_chain = samples_per_chain,
            chains = chains,
            )

    return model

