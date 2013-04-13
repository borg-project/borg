"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import signal
import numpy
import scipy.sparse
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

named = {
    #"weighted" : lambda opb: 1.0 if opb.nonlinear is None else 0.0,
    #"optimization" : lambda opb: 0.0 if opb.objective is None else 1.0,
    "variables": lambda t: t.N,
    "constraints": lambda t: t.M,
    "ratio": lambda t: t.M / float(t.N),
    "ratio_reciprocal": lambda t: float(t.N) / t.M,
    }

def feature(method):
    assert method.__name__.startswith("compute_")

    named[method.__name__[8:]] = method

    return method

@feature
def compute_weights_min(instance):
    return numpy.min(instance.weights)

@feature
def compute_weights_max(instance):
    return numpy.max(instance.weights)

@feature
def compute_weights_mean(instance):
    return numpy.mean(instance.weights)

@feature
def compute_weights_std(instance):
    return numpy.std(instance.weights)

def build_balance_ratios(instance):
    """Gather the positive/negative balance counts."""

    if not hasattr(instance, "variable_balance_ratios"):
        constraints_CV = instance.constraints
        (C, V) = constraints_CV.shape

        positives_CV = \
            scipy.sparse.csr_matrix(
                ((constraints_CV.data + 1) / 2, constraints_CV.indices, constraints_CV.indptr),
                shape = (C, V),
                dtype = int,
                )
        negatives_CV = \
            scipy.sparse.csr_matrix(
                ((constraints_CV.data - 1) / -2, constraints_CV.indices, constraints_CV.indptr),
                shape = (C, V),
                dtype = int,
                )

        appearances_V = numpy.asarray(positives_CV.sum(axis = 0) + negatives_CV.sum(axis = 0))
        appearances_V[appearances_V == 0] = 1

        instance.variable_balance_ratios = positives_CV.sum(axis = 0) / appearances_V
        instance.constraint_balance_ratios = negatives_CV.sum(axis = 1) / (positives_CV.sum(axis = 1) + negatives_CV.sum(axis = 1))

@feature
def compute_constraint_balance_ratio_mean(instance):
    build_balance_ratios(instance)

    return numpy.mean(instance.constraint_balance_ratios)

@feature
def compute_constraint_balance_ratio_std(instance):
    build_balance_ratios(instance)

    return numpy.std(instance.constraint_balance_ratios)

@feature
def compute_variable_balance_ratio_mean(instance):
    build_balance_ratios(instance)

    return numpy.mean(instance.variable_balance_ratios)

@feature
def compute_variable_balance_ratio_std(instance):
    build_balance_ratios(instance)

    return numpy.std(instance.variable_balance_ratios)

def build_node_degrees(instance):
    """Gather the CG, VG, and VCG node degrees."""

    if not hasattr(instance, "vcg_degrees_V"):
        V = instance.N
        C = instance.M

        adjacency_csr_CV = \
            scipy.sparse.csr_matrix(
                (numpy.ones(len(instance.constraints.data), int), instance.constraints.indices, instance.constraints.indptr),
                shape = (C, V),
                )
        adjacency_csr_VC = adjacency_csr_CV.T.tocsr()

        instance.vcg_degrees_V = numpy.asarray(adjacency_csr_VC.sum(axis = 1))[:, 0]
        instance.vcg_degrees_C = numpy.asarray(adjacency_csr_VC.sum(axis = 0))[0, :]

        instance.vg_degrees_V = numpy.zeros(V, int)
        instance.cg_degrees_C = numpy.zeros(C, int)

        for v in xrange(V):
            i = adjacency_csr_VC.indptr[v]
            j = adjacency_csr_VC.indptr[v + 1]

            for k in xrange(i, j):
                c = adjacency_csr_VC.indices[k]

                instance.vg_degrees_V[v] += instance.vcg_degrees_C[c]

            instance.vg_degrees_V[v] -= j - i

        for c in xrange(C):
            i = adjacency_csr_CV.indptr[c]
            j = adjacency_csr_CV.indptr[c + 1]

            for k in xrange(i, j):
                v = adjacency_csr_CV.indices[k]

                instance.cg_degrees_C[c] += instance.vcg_degrees_V[v]

            instance.cg_degrees_C[c] -= j - i

def graph_feature(method):
    def wrapper(instance):
        build_node_degrees(instance)

        return method(instance)

    assert method.__name__.startswith("compute_")

    named[method.__name__[8:]] = wrapper

    return wrapper

@graph_feature
def compute_cg_cnode_degree_mean(instance):
    return numpy.mean(instance.cg_degrees_C)

@graph_feature
def compute_cg_cnode_degree_std(instance):
    return numpy.std(instance.cg_degrees_C)

@graph_feature
def compute_vcg_cnode_degree_std(instance):
    return numpy.std(instance.vcg_degrees_C)

@graph_feature
def compute_vg_node_degree_mean(instance):
    return numpy.mean(instance.vg_degrees_V)

@graph_feature
def compute_vg_node_degree_std(instance):
    return numpy.std(instance.vg_degrees_V)

@graph_feature
def compute_vcg_vnode_degree_mean(instance):
    return numpy.mean(instance.vcg_degrees_V)

@graph_feature
def compute_vcg_vnode_degree_std(instance):
    return numpy.std(instance.vcg_degrees_V)

@graph_feature
def compute_vcg_cnode_degree_mean(instance):
    return numpy.mean(instance.vcg_degrees_C)

def get_features_for(instance_path):
    """Compute all features of a PB instance."""

    with borg.accounting() as accountant:
        with open(instance_path) as task_file:
            instance = borg.domains.max_sat.instance.parse_max_sat_file(task_file)

    logger.info("parsing %s in %.2f s", instance_path, accountant.total.cpu_seconds)

    with borg.accounting() as accountant:
        computed = dict((k, v(instance)) for (k, v) in named.items())

    logger.info("collected features for %s in %.2f s", instance_path, accountant.total.cpu_seconds)

    computed["cpu_cost"] = accountant.total.cpu_seconds

    return (computed.keys(), computed.values())

