"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import numpy
import scipy.sparse
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

named = {
    "optimization" : lambda opb: 0.0 if opb.objective is None else 1.0,
    "variables": lambda opb: opb.N,
    "constraints": lambda opb: len(opb.constraints),
    "ratio": lambda opb: len(opb.constraints) / float(opb.N),
    "ratio_reciprocal": lambda opb: float(opb.N) / len(opb.constraints),
    }

def feature(method):
    assert method.__name__.startswith("compute_")

    named[method.__name__[8:]] = method

    return method

def build_balance_ratios(opb):
    """Gather the positive/negative balance counts."""

    if not hasattr(opb, "variable_balance_ratios"):
        variable_positives = numpy.zeros(opb.N)
        variable_negatives = numpy.zeros(opb.N)
        constraint_positives = numpy.zeros(len(opb.constraints))
        constraint_negatives = numpy.zeros(len(opb.constraints))

        for (c, (terms, _, _)) in enumerate(opb.constraints):
            for (_, literals) in terms:
                for literal in literals:
                    v = abs(literal) - 1

                    if literal > 0:
                        variable_positives[v] += 1
                        constraint_positives[c] += 1
                    else:
                        variable_negatives[v] += 1
                        constraint_negatives[c] += 1

        opb.variable_balance_ratios = variable_positives / (variable_positives + variable_negatives)
        opb.constraint_balance_ratios = constraint_positives / (constraint_positives + constraint_negatives)

@feature
def compute_constraint_balance_ratio_mean(opb):
    build_balance_ratios(opb)

    return numpy.mean(opb.constraint_balance_ratios)

@feature
def compute_constraint_balance_ratio_std(opb):
    build_balance_ratios(opb)

    return numpy.std(opb.constraint_balance_ratios)

@feature
def compute_variable_balance_ratio_mean(opb):
    build_balance_ratios(opb)

    return numpy.mean(opb.variable_balance_ratios)

@feature
def compute_variable_balance_ratio_std(opb):
    build_balance_ratios(opb)

    return numpy.std(opb.variable_balance_ratios)

def build_node_degrees(opb):
    """Gather the CG, VG, and VCG node degrees."""

    if not hasattr(opb, "vcg_degrees_V"):
        V = opb.N
        C = len(opb.constraints)

        adjacency_v = []
        adjacency_c = []

        for (c, (terms, _, _)) in enumerate(opb.constraints):
            for (_, literals) in terms:
                for literal in literals:
                    adjacency_v.append(abs(literal) - 1)
                    adjacency_c.append(c)

        adjacency_csr_VC = \
            scipy.sparse.csr_matrix(
                (numpy.ones(len(adjacency_v), int), (adjacency_v, adjacency_c)),
                (V, C),
                )
        adjacency_csr_CV = adjacency_csr_VC.T.tocsr()

        opb.vcg_degrees_V = numpy.asarray(adjacency_csr_VC.sum(axis = 1))[:, 0]
        opb.vcg_degrees_C = numpy.asarray(adjacency_csr_VC.sum(axis = 0))[0, :]

        opb.vg_degrees_V = numpy.zeros(V, int)
        opb.cg_degrees_C = numpy.zeros(C, int)

        for v in xrange(V):
            i = adjacency_csr_VC.indptr[v]
            j = adjacency_csr_VC.indptr[v + 1]

            for k in xrange(i, j):
                c = adjacency_csr_VC.indices[k]

                opb.vg_degrees_V[v] += opb.vcg_degrees_C[c]

            opb.vg_degrees_V[v] -= j - i

        for c in xrange(C):
            i = adjacency_csr_CV.indptr[c]
            j = adjacency_csr_CV.indptr[c + 1]

            for k in xrange(i, j):
                v = adjacency_csr_CV.indices[k]

                opb.cg_degrees_C[c] += opb.vcg_degrees_V[v]

            opb.cg_degrees_C[c] -= j - i

def graph_feature(method):
    def wrapper(opb):
        build_node_degrees(opb)

        return method(opb)

    assert method.__name__.startswith("compute_")

    named[method.__name__[8:]] = wrapper

    return wrapper

@graph_feature
def compute_cg_cnode_degree_mean(opb):
    return numpy.mean(opb.cg_degrees_C)

@graph_feature
def compute_cg_cnode_degree_std(opb):
    return numpy.std(opb.cg_degrees_C)

@graph_feature
def compute_vcg_cnode_degree_std(opb):
    return numpy.std(opb.vcg_degrees_C)

@graph_feature
def compute_vg_node_degree_mean(opb):
    return numpy.mean(opb.vg_degrees_V)

@graph_feature
def compute_vg_node_degree_std(opb):
    return numpy.std(opb.vg_degrees_V)

@graph_feature
def compute_vcg_vnode_degree_mean(opb):
    return numpy.mean(opb.vcg_degrees_V)

@graph_feature
def compute_vcg_vnode_degree_std(opb):
    return numpy.std(opb.vcg_degrees_V)

@graph_feature
def compute_vcg_cnode_degree_mean(opb):
    return numpy.mean(opb.vcg_degrees_C)

def compute_all(instance):
    """Compute all features of a PB instance."""

    with borg.accounting() as accountant:
        computed = dict((k, v(instance)) for (k, v) in named.items())

    cpu_cost = accountant.total.cpu_seconds

    logger.info("feature computation took %.2f CPU seconds", cpu_cost)

    return (["cpu_cost"] + computed.keys(), [cpu_cost] + computed.values())

