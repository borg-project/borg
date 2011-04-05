"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os.path
import signal
import numpy
import scipy.sparse
import cargo
import borg

logger = cargo.get_logger(__name__, default_level = "INFO")

named = {
    "nonlinear" : lambda opb: 1.0 if opb.nonlinear is None else -1.0,
    "optimization" : lambda opb: -1.0 if opb.objective is None else 1.0,
    "variables": lambda opb: opb.N,
    "constraints": lambda opb: opb.M,
    "ratio": lambda opb: opb.M / float(opb.N),
    "ratio_reciprocal": lambda opb: float(opb.N) / opb.M,
    }

def feature(method):
    assert method.__name__.startswith("compute_")

    named[method.__name__[8:]] = method

    return method

@feature
def compute_totals_log_min(instance):
    return numpy.log(1 + abs(numpy.min(instance.totals)))

@feature
def compute_totals_log_max(instance):
    return numpy.log(1 + abs(numpy.max(instance.totals)))

@feature
def compute_totals_log_mean(instance):
    return numpy.log(1 + abs(numpy.mean(instance.totals)))

@feature
def compute_totals_log_std(instance):
    return numpy.log(1 + abs(numpy.std(instance.totals)))

def build_node_degrees(opb):
    """Gather the CG, VG, and VCG node degrees."""

    if not hasattr(opb, "vcg_degrees_V"):
        # build adjacency matrix
        (C, V) = opb.constraints.shape

        adjacency_csr_CV = \
            scipy.sparse.csr_matrix(
                (numpy.ones(len(opb.constraints.data), int), opb.constraints.indices, opb.constraints.indptr),
                shape = (C, V),
                )
        adjacency_csr_VC = adjacency_csr_CV.T.tocsr()

        # compute variable-clause graph degrees
        opb.vcg_degrees_V = numpy.asarray(adjacency_csr_VC.sum(axis = 1))[:, 0]
        opb.vcg_degrees_C = numpy.asarray(adjacency_csr_VC.sum(axis = 0))[0, :]

        # compute variable graph degrees
        opb.vg_degrees_V = numpy.zeros(V, int)

        for v in xrange(V):
            i = adjacency_csr_VC.indptr[v]
            j = adjacency_csr_VC.indptr[v + 1]

            for k in xrange(i, j):
                c = adjacency_csr_VC.indices[k]

                opb.vg_degrees_V[v] += opb.vcg_degrees_C[c]

            opb.vg_degrees_V[v] -= j - i

        # compute clause graph degrees
        opb.cg_degrees_C = numpy.zeros(C, int)

        for c in xrange(C):
            i = adjacency_csr_CV.indptr[c]
            j = adjacency_csr_CV.indptr[c + 1]

            for k in xrange(i, j):
                v = adjacency_csr_CV.indices[k]

                opb.cg_degrees_C[c] += opb.vcg_degrees_V[v]

            opb.cg_degrees_C[c] -= j - i

        # compute variable coefficient means
        coefficient_sums_C = numpy.asarray(opb.constraints.sum(axis = 1))[:, 0]

        opb.coefficient_means_C = coefficient_sums_C / opb.vcg_degrees_C

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

@graph_feature
def compute_coefficient_means_mean(opb):
    return numpy.mean(opb.coefficient_means_C)

@graph_feature
def compute_coefficient_means_std(opb):
    return numpy.std(opb.coefficient_means_C)

class FeaturesTimedOut(Exception):
    pass

def handle_timeout_signal(number, frame):
    raise FeaturesTimedOut()

def compute_all(instance, cpu_seconds = None):
    """Compute all features of a PB instance."""

    with borg.accounting() as accountant:
        try:
            if cpu_seconds is not None:
                signal.setitimer(signal.ITIMER_VIRTUAL, cpu_seconds)
                signal.signal(signal.SIGVTALRM, handle_timeout_signal)

            computed = dict((k, v(instance)) for (k, v) in named.items())

            signal.setitimer(signal.ITIMER_VIRTUAL, 0.0)
        except FeaturesTimedOut:
            computed = {}

    cpu_cost = accountant.total.cpu_seconds

    if len(computed) > 0:
        logger.info("features took %.2f CPU seconds", cpu_cost)
    else:
        logger.info("features timed out after %.2f CPU seconds", cpu_cost)

    return (["cpu_cost"] + computed.keys(), [cpu_cost] + computed.values())

