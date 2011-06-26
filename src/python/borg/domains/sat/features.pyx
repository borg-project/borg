"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import resource
import numpy
import scipy.sparse
import cargo
import borg

cimport numpy
cimport cython

cdef extern from "math.h":
    double fabs(double)

logger = cargo.get_logger(__name__, default_level = "INFO")

def entropy_of(array, states, maximum = None):
    (N,) = array.shape

    if maximum is None:
        maximum = states

    (binned, _) = numpy.histogram(array, bins = states, range = (0, maximum))

    normed = binned[binned > 0].astype(float)
    normed /= N
    normed *= numpy.log(normed)

    return -numpy.sum(normed)

def array_features(prefix, array, cv = "cv"):
    mu = numpy.mean(array)
    sigma = numpy.std(array)
    features = [("{0}-mean".format(prefix), mu)]

    if cv == "cv":
        epsilon = numpy.finfo(float).eps

        if abs(mu) < epsilon and sigma < epsilon:
            cv = 0.0
        else:
            cv = sigma / abs(mu)

        features += [("{0}-coeff-variation".format(prefix), cv)]
    elif cv == "sd":
        features += [("{0}-stdev".format(prefix), sigma)]

    features += [
        ("{0}-min".format(prefix), numpy.min(array)),
        ("{0}-max".format(prefix), numpy.max(array)),
        ]

    return features

def compute_vc_graph_degrees(adjacency_csr_VC):
    (V, C) = adjacency_csr_VC.shape

    vcg_degrees_V = numpy.asarray(adjacency_csr_VC.sum(axis = 1))[:, 0]
    vcg_degrees_C = numpy.asarray(adjacency_csr_VC.sum(axis = 0))[0, :]

    features = array_features("VCG-VAR", vcg_degrees_V / float(C))
    features += [("VCG-VAR-entropy", entropy_of(vcg_degrees_V, C))]
    features += array_features("VCG-CLAUSE", vcg_degrees_C / float(V))
    features += [("VCG-CLAUSE-entropy", entropy_of(vcg_degrees_C, V))]

    logger.info("computed variable-clause graph statistics")

    return features

@cython.boundscheck(False)
def compute_clause_balance_statistics(constraints_csr_CV):
    cdef int C = constraints_csr_CV.shape[0]
    cdef int V = constraints_csr_CV.shape[1]

    cdef numpy.ndarray[double, ndim = 1] pn_ratios_C = numpy.zeros(C)
    cdef numpy.ndarray[double, ndim = 1] horn_variables_V = numpy.zeros(V)

    cdef int horn_clauses = 0

    cdef numpy.ndarray[int, ndim = 1] constraints_csr_CV_indptr = constraints_csr_CV.indptr
    cdef numpy.ndarray[int, ndim = 1] constraints_csr_CV_indices = constraints_csr_CV.indices
    cdef numpy.ndarray[numpy.int8_t, ndim = 1] constraints_csr_CV_data = constraints_csr_CV.data

    cdef int c
    cdef int i
    cdef int j
    cdef int k
    cdef int v

    for c in xrange(C):
        i = constraints_csr_CV_indptr[c]
        j = constraints_csr_CV_indptr[c + 1]

        positives = 0.0

        for k in xrange(i, j):
            if constraints_csr_CV_data[k] > 0:
                positives += 1.0

        # XXX questionable, but match SATzilla's behavior for now
        if positives <= 1.0:
            horn_clauses += 1

            for k in xrange(i, j):
                v = constraints_csr_CV_indices[k]

                horn_variables_V[v] += 1.0

        pn_ratios_C[c] = 2.0 * fabs(0.5 - positives / (j - i))

    features = array_features("POSNEG-RATIO-CLAUSE", pn_ratios_C)
    features += [("POSNEG-RATIO-CLAUSE-entropy", entropy_of(pn_ratios_C, 100, 1.0))]

    logger.info("computed clause balance statistics")

    return (features, horn_variables_V, horn_clauses)

def compute_variable_balance_statistics(constraints_csr_VC):
    (V, C) = constraints_csr_VC.shape

    pn_ratios_V = numpy.zeros(V)

    for v in xrange(V):
        i = constraints_csr_VC.indptr[v]
        j = constraints_csr_VC.indptr[v + 1]

        positives = 0.0

        for k in xrange(i, j):
            if constraints_csr_VC.data[k] > 0:
                positives += 1.0

        pn_ratios_V[v] = 2.0 * abs(0.5 - positives / (j - i))

    features = array_features("POSNEG-RATIO-VAR", pn_ratios_V, cv = "sd")
    features += [("POSNEG-RATIO-VAR-entropy", entropy_of(pn_ratios_V, 100, 1.0))]

    logger.info("computed variable balance statistics")

    return features

def compute_small_clause_counts(constraints_csr_CV):
    (C, V) = constraints_csr_CV.shape

    unary = 0
    binary = 0
    trinary = 0

    for c in xrange(C):
        i = constraints_csr_CV.indptr[c]
        j = constraints_csr_CV.indptr[c + 1]

        length = j - i

        if length == 1:
            unary += 1
        elif length == 2:
            binary += 1
        elif length == 3:
            trinary += 1

    logger.info("computed small-clause counts")

    return [
        ("UNARY", unary / float(C)),
        ("BINARY+", (unary + binary) / float(C)),
        ("TRINARY+", (unary + binary + trinary) / float(C)),
        ]

def compute_horn_clause_counts(C, horn_variables_V, horn_clauses):
    (V,) = horn_variables_V.shape

    features = array_features("HORNY-VAR", horn_variables_V / float(C))
    features += [("HORNY-VAR-entropy", entropy_of(horn_variables_V, C))]
    features += [("horn-clauses-fraction", horn_clauses / float(C))]

    logger.info("computed Horn-clause counts")

    return features

@cython.boundscheck(False)
def compute_variable_graph_degrees(constraints_csr_CV, constraints_csr_VC):
    cdef int C = constraints_csr_CV.shape[0]
    cdef int V = constraints_csr_CV.shape[1]

    cdef numpy.ndarray[long] vg_degrees_V = numpy.zeros(V, int)
    cdef numpy.ndarray[numpy.uint8_t] vg_setmask_V = numpy.empty(V, numpy.uint8)

    cdef numpy.ndarray[int] constraints_csr_CV_indptr = constraints_csr_CV.indptr
    cdef numpy.ndarray[int] constraints_csr_CV_indices = constraints_csr_CV.indices

    cdef numpy.ndarray[int] constraints_csr_VC_indptr = constraints_csr_VC.indptr
    cdef numpy.ndarray[int] constraints_csr_VC_indices = constraints_csr_VC.indices

    cdef int v
    cdef int i
    cdef int j
    cdef int k
    cdef int c
    cdef int a
    cdef int b
    cdef int w

    for v in xrange(V):
        i = constraints_csr_VC_indptr[v]
        j = constraints_csr_VC_indptr[v + 1]

        vg_setmask_V[:] = 0

        for k in xrange(i, j):
            c = constraints_csr_VC_indices[k]
            a = constraints_csr_CV_indptr[c]
            b = constraints_csr_CV_indptr[c + 1]

            for d in xrange(a, b):
                w = constraints_csr_CV_indices[d]

                if w != v and not vg_setmask_V[w]:
                    vg_setmask_V[w] = 1
                    vg_degrees_V[v] += 1

    features = array_features("VG", vg_degrees_V / float(C))
    features += [("KLB-featuretime", 0.0)] # XXX

    logger.info("computed variable graph statistics")

    return features

@cython.boundscheck(False)
def construct_clause_graph(constraints_csr_CV, constraints_csr_VC):
    cdef int C = constraints_csr_CV.shape[0]
    cdef int V = constraints_csr_CV.shape[1]

    cc_indices = []
    cc_indptrs = [0]

    cdef numpy.ndarray[numpy.uint8_t] cg_setmask_C = numpy.empty(C, numpy.uint8)

    cdef numpy.ndarray[int] constraints_csr_CV_indptr = constraints_csr_CV.indptr
    cdef numpy.ndarray[int] constraints_csr_CV_indices = constraints_csr_CV.indices
    cdef numpy.ndarray[numpy.int8_t] constraints_csr_CV_data = constraints_csr_CV.data

    cdef numpy.ndarray[int] constraints_csr_VC_indptr = constraints_csr_VC.indptr
    cdef numpy.ndarray[int] constraints_csr_VC_indices = constraints_csr_VC.indices
    cdef numpy.ndarray[numpy.int8_t] constraints_csr_VC_data = constraints_csr_VC.data

    cdef int c
    cdef int d
    cdef int i
    cdef int j
    cdef int v
    cdef int vp
    cdef int jp

    for c in xrange(C):
        cg_setmask_C[:] = False

        for i in xrange(constraints_csr_CV_indptr[c], constraints_csr_CV_indptr[c + 1]):
            v = constraints_csr_CV_indices[i]
            vp = constraints_csr_CV_data[i]

            for j in xrange(constraints_csr_VC_indptr[v], constraints_csr_VC_indptr[v + 1]):
                d = constraints_csr_VC_indices[j]
                jp = constraints_csr_VC_data[j]

                if d != c and (vp > 0) != (jp > 0) and not cg_setmask_C[d]:
                    cg_setmask_C[d] = True

                    cc_indices.append(d)

        cc_indptrs.append(len(cc_indices))

    logger.info("constructed clause constraint graph")

    return \
        scipy.sparse.csr_matrix(
            (numpy.ones(len(cc_indices), bool), numpy.array(cc_indices, int), cc_indptrs),
            (C, C),
            )

def compute_clause_graph_degrees(adjacency_csr_CC):
    (C, _) = adjacency_csr_CC.shape

    cg_degrees_C = numpy.zeros(C, int)

    for c in xrange(C):
        a = adjacency_csr_CC.indptr[c]
        b = adjacency_csr_CC.indptr[c + 1]

        cg_degrees_C[c] = b - a

    features = array_features("CG", cg_degrees_C / float(C))
    features += [("CG-entropy", entropy_of(cg_degrees_C, C))]

    logger.info("computed clause constraint graph statistics")

    return features

@cython.boundscheck(False)
def compute_cluster_coefficients(adjacency_csr_CC):
    cdef int C = adjacency_csr_CC.shape[0]

    cdef numpy.ndarray[double] cg_coefficients_C = numpy.empty(C, float)
    cdef numpy.ndarray[numpy.uint8_t] cg_setmask_C = numpy.empty(C, numpy.uint8)

    cdef numpy.ndarray[int] adjacency_csr_CC_indptr = adjacency_csr_CC.indptr
    cdef numpy.ndarray[int] adjacency_csr_CC_indices = adjacency_csr_CC.indices

    cdef unsigned int c
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int d
    cdef unsigned int l
    cdef unsigned int m
    cdef unsigned int e
    cdef unsigned int edges

    #print len(adjacency_csr_CC.indices)

    for c in xrange(C):
        i = adjacency_csr_CC_indptr[c]
        j = adjacency_csr_CC_indptr[c + 1]

        edges = j - i

        for k in xrange(C):
            cg_setmask_C[k] = 0

        for k in xrange(i, j):
            d = adjacency_csr_CC_indices[k]

            cg_setmask_C[d] = 1

        cg_setmask_C[c] = 1

        for k in xrange(i, j):
            d = adjacency_csr_CC_indices[k]
            l = adjacency_csr_CC_indptr[d]
            m = adjacency_csr_CC_indptr[d + 1]

            for n in xrange(l, m):
                e = adjacency_csr_CC_indices[n]

                if e != d and cg_setmask_C[e]:
                    edges += 1

        cg_coefficients_C[c] = edges / <double>((j - i + 1) * (j - i))

        #print c, j - i

    features = array_features("cluster-coeff", cg_coefficients_C)
    features += [("cluster-coeff-entropy", entropy_of(cg_coefficients_C, 100, 1.0))]
    features += [("CG-featuretime", 0.0)] # XXX

    logger.info("computed clause constraint clustering coefficients")

    return features

def compute_features(cnf):
    """Gather the basic structural features."""

    # mise en place
    (C, V) = cnf.constraints.shape

    constraints_csr_CV = cnf.constraints
    constraints_csr_VC = constraints_csr_CV.T.tocsr()

    adjacency_csr_CV = \
        scipy.sparse.csr_matrix(
            (
                numpy.ones(len(cnf.constraints.data), int),
                cnf.constraints.indices,
                cnf.constraints.indptr,
                ),
            shape = (C, V),
            )
    adjacency_csr_VC = adjacency_csr_CV.T.tocsr()

    # compute feature groups
    features = [
        ("nvars", V),
        ("nclauses", C),
        ("vars-clauses-ratio", float(V) / C),
        ]

    features += compute_vc_graph_degrees(adjacency_csr_VC)

    (cb_features, horn_variables_V, horn_clauses) = \
        compute_clause_balance_statistics(constraints_csr_CV)

    features += cb_features
    features += compute_variable_balance_statistics(constraints_csr_VC)
    features += compute_small_clause_counts(constraints_csr_CV)
    features += compute_horn_clause_counts(C, horn_variables_V, horn_clauses)
    features += compute_variable_graph_degrees(constraints_csr_CV, constraints_csr_VC)

    # XXX SATzilla's tool spits out useless values if its CPU expenditure for CG computation
    # XXX exceeds a (quite low) threshold, making these features impossible to simulate

    #adjacency_csr_CC = construct_clause_graph(constraints_csr_CV, constraints_csr_VC)
    #features += compute_clause_graph_degrees(adjacency_csr_CC)
    #features += compute_cluster_coefficients(adjacency_csr_CC)
    features += [
        ("CG-mean", -1),
        ("CG-coeff-variation", 0),
        ("CG-min", -1),
        ("CG-max", 0),
        ("CG-entropy", -1),
        ("cluster-coeff-mean", -1),
        ("cluster-coeff-coeff-variation", 0),
        ("cluster-coeff-min", -1),
        ("cluster-coeff-max", 0),
        ("cluster-coeff-entropy", 0),
        ("CG-featuretime", -1),
        ]

    return features

def get_features_for(cnf_path):
    """Obtain features of a CNF."""

    previous_utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime

    with open(cnf_path) as cnf_file:
        cnf = borg.instance.parse_sat_file(cnf_file)

    core_features = compute_features(cnf)

    cost = resource.getrusage(resource.RUSAGE_SELF).ru_utime - previous_utime

    logger.info("collected features for %s in %.2f s", cnf_path, cost)

    return zip(*([("cost", 0.5)] + core_features)) # XXX

