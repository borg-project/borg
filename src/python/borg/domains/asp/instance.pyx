"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import os
import numpy
import scipy.sparse

cimport libc.errno
cimport libc.string
cimport libc.stdlib
cimport posix.unistd
cimport cpython.exc
cimport numpy

class SAT_Instance(object):
    """A propositional formula in CNF."""

    def __init__(self, constraints):
        """Initialize."""

        self.constraints = constraints
        (self.M, self.N) = constraints.shape

    def to_clauses(self):
        """Return the list of clauses in this formula."""

        indptr = self.constraints.indptr
        indices = self.constraints.indices
        data = self.constraints.data

        clauses = []

        for m in xrange(self.M):
            i = self.constraints.indptr[m]
            j = self.constraints.indptr[m + 1]
            clause = [(indices[k] + 1) * numpy.sign(data[k]) for k in xrange(i, j)]

            clauses.append(clause)

        return clauses

    def write(self, out_file):
        """Write this CNF to a file, in DIMACS format."""

        out_file.write("p cnf {0} {1}\n".format(self.N, self.M))

        for clause in self.to_clauses():
            out_file.write(" ".join(map(str, clause)))
            out_file.write(" 0\n")

    #def satisfied(self, certificate):
        #"""Verify that the certificate satisfies this expression."""

        #assignment = numpy.ones(self.N, bool)

        #for literal in certificate:
            #variable = abs(literal) - 1

            #if literal == 0:
                #break
            #elif variable >= assignment.size:
                #return None

            #assignment[variable] = literal > 0

        #for clause in self.clauses:
            #satisfied = False

            #for literal in clause:
                #if assignment[abs(literal)] == literal > 0:
                    #satisfied = True

                    #break

            #if not satisfied:
                #return None

        #return assignment

    @staticmethod
    def from_clauses(clauses, N):
        values = []
        m_indices = []
        n_indices = []

        for (i, clause) in enumerate(clauses):
            for l in clause:
                values.append(int(numpy.sign(l)))

                m_indices.append(i)
                n_indices.append(abs(l) - 1)

        coo = \
            scipy.sparse.coo_matrix(
                (values, (m_indices, n_indices)),
                [len(clauses), N],
                )
        csr = coo.tocsr()

        return SAT_Instance(csr)

cdef struct DIMACS_Token:
    char* p
    int n

cdef class DIMACS_Lexer(object):
    cdef char* _p
    cdef char* _q
    cdef object _ward

    cdef start(self, char* p, object ward):
        self._p = p
        self._q = self._p
        self._ward = ward

    cdef DIMACS_Token lex(self):
        while self._q[0] == ' ' and self._q[0] != '\0':
            self._q += 1

        if self._q[0] == '\n':
            self._p = self._q
            self._q += 1
        elif self._q[0] == '\0':
            self._p = self._q
        else:
            self._p = self._q

            while self._q[0] != " " and self._q[0] != "\n" and self._q[0] != "\0":
                self._q += 1

        cdef DIMACS_Token token

        token.p = self._p
        token.n = self._q - self._p

        return token

cdef class ArrayVectorInt8(object):
    """Simple append-only resizing array."""

    cdef int size
    cdef int capacity
    cdef numpy.ndarray array
    cdef numpy.int8_t* data

    def __init__(self):
        """Initialize."""

        self.size = 0
        self.capacity = 1 * 1024 * 1024
        self.array = numpy.empty(self.capacity, numpy.int8)
        self.data = <numpy.int8_t*>self.array.data

    cdef bint append(self, numpy.int8_t v) except False:
        """Append a single element to the vector."""

        if self.size == self.capacity:
            self.expand()

        self.data[self.size] = v

        self.size += 1

        return True

    cdef expand(self):
        """Increase the size of the storage array."""

        self.capacity *= 2

        self.array.resize(self.capacity)

        self.data = <numpy.int8_t*>self.array.data

    cdef trim(self):
        """Reduce the size of the storage array to capacity."""

        self.capacity = self.size

        self.array.resize(self.capacity)

        self.data = <numpy.int8_t*>self.array.data

cdef class ArrayVectorInt32(object):
    """Simple append-only resizing array."""

    cdef int size
    cdef int capacity
    cdef numpy.ndarray array
    cdef numpy.int32_t* data

    def __init__(self):
        """Initialize."""

        self.size = 0
        self.capacity = 1 * 1024 * 1024
        self.array = numpy.empty(self.capacity, numpy.int32)
        self.data = <numpy.int32_t*>self.array.data

    cdef bint append(self, numpy.int32_t v) except False:
        """Append a single element to the vector."""

        if self.size == self.capacity:
            self.expand()

        self.data[self.size] = v

        self.size += 1

        return True

    cdef expand(self):
        """Increase the size of the storage array."""

        self.capacity *= 2

        self.array.resize(self.capacity)

        self.data = <numpy.int32_t*>self.array.data

    cdef trim(self):
        """Reduce the size of the storage array to capacity."""

        self.capacity = self.size

        self.array.resize(self.capacity)

        self.data = <numpy.int32_t*>self.array.data

cdef int strntol(char* p, int n) except? 2147483647:
    cdef char t[32]

    if n >= 32:
        raise ValueError("token length exceeds maximum")

    libc.string.strncpy(&t[0], p, n)

    t[n] = 0

    libc.errno.errno = 0

    v = libc.stdlib.strtol(&t[0], NULL, 10)

    if libc.errno.errno == 0:
        return v
    else:
        cpython.exc.PyErr_SetFromErrno(ValueError)

cdef class DIMACS_Parser(object):
    cdef DIMACS_Lexer _lexer
    cdef ArrayVectorInt8 _csr_data
    cdef ArrayVectorInt32 _csr_indices
    cdef ArrayVectorInt32 _csr_indptrs

    def parse(self, lexer):
        self._lexer = lexer
        self._csr_data = ArrayVectorInt8()
        self._csr_indices = ArrayVectorInt32()
        self._csr_indptrs = ArrayVectorInt32()

        while True:
            token = self._lexer.lex()

            if token.n == 0:
                return
            elif token.p[0] == "c":
                self.parse_comment()
            elif token.p[0] == "p":
                (kind, N, M) = self.parse_header()

                break

        if kind != "cnf":
            raise RuntimeError("unknown problem instance type")

        self._csr_indptrs.append(0)

        while self.parse_constraint():
            pass

        self._csr_data.trim()
        self._csr_indices.trim()
        self._csr_indptrs.trim()

        constraints = \
            scipy.sparse.csr_matrix(
                (self._csr_data.array, self._csr_indices.array, self._csr_indptrs.array),
                shape = (M, N),
                dtype = numpy.int8,
                )

        return SAT_Instance(constraints)

    cdef parse_comment(self):
        while True:
            token = self._lexer.lex()

            if token.n == 0 or token.p[0] == "\n":
                break

    cdef parse_header(self):
        token = self._lexer.lex()
        kind = token.p[:token.n]

        token = self._lexer.lex()
        N = strntol(token.p, token.n)

        token = self._lexer.lex()
        M = strntol(token.p, token.n)

        return (kind, N, M)

    cdef int parse_constraint(self) except -1:
        while True:
            token = self._lexer.lex()

            if token.n == 0:
                return 0
            elif token.p[0] == "0":
                self._csr_indptrs.append(self._csr_data.size)

                return 1
            elif token.p[0] != "\n":
                literal = strntol(token.p, token.n)

                if literal > 0:
                    value = 1
                else:
                    value = -1

                self._csr_data.append(value)
                self._csr_indices.append(libc.stdlib.abs(literal) - 1)

cdef extern from "sys/mman.h":
    void* mmap(void* start, size_t length, int prot, int flags, int fd, posix.unistd.off_t offset)
    int munmap(void* start, size_t length)

    cdef int PROT_READ
    cdef int MAP_SHARED
    cdef void* MAP_FAILED

cdef class MappedFile(object):
    """Manage a memory-mapped file."""

    cdef void* region
    cdef size_t length

    def __cinit__(self, fd):
        self.length = os.fstat(fd).st_size
        self.region = mmap(NULL, self.length, PROT_READ, MAP_SHARED, fd, 0)

        if self.region == MAP_FAILED:
            raise IOError("mmap failed")

    def __dealloc__(self):
        if self.region != NULL and self.region != MAP_FAILED:
            munmap(self.region, self.length)

def parse_sat_file(task_file):
    """Parse a SAT instance stored in DIMACS CNF format."""

    fileno = getattr(task_file, "fileno", None)
    lexer = DIMACS_Lexer()

    if fileno is None:
        contents = task_file.read()

        lexer.start(contents, contents)
    else:
        mapped = MappedFile(fileno())

        lexer.start(<char*>mapped.region, mapped)

    return DIMACS_Parser().parse(lexer)

