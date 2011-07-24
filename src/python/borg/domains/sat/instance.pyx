"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import scipy.sparse

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

cdef class DIMACS_Lexer(object):
    cdef bytes _text
    cdef char* _p
    cdef char* _q

    def __init__(self, text):
        self._text = text
        self._p = self._text
        self._q = self._p

    cdef bytes lex(self):
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

        if self._p == self._q:
            return None
        else:
            return self._p[:self._q - self._p]

cdef class DIMACS_Parser(object):
    cdef DIMACS_Lexer _lexer
    cdef list _csr_data
    cdef list _csr_indices
    cdef list _csr_indptrs

    def parse(self, text):
        self._lexer = DIMACS_Lexer(text)
        self._csr_data = []
        self._csr_indices = []
        self._csr_indptrs = [0]

        while True:
            token = self._lexer.lex()

            if len(token) == 0:
                return
            elif token[0] == "c":
                self.parse_comment()
            elif token[0] == "p":
                (kind, N, M) = self.parse_header()

                break

        if kind == "cnf":
            while self.parse_constraint():
                pass
        else:
            raise RuntimeError("unknown instance type")

        constraints = \
            scipy.sparse.csr_matrix(
                (self._csr_data, self._csr_indices, self._csr_indptrs),
                shape = (M, N),
                dtype = numpy.int8,
                )

        return SAT_Instance(constraints)

    cdef parse_comment(self):
        while True:
            token = self._lexer.lex()

            if len(token) == 0 or token[0] == "\n":
                break

    cdef parse_header(self):
        kind = self._lexer.lex()
        N = int(self._lexer.lex())
        M = int(self._lexer.lex())

        return (kind, N, M)

    cdef int parse_constraint(self):
        cdef char* token
        cdef int literal

        while True:
            str_token = self._lexer.lex()

            if str_token is None:
                return False
            else:
                token = str_token

                if token[0] == "0":
                    self._csr_indptrs.append(len(self._csr_data))

                    return True
                elif token[0] != "\n":
                    literal = int(str_token)

                    if literal > 0:
                        value = 1
                    else:
                        value = -1

                    self._csr_data.append(value)
                    self._csr_indices.append(abs(literal) - 1)

def parse_sat_file(task_file):
    return DIMACS_Parser().parse(task_file.read())

