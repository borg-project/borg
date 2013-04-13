"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import numpy
import scipy.sparse
import borg

logger = borg.get_logger(__name__)

class PseudoBooleanInstance(object):
    def __init__(self, objective, totals, relations, constraints, nonlinear = False):
        (self.M, self.N) = constraints.shape
        self.objective = objective
        self.totals = numpy.asarray(totals)
        self.relations = numpy.asarray(relations)
        self.constraints = constraints
        self.nonlinear = nonlinear

cdef enum OPB_Relation:
    OPB_RELATION_NONE = 0
    OPB_RELATION_EQ = 1
    OPB_RELATION_GEQ = 2

cdef enum OPB_TokenKind:
    OPB_TOKEN_NONE
    OPB_TOKEN_SPACE #  *
    OPB_TOKEN_COMMENT # \*.*\n
    OPB_TOKEN_MIN # min:
    OPB_TOKEN_INTEGER # [+-]?[0-9]+
    OPB_TOKEN_VARIABLE # x[0-9]+
    OPB_TOKEN_RELATION_EQ # =
    OPB_TOKEN_RELATION_GEQ # >=
    OPB_TOKEN_SEMICOLON # ;

cdef struct OPB_Token:
    OPB_TokenKind kind
    char* p
    char* q

cdef class OPB_Lexer(object):
    cdef bytes _text
    cdef int _line
    cdef OPB_TokenKind _kind
    cdef char* _p
    cdef char* _q

    def __init__(self, text):
        self._text = text # keep the string alive
        self._p = text
        self._q = text
        self._line = 1

        self.lex()

    cdef raise_error(self):
        raise RuntimeError("OPB syntax error on line {0}".format(self._line))

    cdef OPB_Token peek(self) except *:
        return OPB_Token(self._kind, self._p, self._q)

    cdef OPB_Token take(self) except *:
        cdef OPB_Token token = self.peek()

        self.lex()

        return token

    cdef str take_comment(self):
        token = self.take()

        if token.kind != OPB_TOKEN_COMMENT:
            self.raise_error()
        else:
            return str(token.p[:token.q - token.p])

    cdef int take_integer(self) except? -1:
        token = self.take()

        if token.kind != OPB_TOKEN_INTEGER:
            self.raise_error()
        else:
            return int(token.p[:token.q - token.p])

    cdef int take_variable(self) except -1:
        token = self.take()

        if token.kind != OPB_TOKEN_VARIABLE:
            self.raise_error()
        else:
            return int(token.p[1:token.q - token.p])

    cdef OPB_Relation take_relation(self) except OPB_RELATION_NONE:
        token = self.take()

        if token.kind == OPB_TOKEN_RELATION_EQ:
            return OPB_RELATION_EQ
        elif token.kind == OPB_TOKEN_RELATION_GEQ:
            return OPB_RELATION_GEQ
        else:
            self.raise_error()

    cdef int take_semicolon(self) except -1:
        token = self.take()

        if token.kind != OPB_TOKEN_SEMICOLON:
            self.raise_error()

    cdef int lex(self) except -1:
        while True:
            self._p = self._q
            self._q += 1

            self.lex_raw()

            if self._kind != OPB_TOKEN_SPACE:
                break

    cdef int lex_raw(self) except -1:
        # space
        if self._p[0] == ' ' or self._p[0] == '\n':
            if self._p[0] == '\n':
                self._line += 1

            while self._q[0] == ' ' or self._q[0] == '\n':
                self._q += 1

                if self._q[0] == '\n':
                    self._line += 1

            self._kind = OPB_TOKEN_SPACE
        # comment
        elif self._p[0] == '*':
            while self._q[0] != '\n' and self._q[0] != '\0':
                self._q += 1

            self._kind = OPB_TOKEN_COMMENT
        # objective flag
        elif self._p[0] == 'm':
            if self._q[0] != 'i' or self._q[1] != 'n' or self._q[2] != ':':
                self.raise_error()
            else:
                self._q += 3
                self._kind = OPB_TOKEN_MIN
        # explicitly-signed integer
        elif self._p[0] == '+' or self._p[0] == '-':
            while self._q[0] >= '0' and self._q[0] <= '9':
                self._q += 1

            if self._q == self._p + 1:
                self.raise_error()
            else:
                self._kind = OPB_TOKEN_INTEGER
        # implicitly-signed integer
        elif self._p[0] >= '0' and self._p[0] <= '9':
            while self._q[0] >= '0' and self._q[0] <= '9':
                self._q += 1

            self._kind = OPB_TOKEN_INTEGER
        # variable name
        elif self._p[0] == 'x':
            while self._q[0] >= '0' and self._q[0] <= '9':
                self._q += 1

            if self._q == self._p + 1:
                self.raise_error()
            else:
                self._kind = OPB_TOKEN_VARIABLE
        # equality relation
        elif self._p[0] == '=':
            self._kind = OPB_TOKEN_RELATION_EQ
        # inequality relation
        elif self._p[0] == '>':
            if self._q[0] != '=':
                self.raise_error()
            else:
                self._q += 1

                self._kind = OPB_TOKEN_RELATION_GEQ
        # end of constraint
        elif self._p[0] == ';':
            self._kind = OPB_TOKEN_SEMICOLON
        # end of file
        elif self._p[0] == '\0':
            self._kind = OPB_TOKEN_NONE
            self._p = NULL
            self._q = NULL
        # lex error
        else:
            self.raise_error()

cdef class OPB_Parser(object):
    cdef OPB_Lexer _lexer
    cdef list _objective
    cdef list _relations
    cdef list _totals
    cdef list _csr_data
    cdef list _csr_indices
    cdef list _csr_indptrs

    def parse(self, text):
        self._lexer = OPB_Lexer(text)
        self._objective = None
        self._relations = []
        self._totals = []
        self._csr_data = []
        self._csr_indices = []
        self._csr_indptrs = [0]

        # parse the header comment
        (M, N, nonlinear) = parse_opb_file_header(self._lexer.take_comment())

        # parse the objective, if any
        while True:
            token = self._lexer.peek()

            if token.kind == OPB_TOKEN_NONE:
                break
            elif token.kind == OPB_TOKEN_COMMENT:
                self._lexer.lex()
            elif token.kind == OPB_TOKEN_MIN:
                self._lexer.lex()

                self.parse_objective()
            else:
                break

        # parse the constraints
        while True:
            token = self._lexer.peek()

            if token.kind == OPB_TOKEN_NONE:
                break
            elif token.kind == OPB_TOKEN_COMMENT:
                self._lexer.lex()
            elif token.kind == OPB_TOKEN_INTEGER:
                self.parse_constraint()
            else:
                self._lexer.raise_error()

        constraints = \
            scipy.sparse.csr_matrix(
                (self._csr_data, self._csr_indices, self._csr_indptrs),
                shape = (M, N),
                dtype = numpy.int64,
                )

        return PseudoBooleanInstance(self._objective, self._totals, self._relations, constraints)

    cdef parse_objective(self):
        self._objective = []

        while True:
            token = self._lexer.peek()

            if token.kind == OPB_TOKEN_INTEGER:
                self._objective.append((
                    self._lexer.take_integer(),
                    self._lexer.take_variable() - 1,
                    ))
            elif token.kind == OPB_TOKEN_SEMICOLON:
                self._lexer.lex()

                break
            else:
                self._lexer.raise_error()

    cdef parse_constraint(self):
        while True:
            token = self._lexer.peek()

            if token.kind == OPB_TOKEN_INTEGER:
                self._csr_data.append(self._lexer.take_integer())
                self._csr_indices.append(self._lexer.take_variable() - 1)
            else:
                break

        self._relations.append(self._lexer.take_relation())
        self._totals.append(self._lexer.take_integer())
        self._csr_indptrs.append(len(self._csr_data))

        self._lexer.take_semicolon()

def parse_opb_file_header(line):
    (N,) = map(int, re.findall("#variable= *([0-9]+)", line))
    (M,) = map(int, re.findall("#constraint= *([0-9]+)", line))
    nonlinear = re.search("#product=", line) is not None

    return (M, N, nonlinear)

def parse_opb_file_linear(task_file):
    return OPB_Parser().parse(task_file.read())

