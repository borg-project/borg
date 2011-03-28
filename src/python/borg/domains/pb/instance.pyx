"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import re
import cargo

logger = cargo.get_logger(__name__)

class PseudoBooleanInstance(object):
    def __init__(self, N, constraints, objective = None, nonlinear = False):
        self.N = N
        self.constraints = constraints
        self.objective = objective
        self.nonlinear = nonlinear

#def parse_opb_file_linear(opb_file):
    #"""Parse a pseudo-Boolean satisfiability instance."""

    ## prepare constraint parsing
    #def parse_terms(parts):
        #terms = []

        #assert len(parts) % 2 == 0

        #for i in xrange(0, len(parts), 2):
            #weight = int(parts[i][1:])
            #literal = int(parts[i + 1][1:])

            #terms.append((weight, literal))

        #return terms

    ## parse all lines
    #constraints = []
    #objective = None

    #for (i, line) in enumerate(opb_file):
        #if i == 0:
            #(M, N, nonlinear) = parse_opb_file_header(line)
        #elif not line.startswith("*"):
            #if line.startswith("min:"):
                ## objective line
                #if objective is not None:
                    #raise RuntimeError("multiple objectives in PB input")

                #objective = parse_terms(line[4:-2].split())
            #else:
                ## constraint line
                #parts = line[:-2].split()
                #terms = parse_terms(parts[:-2])
                #relation = parts[-2]
                #value = int(parts[-1])

                #constraints.append((terms, relation, value))

    ## ...
    #return PseudoBooleanInstance(N, constraints, objective, nonlinear)

def parse_opb_file_header(line):
    (N,) = map(int, re.findall("#variable= *([0-9]+)", line))
    (M,) = map(int, re.findall("#constraint= *([0-9]+)", line))
    nonlinear = re.search("#product=", line) is not None

    return (M, N, nonlinear)

cdef enum OPB_TokenKind:
    OPB_TOKEN_NONE
    OPB_TOKEN_SPACE #  *
    OPB_TOKEN_COMMENT # \*.*\n
    OPB_TOKEN_INTEGER # [+-]?[0-9]+
    OPB_TOKEN_VARIABLE # x[0-9]+
    OPB_TOKEN_RELATION # >?=
    OPB_TOKEN_SEMICOLON # ;

cdef struct OPB_Token:
    OPB_TokenKind kind
    char* p
    char* q

cdef class OPB_Lexer(object):
    cdef bytes _text
    cdef char* _p
    cdef char* _q

    def __init__(self, text):
        self._text = text # keep the string alive
        self._p = text

    cdef OPB_Token lex(self) except *:
        cdef OPB_Token 

        while True:
            self._q = self._p + 1

            token = self.inner_lex()

            self._p = self._q

            if token.kind != OPB_TOKEN_SPACE:
                return token

    cdef OPB_Token inner_lex(self) except *:
        # space
        if self._p[0] == ' ' or self._p[0] == '\n':
            while self._q[0] == ' ' or self._q[0] == '\n':
                self._q += 1

            return OPB_Token(OPB_TOKEN_SPACE, self._p, self._q)
        # comment
        if self._p[0] == '*':
            while self._q[0] != '\n' and self._q[0] != '\0':
                self._q += 1

            return OPB_Token(OPB_TOKEN_COMMENT, self._p, self._q)
        # explicitly-signed integer
        elif self._p[0] == '+' or self._p[0] == '-':
            while self._q[0] >= '0' and self._q[0] <= '9':
                self._q += 1

            if self._q == self._p + 1:
                pass # XXX
            else:
                return OPB_Token(OPB_TOKEN_INTEGER, self._p, self._q)
        # implicitly-signed integer
        elif self._p[0] >= '0' and self._p[0] <= '9':
            while self._q[0] >= '0' and self._q[0] <= '9':
                self._q += 1

            return OPB_Token(OPB_TOKEN_INTEGER, self._p, self._q)
        # variable name
        elif self._p[0] == 'x':
            while self._q[0] >= '0' and self._q[0] <= '9':
                self._q += 1

            if self._q == self._p + 1:
                pass # XXX
            else:
                return OPB_Token(OPB_TOKEN_VARIABLE, self._p, self._q)
        # equality relation
        elif self._p[0] == '=':
            return OPB_Token(OPB_TOKEN_RELATION, self._p, self._q)
        # inequality relation
        elif self._p[0] == '>':
            if self._q[0] != '=':
                pass # XXX
            else:
                self._q += 1

                return OPB_Token(OPB_TOKEN_RELATION, self._p, self._q)
        # end of constraint
        elif self._p[0] == ';':
            return OPB_Token(OPB_TOKEN_SEMICOLON, self._p, self._q)
        # end of file
        elif self._p[0] == '\0':
            return OPB_Token(OPB_TOKEN_NONE, NULL, NULL)
        # lex error
        else:
            assert False # XXX

cdef class OPB_Parser(object):
    cdef OPB_Lexer _lexer
    cdef list _weights
    cdef list _csr_data
    cdef list _csr_indices
    cdef list _csr_indptrs

    def parse(self, text):
        self._lexer = OPB_Lexer(text)
        self._weights = []
        self._csr_data = []
        self._csr_indices = []
        self._csr_indptrs = [0]

        while True:
            token = self._lexer.lex()

            if token.kind == OPB_TOKEN_NONE:
                break
            elif token.kind == OPB_TOKEN_COMMENT:
                pass
            elif token.kind == OPB_TOKEN_INTEGER:
                self.parse_constraint(token)
            else:
                assert False # XXX

            #if len(token) == 0:
                #return
            #elif token[0] == "c":
                #self.parse_comment()
            #elif token[0] == "p":
                #(kind, N, M) = self.parse_header()

                #break

        #if kind == "cnf":
            #while self.parse_constraint():
                #self._weights.append(1)
        #elif kind == "wcnf":
            #while self.parse_weighted_constraint():
                #pass
        #else:
            #raise RuntimeError("unknown instance type")

        #constraints = \
            #scipy.sparse.csr_matrix(
                #(self._csr_data, self._csr_indices, self._csr_indptrs),
                #shape = (M, N),
                #dtype = numpy.int8,
                #)

        #return MAX_SAT_Instance(self._weights, constraints)

        raise SystemExit()

    cdef parse_constraint(self, OPB_Token token):
        assert token.kind == OPB_TOKEN_INTEGER

        term = self.parse_term(token)

        while True:
            token = self._lexer.lex()

            if token.kind == OPB_TOKEN_INTEGER:
                self.parse_term(token) # XXX
            elif token.kind == OPB_TOKEN_RELATION:
                pass # XXX
            elif token.kind == OPB_TOKEN_SEMICOLON:
                break
            else:
                assert False # XXX

    cdef parse_term(self, OPB_Token token):
        assert token.kind == OPB_TOKEN_INTEGER

        weight = int(token.p[:token.q - token.p])

        token = self._lexer.lex()

        assert token.kind == OPB_TOKEN_VARIABLE

        print token.p[:token.q - token.p]

        variable = int(token.p[1:token.q - token.p - 1])

        print weight, variable

        pass # XXX

def parse_opb_file_linear(task_file):
    return OPB_Parser().parse(task_file.read())

