"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import cStringIO as StringIO
import nose
import nose.tools
import borg

input_text_plain = \
"""* foo!@#$
-1 x1 +23 x2 = +0;
-1 x1 +23 x2 >= -0;
-1 x1 +23 x2 >= -1;
* foo!@#$
+1 x1   >=   42 ;
* foo!@#$
* foo!@#$
"""

input_text_nlc = \
"""* foo!@#$
-1 x1 x2 +23 x2 = +0;
-1 x1 +23 x2 ~x1 >= -0;
-1 x1 +23 x2 >= -1;
* foo!@#$
+1 ~x1   >=   42 ;
* foo!@#$
* foo!@#$
"""

def test_parse_pbs_lin():
    input_file = StringIO.StringIO(input_text_plain)
    instance = borg.domains.pb.opb.parse_opb_file(input_file)
    constraints = [
        ([(-1, [1]), (23, [2])], "=", 0),
        ([(-1, [1]), (23, [2])], ">=", 0),
        ([(-1, [1]), (23, [2])], ">=", -1),
        ([(1, [1])], ">=", 42),
        ]

    nose.tools.assert_true(instance.objective is None)
    nose.tools.assert_equal(len(instance.constraints), 4)
    nose.tools.assert_equal(instance.constraints, constraints)

def test_parse_pbo_lin():
    input_text = "*zorro!\nmin: -40 x1 3 x2 \n" + input_text_plain
    input_file = StringIO.StringIO(input_text)
    instance = borg.domains.pb.opb.parse_opb_file(input_file)

    nose.tools.assert_equal(len(instance.constraints), 4)
    nose.tools.assert_equal(instance.objective, [(-40, [1]), (3, [2])])

def test_parse_pbs_nlc():
    input_file = StringIO.StringIO(input_text_nlc)
    instance = borg.domains.pb.opb.parse_opb_file(input_file)
    constraints = [
        ([(-1, [1, 2]), (23, [2])], "=", 0),
        ([(-1, [1]), (23, [2, -1])], ">=", 0),
        ([(-1, [1]), (23, [2])], ">=", -1),
        ([(1, [-1])], ">=", 42),
        ]

    nose.tools.assert_true(instance.objective is None)
    nose.tools.assert_equal(len(instance.constraints), 4)
    nose.tools.assert_equal(instance.constraints, constraints)

def test_parse_pbo_nlc():
    input_text = "*zorro!\nmin: -40 x1 3 x2 ~x3 \n" + input_text_nlc
    input_file = StringIO.StringIO(input_text)
    instance = borg.domains.pb.opb.parse_opb_file(input_file)

    nose.tools.assert_equal(len(instance.constraints), 4)
    nose.tools.assert_equal(instance.objective, [(-40, [1]), (3, [2, -3])])

if __name__ == "__main__":
    nose.main(defaultTest = __name__)

