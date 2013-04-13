from setuptools import setup
from setuptools.extension import Extension

try:
    import Cython.Distutils
except ImportError:
    cmdclass = {}
    ext_modules = None

    print "WARNING: unable to import Cython."
else:
    cmdclass = {"build_ext": Cython.Distutils.build_ext}
    ext_modules = [
        Extension("borg.bregman", ["borg/bregman.pyx"]),
        Extension("borg.models", ["borg/models.pyx"]),
        Extension("borg.planners", ["borg/planners.pyx"]),
        Extension("borg.statistics", ["borg/statistics.pyx"]),
        Extension("borg.domains.max_sat.features", ["borg/domains/max_sat/features.pyx"]),
        Extension("borg.domains.max_sat.instance", ["borg/domains/max_sat/instance.pyx"]),
        Extension("borg.domains.pb.features", ["borg/domains/pb/features.pyx"]),
        Extension("borg.domains.pb.instance", ["borg/domains/pb/instance.pyx"]),
        Extension("borg.domains.sat.features", ["borg/domains/sat/features.pyx"]),
        Extension("borg.domains.sat.instance", ["borg/domains/sat/instance.pyx"]),
        Extension("borg.test.test_statistics_c", ["borg/test/test_statistics_c.pyx"])]

with open("requirements.txt") as file_:
    requires = [line for line in file_.readlines() if not line.startswith("git+")]

setup(
    name = "borg",
    version = "2012.4.01",
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    install_requires = requires,
    author = "Bryan Silverthorn",
    author_email = "bsilverthorn@gmail.com",
    description = "the borg algorithm portfolio toolkit",
    license = "MIT",
    keywords = "borg algorithm portfolio solver SAT PB satisfiability",
    url = "http://nn.cs.utexas.edu/pages/research/borg/",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2.6",
        "Operating System :: Unix",
        "Topic :: Utilities",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"])
