import setuptools
import setuptools.extension

try:
    import Cython.Distutils
except ImportError:
    cmdclass = {}
    ext_modules = None
else:
    cmdclass = {"build_ext": Cython.Distutils.build_ext}
    ext_modules = [
        setuptools.extension.Extension("borg.bregman", ["src/python/borg/bregman.pyx"]),
        setuptools.extension.Extension("borg.models", ["src/python/borg/models.pyx"]),
        setuptools.extension.Extension("borg.planners", ["src/python/borg/planners.pyx"]),
        setuptools.extension.Extension("borg.statistics", ["src/python/borg/statistics.pyx"]),
        setuptools.extension.Extension("borg.domains.max_sat.features", ["src/python/borg/domains/max_sat/features.pyx"]),
        setuptools.extension.Extension("borg.domains.max_sat.instance", ["src/python/borg/domains/max_sat/instance.pyx"]),
        setuptools.extension.Extension("borg.domains.pb.features", ["src/python/borg/domains/pb/features.pyx"]),
        setuptools.extension.Extension("borg.domains.pb.instance", ["src/python/borg/domains/pb/instance.pyx"]),
        setuptools.extension.Extension("borg.domains.sat.features", ["src/python/borg/domains/sat/features.pyx"]),
        setuptools.extension.Extension("borg.domains.sat.instance", ["src/python/borg/domains/sat/instance.pyx"]),
        setuptools.extension.Extension("borg.test.test_statistics_c", ["src/python/borg/test/test_statistics_c.pyx"]),
        ]

setuptools.setup(
    name = "borg",
    version = "2012.4.01",
    packages = setuptools.find_packages("src/python"),
    package_dir = {"": "src/python"},
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    install_requires = [
        "Cython>=0.15.1",
        "numpy>=1.6.1",
        "plac>=0.9.0",
        "scipy>=0.10.0",
        ],
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )

