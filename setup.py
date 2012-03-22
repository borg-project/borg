import setuptools
import setuptools.extension
#import Cython.Distutils

#extensions = [
    #setuptools.extension.Extension(
        #"za.embedded",
        #[
            #"za/embedded.pyx",
            #"za/embedded/goertzel.c",
            #"za/embedded/early_late.c",
            #"za/embedded/fsk4_demapper.c",
            #"za/embedded/fsk_demodulator.c",
            #],
        #extra_compile_args = ["-std=c99"],
        #),
    #]

setuptools.setup(
    name = "borg",
    version = "2012.4.01",
    packages = setuptools.find_packages("src/python"),
    package_dir = {"": "src/python"},
    #cmdclass = {'build_ext': Cython.Distutils.build_ext},
    #ext_modules = extensions,
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

