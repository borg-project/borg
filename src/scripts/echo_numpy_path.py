try:
    import numpy

    from os.path import dirname

    print dirname(numpy.__file__)
except:
    raise SystemExit(1)

