try:
    import os.path
    import numpy

    print os.path.dirname(numpy.__file__)
except:
    raise SystemExit(1)

