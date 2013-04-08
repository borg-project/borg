def make(name, workers=0):
    distributors = {
        "condor": CondorDistributor,
        "ipython": IPythonDistributor}

    return distributors[name](workers=workers)


class CondorDistributor(object):
    def __init__(self, workers):
        self._workers = workers

    def do(self, tasks):
        import condor

        condor.defaults.condor_matching = (
            "InMastodon"
            " && regexp(\"rhavan-.*\", ParallelSchedulingGroup)"
            " && (Arch == \"X86_64\")"
            " && (OpSys == \"LINUX\")"
            " && (Memory > 1024)")

        return condor.do(tasks, workers=self._workers)


class IPythonDistributor(object):
    def __init__(self, workers):
        # TODO apply the workers parameter
        self._workers = workers

    def do(self, tasks):
        import IPython.parallel

        client = IPython.parallel.Client()
        view = client.load_balanced_view()

        return view.map(self._springboard, list(tasks))

    @staticmethod
    def _springboard((function, function_args)):
        return function(*function_args)
