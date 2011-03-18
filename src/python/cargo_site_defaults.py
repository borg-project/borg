"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

# we add a Memory requirement to prevent Condor from adding an automatic
# Memory >= ImageSize constraint; its ImageSize detection is poor.

condor_matching = \
    "InMastodon" \
    " && (Arch == \"X86_64\")" \
    " && (OpSys == \"LINUX\")" \
    " && regexp(\"rhavan-.*\", ParallelSchedulingGroup)" \
    " && (Memory > 1024)"

