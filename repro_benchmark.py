import os
import resource
import sys
import time

import ROOT
import numpy as np
import torch
from loader import load_tree


# verbosity = ROOT.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), 1000)

ttree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data/SkimmedTree_UpdatednSigmaAndExpdEdx_LHC24ar_apass2_TPCSignal_50NN.root")
FEATURES = ["fFt0Occ", "fHadronicRate", "fInvDeDxExpTPC", "fMass", "fNormMultTPC",
            "fNormNClustersTPC", "fPhi", "fSigned1Pt", "fTPCInnerParam", "fTgl"]
TARGETS = ["fTPCSignal"]
COLS = FEATURES + TARGETS
BATCH_SIZE = 262144
EPOCHS = 10

which = sys.argv[1] if len(sys.argv) > 1 else "both"
peak_rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
baseline_rss = peak_rss()
results = []

# ----------------------------------------------------------------- uproot
if which in ("both", "uproot"):
    t0 = time.perf_counter()
    _, array = load_tree(num_workers=1).load(
        ttree_path,
        use_vars=FEATURES + TARGETS,
        key="O2tpcskimv0tree",
        load_latest=True,
        dtype=np.float32,
    )

    tensor = torch.from_numpy(array)
    uproot_setup = time.perf_counter() - t0

    uproot_epochs = []
    for epoch in range(EPOCHS):
        t = time.perf_counter()
        for begin in range(0, len(tensor), BATCH_SIZE):
            batch = tensor[begin:begin + BATCH_SIZE]
            x = batch[:, :len(FEATURES)].contiguous()
            y = batch[:, len(FEATURES):].contiguous()
        uproot_epochs.append(time.perf_counter() - t)
    results.append(("uproot (in memory)", uproot_setup, uproot_epochs, peak_rss()))

# ----------------------------------------------------------------- RDataLoader eager
if which in ("both", "rdataloader"):
    # ROOT.EnableImplicitMT(8)

    t0 = time.perf_counter()
    rdf = ROOT.RDataFrame("O2tpcskimv0tree", ttree_path)

    loader = ROOT.Experimental.ML.RDataLoader(
        rdf,
        columns=FEATURES + TARGETS,
        target=TARGETS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_remainder=False,
        load_eager=True,
    )
    rdl_setup = time.perf_counter() - t0  # the eager read itself happens inside epoch 1

    rdl_epochs = []
    for epoch in range(EPOCHS):
        t = time.perf_counter()
        for x, y in loader.as_torch():
            ...
        rdl_epochs.append(time.perf_counter() - t)
    results.append(("RDataLoader eager", rdl_setup, rdl_epochs, peak_rss()))

# ----------------------------------------------------------------- summary
print(f"\nROOT {ROOT.gROOT.GetVersion()} from {ROOT.__file__}")
print(f"{7256529} rows x {len(COLS)} cols, batch {BATCH_SIZE}, {EPOCHS} epochs, baseline RSS {baseline_rss:.2f} GiB")
print(f"{'':22s} {'setup':>8s} " + " ".join(f"{'ep' + str(i + 1):>8s}" for i in range(EPOCHS))
      + f" {'total':>8s} {'peakRSS':>8s}")
for name, setup, epochs, rss in results:
    print(f"{name:22s} {setup:8.2f} " + " ".join(f"{e:8.2f}" for e in epochs)
          + f" {setup + sum(epochs):8.2f} {rss:7.2f}G")
if which == "both":
    print("note: peak RSS is cumulative for the process; run the phases separately for per-loader RSS")

"""
                          setup      ep1      ep2      ep3      ep4      ep5      ep6      ep7      ep8      ep9     ep10    total  peakRSS
uproot (in memory)         2.05     0.05     0.05     0.05     0.05     0.06     0.05     0.05     0.05     0.05     0.05     2.56    0.79G
RDataLoader eager          1.05     3.82     0.04     0.04     0.04     0.04     0.03     0.04     0.03     0.04     0.04     5.19    1.51G
"""
