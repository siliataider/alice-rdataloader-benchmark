# Reproduce

```
git clone https://github.com/siliataider/alice-rdataloader-benchmark.git
cd alice-rdataloader-benchmark
# put the data file at alice-rdataloader-benchmark/data/SkimmedTree_UpdatednSigmaAndExpdEdx_LHC24ar_apass2_TPCSignal_50NN.root.
python3 -m venv .venv
source .venv/bin/activate
pip install root==6.43.1a1 --extra-index-url https://root-experimental-python-wheels.web.cern.ch/
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "uproot>=5.7" awkward xxhash pandas
python repro_benchmark.py
```

## Current results with master
```
                          setup      ep1      ep2      ep3      ep4      ep5      ep6      ep7      ep8      ep9     ep10    total  peakRSS
uproot (in memory)         2.05     0.05     0.05     0.05     0.05     0.06     0.05     0.05     0.05     0.05     0.05     2.56    0.79G
RDataLoader eager          1.05     3.82     0.04     0.04     0.04     0.04     0.03     0.04     0.03     0.04     0.04     5.19    1.51G
```
