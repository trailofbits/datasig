# Dataset fingerprinting
This repository contains our proof-of-concept for fingerprinting a dataset.

## Local installation
```python
git clone https://github.com/trailofbits/datasig && cd datasig
python3 -m pip install .
```

## Usage
The code below shows experimental usage of the library.
This will be subject to frequent changes in early development stages. 

```python
from torchvision.datasets import MNIST
from datasig.dataset import TorchVisionDataset, CanonicalDataset

torch_dataset = MNIST(root="/tmp/data", train=True, download=True)
dataset = TorchVisionDataset(torch_dataset)
canonical = CanonicalDataset(dataset)
print("Dataset UID: ", canonical.uid)
print("Dataset fingerprint: ", canonical.fingerprint)
```

## Development
### Unit tests
Tests are in the `datasig/test` directory. You can run the tests with:

```bash
python3 -m pytest # Run all tests
python3 -m pytest -s datasig/test/test_csv.py # Run only one test file
python3 -m pytest -s datasig/test/test_csv.py -k test_similarity # Run only one specific test function
```

### Profiling
The profiling script generates a profile for dataset processing and fingerprint generation using cProfile. To profile the MNIST dataset from the torch framework,
you can run:

```bash
python3 profiling.py torch_mnist --full
```

The `--full` argument tells the script to include dataset canonization, UID generation, and fingerprint generation in the profile. If you want to profile only some of these steps you can cherry pick by using or omitting the following arguments instead:

```bash
python3 profiling.py torch_mnist --canonical --uid --fingerprint
```

You can optionally specify the datasig config version to use (at the time of writing we have only v0) with:  

```bash
python3 profiling.py torch_mnist -v 0 --all
```

Currently we support only one target dataset: `torch_mnist`. To add another dataset, add a class in `profiling.py` similar to `TorchMNISTV0`, that implements the `_setup()` method which is responsible for loading the dataset.

### Benchmarking
Datasig has a built-in `benchmark` module that allows to run experiments to benchmark speed and accuracy of various fingerprinting methods with varying configurations and on several datasets.

Benchmarks are configured programmatically using the `datasig` library directly.
The `benchmarking.py` script gives a comprehensive overview of how to configure and run a benchmark, export results, as well as plot them on graph.

You can run the example benchmark with

```bash
python3 benchmarking.py
```