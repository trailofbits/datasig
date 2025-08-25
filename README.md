# Dataset fingerprinting
This repository contains our proof-of-concept for fingerprinting a dataset.

## Local installation
```python
git clone https://github.com/trailofbits/datasig && cd datasig
uv sync
```

## Usage
### Fingerprinting
The code below shows experimental usage of the library.
This will be subject to frequent changes in early development stages. 

```python
from torchvision.datasets import MNIST
from datasig.dataset import TorchVisionDataset
from datasig.algo import KeyedShaMinHash, UID

torch_dataset = MNIST(root="/tmp/data", train=True, download=True)

# Wrap the dataset with one of the classes in `datasig.dataset`.
# These classes provide a uniform interface to access serialized data points.
dataset = TorchVisionDataset(torch_dataset)

# Pass the dataset to the fingerprinting algorithm.
print("Dataset UID: ", UID(dataset).digest())
print("Dataset fingerprint: ", KeyedShaMinHash(dataset).digest())
```

### Manual serialization/deserialization
The dataset classes defined in `datasig.dataset` provide static serialization
and deserialization to convert datapoints between their usual representation
and bytes.

```python
from torchvision.datasets import MNIST
from datasig.dataset import TorchVisionDataset

torch_dataset = MNIST(root="/tmp/data", train=True, download=True)

# Serializing data points to bytes
serialized = TorchVisionDataset.serialize_data_point(torch_dataset[0])

# Deserializing data points from bytes
deserialized = TorchVisionDataset.deserialize_data_point(serialized)
```

## Development
### Unit tests
Tests are in the `datasig/test` directory. You can run the tests with:

```bash
uv run python -m pytest # Run all tests
uv run python -m pytest -s datasig/test/test_csv.py # Run only one test file
uv run python -m pytest -s datasig/test/test_csv.py -k test_similarity # Run only one specific test function
```

### Profiling
The profiling script generates a profile for dataset processing and fingerprint generation using cProfile. To profile the MNIST dataset from the torch framework,
you can run:

```bash
uv run python profiling.py torch_mnist --full
```

The `--full` argument tells the script to include dataset canonization, UID generation, and fingerprint generation in the profile. If you want to profile only some of these steps you can cherry pick by using or omitting the following arguments instead:

```bash
uv run python profiling.py torch_mnist --canonical --uid --fingerprint
```

You can optionally specify the datasig config version to use (at the time of writing we have only v0) with:  

```bash
uv run python profiling.py torch_mnist -v 0 --all
```

Currently we support only one target dataset: `torch_mnist`. To add another dataset, add a class in `profiling.py` similar to `TorchMNISTV0`, that implements the `_setup()` method which is responsible for loading the dataset.

### Benchmarking

!!! This is currently broken !!!

Datasig has a built-in `benchmark` module that allows to run experiments to benchmark speed and accuracy of various fingerprinting methods with varying configurations and on several datasets.

Benchmarks are configured programmatically using the `datasig` library directly.
The `benchmarking.py` script gives a comprehensive overview of how to configure and run a benchmark, export results, as well as plot them on graph.

You can run the example benchmark with

```bash
uv run python benchmarking.py
```