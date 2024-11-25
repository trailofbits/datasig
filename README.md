# Dataset fingerprinting
This repository contains our proof-of-concept for fingerprinting a dataset.
We currently have tested it on MNIST.

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
