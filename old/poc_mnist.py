import hashlib
import torch
import numpy as np
from io import BytesIO
from torch.utils.data import DataLoader
from torch.nn import Module
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.transforms import v2

magic_values = np.zeros((400,), dtype="|S32")
for i in range(400):
    magic_values[i] = np.random.bytes(32)


def tensor_to_bytes(x: torch.Tensor) -> bytes:
    buff = BytesIO()
    torch.save(x, buff)
    buff.seek(0)
    return buff.read()


class Shingling(Module):
    def forward(self, x: torch.Tensor) -> bytes:
        return hashlib.sha256(tensor_to_bytes(x)).digest()


class LSH(Module):
    def forward(self, x: bytes) -> np.ndarray:
        res = np.zeros((400,), dtype="|S32")
        for i in range(400):
            res[i] = hashlib.sha256(x + magic_values[i]).digest()
        return res


transforms = v2.Compose(
    [
        v2.ToTensor(),
        Shingling(),
        LSH(),
    ]
)

data = MNIST(root="./data", train=True, download=True, transform=transforms)
test = MNIST(root="./data", train=False, download=True, transform=transforms)


def min_hash(dataset: torch.utils.data.Dataset) -> np.ndarray:
    m = None
    for x, _ in dataset:
        if m is None:
            m = x

        for i in range(400):
            x_i = int.from_bytes(x[i])
            m_i = int.from_bytes(m[i])
            if x_i < m_i:
                m[i] = x[i]

    return m


mnist_sig = min_hash(data)


def half(data):
    def gen():
        i = 0
        while True:
            try:
                yield data[2 * i]
                i += 1
            except:
                break

    return gen()


def offset(data, offset):
    def gen():
        i = 0
        while True:
            try:
                yield data[i + offset]
                i += 1
            except:
                break

    return gen()


# half_mnist_sig = min_hash(half(data))
# offset_mnist_sig = min_hash(offset(data, 100))

# print((mnist_sig == half_mnist_sig).sum() / 400)
# print((mnist_sig == offset_mnist_sig).sum() / 400)


def concat(data_a, data_b):
    def gen():
        i = 0
        while True:
            try:
                yield data_a[i]
                i += 1
            except:
                break
        i = 0
        while True:
            try:
                yield data_b[i]
                i += 1
            except:
                break

    return gen()


full_mnist_sig = min_hash(concat(data, test))
test_mnist_sig = min_hash(test)

print((mnist_sig == full_mnist_sig).sum() / 400)
print((mnist_sig == test_mnist_sig).sum() / 400)
