from torchvision.datasets import MNIST
from datasig.dataset import CanonicalDataset, Dataset, TorchVisionDataset
from tqdm import tqdm
from time import thread_time

class TimeTester:
    def __init__(self, dataset: Dataset) -> None:
        self.tv_dataset = TorchVisionDataset(dataset)

    def run_test_case(self) -> tuple[float, float]:
        start = thread_time()
        
        _ = CanonicalDataset(self.tv_dataset).fingerprint

        end = thread_time() - start

        return end


if __name__ == "__main__":
    dataset = MNIST(root="/tmp/mnist_data", train=True, download=True)

    mean_time = 0

    t = TimeTester(dataset)

    for i in tqdm(range(10)):
        iter_time = t.run_test_case()
        mean_time += (1 / (i + 1)) * (iter_time - mean_time)
    
    print(f"Mean time: {mean_time}")
