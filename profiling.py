from torchvision.datasets import MNIST
from datasig.config import ConfigV0
from datasig.dataset import CanonicalDataset, Dataset, TorchVisionDataset, DatasetFingerprint
import cProfile
import argparse
import pstats
import typing as t


class FingerprintProfile:
    """Abtract class for profiling dataset fingerprinting on a given dataset"""

    def __init__(self, config=ConfigV0):
        self.config = config
        self.canonical: CanonicalDataset = None
        self.profile = cProfile.Profile()
        self.dataset: Dataset = None

    def _setup(self) -> None:
        """Method that needs to be overridden by child classes to set self.dataset"""
        raise NotImplemented("Need to be implemented in child class")

    def _to_canonical(self, profile=True):
        if profile:
            self.profile.enable()
        self.canonical = CanonicalDataset(self.dataset, config=self.config)
        self.profile.disable()

    def _run_profile(self, uid=True, fingerprint=True):
        self.profile.enable()
        if uid:
            _ = self.canonical.uid
        if fingerprint:
            _ = self.canonical.fingerprint
        self.profile.disable()

    def __call__(
        self,
        profile_canonical: bool = True,
        profile_uid: bool = True,
        profile_fingerprint: bool = True,
    ) -> cProfile.Profile:
        self._setup()
        self._to_canonical(profile=profile_canonical)
        self._run_profile(uid=profile_uid, fingerprint=profile_fingerprint)
        return self.profile

def torch_mnist_v0(prop: t.Callable[[CanonicalDataset], DatasetFingerprint]) -> t.Callable:
    class TorchMNISTV0(FingerprintProfile):
        def _setup(self):
            train_data = MNIST(root="/tmp/mnist_data", train=True, download=True)
            self.dataset = TorchVisionDataset(train_data)

        def _run_profile(self, uid=True, fingerprint=True):
            self.profile.enable()
            if uid:
                _ = self.canonical.uid
            if fingerprint:
                _ = prop(self.canonical)
            self.profile.disable()
    
    return TorchMNISTV0


def print_profile(profile: cProfile.Profile):
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(profile).sort_stats(sortby)
    ps.print_stats()


def main(args):
    # Keys are (Target, datasig config version)
    target_map = {
        ("torch_mnist_keyed_sha", "0"): torch_mnist_v0(lambda x: x.fingerprint),
        ("torch_mnist_xor", "0"): torch_mnist_v0(lambda x: x.xor_fingerprint),
        ("torch_mnist_single_sha", "0"): torch_mnist_v0(lambda x: x.single_fingerprint),
        ("torch_mnist_datasketch", "0"): torch_mnist_v0(lambda x: x.datasketch_fingerprint),
    }

    for target in args.targets:
        profile = target_map[(target, args.version)]()(
            profile_canonical=args.canonical,
            profile_uid=args.uid,
            profile_fingerprint=args.fingerprint,
        )
        print(f"#### Pofile for {target} - datasig v{args.version}\n")
        print_profile(profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*")
    parser.add_argument(
        "--canonical", action="store_true", help="Include dataset canonization in stats"
    )
    parser.add_argument("--uid", action="store_true", help="Include UID computation in stats")
    parser.add_argument(
        "--fingerprint", action="store_true", help="Include fingerprint computation in stats"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include all canonization, UID, and fingerprint computation in stats",
    )
    parser.add_argument(
        "-v", "--version", default="0", help="Datasig config version to use. Defaults to v0"
    )
    args = parser.parse_args()
    if args.full:
        args.canonical = True
        args.uid = True
        args.fingerprint = True

    main(args)
