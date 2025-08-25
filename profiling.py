from torchvision.datasets import MNIST  # pyright: ignore
from datasig.algo import UID, KeyedShaMinHash
from datasig.dataset import TorchVisionDataset, IterableDataset
import cProfile
import argparse
import pstats
from abc import ABC, abstractmethod


class FingerprintProfile(ABC):
    """Abtract class for profiling dataset fingerprinting on a given dataset"""

    def __init__(self, nb_signatures: int = 400):
        self.profile = cProfile.Profile()
        self.dataset: IterableDataset

    @abstractmethod
    def _setup(self) -> None:
        """Method that needs to be overridden by child classes to set self.dataset"""
        pass

    def _run_profile(
        self, uid: bool = True, fingerprint: bool = True, profile_canonization: bool = True
    ):
        if profile_canonization:
            self.profile.enable()

        if uid:
            uid_alg = UID(self.dataset)

        if fingerprint:
            fingerprint_alg = KeyedShaMinHash(self.dataset)

        if not profile_canonization:
            self.profile.enable()

        if uid:
            _ = uid_alg.digest()  # pyright: ignore
        if fingerprint:
            _ = fingerprint_alg.digest()  # pyright: ignore

        self.profile.disable()

    def __call__(
        self,
        profile_canonization: bool = True,
        profile_uid: bool = True,
        profile_fingerprint: bool = True,
    ) -> cProfile.Profile:
        self._setup()
        self._run_profile(
            uid=profile_uid,
            fingerprint=profile_fingerprint,
            profile_canonization=profile_canonization,
        )
        return self.profile


class TorchMNISTV0(FingerprintProfile):
    def _setup(self):
        train_data = MNIST(root="/tmp/mnist_data", train=True, download=True)
        self.dataset = TorchVisionDataset(train_data)  # pyright: ignore


def print_profile(profile: cProfile.Profile):
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(profile).sort_stats(sortby)
    ps.print_stats()


def main(args: argparse.Namespace):
    # Keys are (Target, datasig config version)
    target_map = {
        ("torch_mnist", "0"): TorchMNISTV0,
    }

    for target in args.targets:
        profile = target_map[(target, args.version)]()(
            profile_canonization=args.canonical,
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
