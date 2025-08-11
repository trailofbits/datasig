import datasketch
import hashlib
from .fingerprint import (
    DatasetUID,
    DatasketchFingerprint,
    DatasetFingerprint,
    BasicDatasetFingerprint,
)
from .dataset import Dataset
import multiprocessing as mp
import psutil
from typing import Optional
from .config import ConfigV0


class CanonicalDataset:
    """A format agnostic canonical dataset representation to compute fingerprints"""

    def __init__(self, dataset: Dataset, config=None):
        self.data_point_hashes: List[bytes] = []
        self._uid: Optional[DatasetUID] = None
        # Fingerprints computed with different methods
        self._fingerprints: Dict[str, DatasetFingerprint] = dict()
        self.preprocessed: bool = False
        self.default_config = config if config else ConfigV0()

        # Canonize all data points from underlying dataset
        for d in dataset.encoded_data_points:
            self.add_data_point(d)
        # self._load_all_data_points(dataset)

    @staticmethod
    def _canonize_data_point(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    def _load_all_data_points(self, dataset: Dataset, nthreads: Optional[int] = 1):
        """Load all data points from the underlying dataset"""
        if nthreads is None:
            # Get all available physical cores
            nthreads = psutil.cpu_count(logical=False)
        print(f"USING {nthreads} CPUs")
        hashes = []
        pool = mp.Pool(processes=nthreads)
        for d in dataset.encoded_data_points:
            hashes.append(pool.apply_async(self._canonize_data_point, args=(d,)))
        pool.close()
        pool.join()
        self.data_point_hashes = [r.get() for r in hashes]

        print(f"DEBUG got {len(hashes)} for datapoints {len(dataset)}")
        print(type(self.data_point_hashes[0]))
        print(self.data_point_hashes[0])

    def add_data_point(self, data: bytes) -> None:
        """Compute the hash of the data point and add it to the list."""
        self.data_point_hashes.append(self._canonize_data_point(data))

    def _preprocess(self):
        """Preprocessing steps required before computing fingerprint"""
        # Sort the hash list
        list.sort(self.data_point_hashes)
        self.preprocessed = True

    def _check_preprocess(self, raise_exc=False):
        if not self.preprocessed:
            if raise_exc:
                raise Exception("Canonical dataset must be preprocessed")
            else:
                self._preprocess()

    @property
    def uid(self) -> DatasetUID:
        """Generate dataset unique identifier"""
        self._check_preprocess()
        if self._uid is None:
            # Hash the concatenation of the data points
            hash_function = hashlib.sha256()
            for h in self.data_point_hashes:
                hash_function.update(h)
            self._uid = hash_function.digest()
        return self._uid

    @property
    def fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._fingerprint()

    def _fingerprint(self, config=None) -> DatasetFingerprint:
        """Generate dataset fingerprint using the keyed SHA method.

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by keyed SHA256 functions i.e. random bytes are appended to the input of SHA256.

        See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with k hash functions.
        """
        self._check_preprocess()
        FP_KEY = "KEYED_SHA"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            res = [None] * 400
            for i in range(config.nb_signatures):
                for h in self.data_point_hashes:
                    h2 = hashlib.sha256(h + config.lsh_magic_numbers[i]).digest()
                    if res[i] is None or res[i] > h2:
                        res[i] = h2
            self._fingerprints[FP_KEY] = BasicDatasetFingerprint(res)
        return self._fingerprints[FP_KEY]

    @property
    def xor_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._xor_fingerprint()

    def _xor_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint based on XOR permutations.

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by the output of SHA256 XORed with random 256 bits values.
        """
        self._check_preprocess()
        FP_KEY = "XOR"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            res = [None] * 400
            for i in range(config.nb_signatures):
                for h in self.data_point_hashes:
                    # Bitwise XOR with the magic number
                    h2 = [a ^ b for a, b in zip(h, config.xor_magic_numbers[i])]
                    if res[i] is None or res[i] > h2:
                        res[i] = h2
            res = [bytes(x) for x in res]
            self._fingerprints[FP_KEY] = BasicDatasetFingerprint(res)
        return self._fingerprints[FP_KEY]

    @property
    def single_sha_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._single_sha_fingerprint()

    def _single_sha_fingerprint(self, config=None) -> DatasetFingerprint:
        """Generate dataset fingerprint based on a single SHA permutation

        The figerprint is computed using the MinHash scheme with a single permutation approximated by SHA256.

        See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with one hash functions.
        And https://en.wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function.
        """
        self._check_preprocess()
        FP_KEY = "SINGLE_SHA"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            # Relies on the fact that the data point hashes are sorted
            res = self.data_point_hashes[: config.nb_signatures]
            self._fingerprints[FP_KEY] = BasicDatasetFingerprint(res)
        return self._fingerprints[FP_KEY]

    @property
    def datasketch_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._datasketch_fingerprint()

    def _datasketch_fingerprint(self, config=None) -> DatasetFingerprint:
        """Generate dataset fingerprint based on universal hashing permutations

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by universal hashing.

        See https://ekzhu.com/datasketch/minhash.html.
        We can customize this further using different hash functions and
        number of permutations.
        """
        self._check_preprocess()
        FP_KEY = "DATASKETCH"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            res = datasketch.MinHash(num_perm=config.nb_signatures)
            for d in self.data_point_hashes:
                res.update(d)
            self._fingerprints[FP_KEY] = DatasketchFingerprint(res)
        return self._fingerprints[FP_KEY]