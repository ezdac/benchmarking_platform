# TODO here you define classes, that have to comply to the zope Interface....
# in the cli code, we extract all the classes from here, check them for interface complyance,
# and then use them for fingerprint benchmarking
# TODO also, all parameters that can be provided to the classes, should automatically be settable in the config!
# only classes that are provided in the config, will get searched in the implementations.py file,
# and automatically added to config checker and interface checker, and then calculated
# result files will contain the name (with parameters) of the used fingerprint methods

from abc import ABC, abstractmethod
from fiprihash.fingerprint import FoldingHash, KMeansCHash
from rdkit.Chem import AllChem
from fipribench.utils import fp_vector_to_nparray
import pandas


class HashingMethod(ABC):

    @property
    @classmethod
    @abstractmethod
    def _fingerprinter_class(cls):
        pass

    def __init__(self, bit_size, ecfp_radius, possible_classifications):
        self._bit_size = bit_size
        # TODO provide all parameters from the init
        self._fingerprinter = self._fingerprinter_class(bitsize=bit_size, ecfp_radius=ecfp_radius,
                                                        possible_classifications=possible_classifications)

    @property
    def name(self):
        # FIXME this can only be called after __init; is this safe?
        return self._fingerprinter.name

    def train(self, molecules, classifications):
        mols = []
        for smiles, classification in zip(molecules, classifications):
            mol = AllChem.MolFromSmiles(smiles)
            # FIXME refactor testing assert
            assert classification in (0, 1)
            mol.SetIntProp('classification', classification)
            mols.append(mol)
        # train the fingerprinters internal LuT
        self._fingerprinter.train(mols)

    def calculate_fingerprint(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        rdkit_vect = self._fingerprinter(mol)
        return fp_vector_to_nparray(rdkit_vect)


class KMeans(HashingMethod):
    _fingerprinter_class = KMeansCHash


class Folding(HashingMethod):
    _fingerprinter_class = FoldingHash
