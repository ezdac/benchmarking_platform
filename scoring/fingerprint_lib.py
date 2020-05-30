#
# $Id$
#
# module to calculate a fingerprint from SMILES

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

# implemented fingerprints:
# ECFC0 (ecfc0), ECFP0 (ecfp0), MACCS (maccs),
# atom pairs (ap), atom pairs bit vector (apbv), topological torsions (tt)
# hashed atom pairs (hashap), hashed topological torsions (hashtt) --> with 1024 bits
# ECFP4 (ecfp4), ECFP6 (ecfp6), ECFC4 (ecfc4), ECFC6 (ecfc6) --> with 1024 bits
# FCFP4 (fcfp4), FCFP6 (fcfp6), FCFC4 (fcfc4), FCFC6 (fcfc6) --> with 1024 bits
# Avalon (avalon) --> with 1024 bits
# long Avalon (laval) --> with 16384 bits
# long ECFP4 (lecfp4), long ECFP6 (lecfp6), long FCFP4 (lfcfp4), long FCFP6 (lfcfp6) --> with 16384 bits
# RDKit with path length = 5 (rdk5), with path length = 6 (rdk6), with path length = 7 (rdk7)
# 2D pharmacophore (pharm) ?????????????

nbits = 1024
longbits = 16384

# dictionary
fpdict = {}
fpdict["ecfp0"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpdict["ecfp2"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpdict["ecfp4"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpdict["ecfp6"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpdict["ecfc0"] = lambda m: AllChem.GetMorganFingerprint(m, 0)
fpdict["ecfc2"] = lambda m: AllChem.GetMorganFingerprint(m, 1)
fpdict["ecfc4"] = lambda m: AllChem.GetMorganFingerprint(m, 2)
fpdict["ecfc6"] = lambda m: AllChem.GetMorganFingerprint(m, 3)
fpdict["fcfp2"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 1, useFeatures=True, nBits=nbits
)
fpdict["fcfp4"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 2, useFeatures=True, nBits=nbits
)
fpdict["fcfp6"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 3, useFeatures=True, nBits=nbits
)
fpdict["fcfc2"] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
fpdict["fcfc4"] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
fpdict["fcfc6"] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
fpdict["lecfp4"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpdict["lecfp6"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpdict["lfcfp4"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 2, useFeatures=True, nBits=longbits
)
fpdict["lfcfp6"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 3, useFeatures=True, nBits=longbits
)
fpdict["maccs"] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpdict["ap"] = lambda m: Pairs.GetAtomPairFingerprint(m)
fpdict["tt"] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
fpdict["hashap"] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
    m, nBits=nbits
)
fpdict[
    "hashtt"
] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
    m, nBits=nbits
)
fpdict["avalon"] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpdict["laval"] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpdict["rdk5"] = lambda m: Chem.RDKFingerprint(
    m, maxPath=5, fpSize=nbits, nBitsPerHash=2
)
fpdict["rdk6"] = lambda m: Chem.RDKFingerprint(
    m, maxPath=6, fpSize=nbits, nBitsPerHash=2
)
fpdict["rdk7"] = lambda m: Chem.RDKFingerprint(
    m, maxPath=7, fpSize=nbits, nBitsPerHash=2
)


# TODO add trainable fps
fp_train_dict = {}

import os
import zerorpc
import collections


def convert_utf8(data):
    if isinstance(data, basestring):
        return unicode(data, "UTF-8")
    elif isinstance(data, collections.Mapping):
        return dict(map(convert_utf8, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert_utf8, data))
    else:
        return data


class RemoteFingerprinter:
    def __init__(self, client, uid):
        self.remote_fingerprinter = client
        self.id_ = uid

    def get_fingerprint_params(self):
        # CHECKME coding necessary?
        return self.remote_fingerprinter.get_fingerprint_params(self.id_)

    def hash(self, molecule_smiles):
        # returns e.g. {'bit_size':1024, 'on_bits'=[123,342]}
        val = self.remote_fingerprinter.hash(self.id_, molecule_smiles)
        bit_size = int(val["bit_size"])
        on_bits = val["on_bits"]
        bit_vector = DataStructs.ExplicitBitVect(bit_size)
        bit_vector.SetBitsFromList(on_bits)
        return bit_vector

    @classmethod
    def create_remote(
        cls, client, fingerprinter_class, training_smiles, uid=None, **init_kwargs
    ):
        kwargs = convert_utf8(init_kwargs)
        fingerprinter_class = convert_utf8(fingerprinter_class)
        training_classes = convert_utf8(["active"] * len(training_smiles))

        print "Sending init to remote fingerprinter"
        uid = client.init_trained_fingerprinter(
            fingerprinter_class, training_smiles, training_classes, kwargs
        )
        return cls(client, uid)


def make_series_fingerprinter(seed_fp, k_values):

    clustered_region_size = int(
        seed_fp.get_fingerprint_params()["clustered_region_size"]
    )
    k_values = [k for k in sorted(k_values) if k < clustered_region_size]
    k_values = k_values.reverse()

    k_deltas = []
    current_k = clustered_region_size
    for val in k_values:
        delta = current_k - val
        k_deltas.append(delta)
        current_k = val
    series_uids = client.make_series_from_seed(seed_fp.id_, k_deltas)

    fingerprinter = [RemoteFingerprinter(client, uid) for uid in series_uids]

    # map k values to remote fingerprinter
    dic = {k: fp for k, fp in zip(k_values, fingerprinter)}
    dic[clustered_region_size] = seed_fp
    return dic


FIPRI_SERVER_PORT = os.environ.get("FIPRI_SERVER_PORT")
FIPRI_SERVER_HOST = os.environ.get("FIPRI_SERVER_HOST")

trainable_fps = {}


class DummyFingerprinter:
    def __init__(self, ecfp_radius, nbits):
        self.ecfp_radius = ecfp_radius
        self.nbits = nbits

    def hash(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError("SMILES cannot be converted to a RDKit molecules:", smiles)
        return AllChem.GetMorganFingerprintAsBitVect(
            m, self.ecfp_radius, nBits=self.nbits
        )


def query_remote_bit_size_and_construct_ecfp(remote_fingerprinter, ecfp_radius):
    fp_params = remote_fingerprinter.get_fingerprint_params()
    print (fp_params)
    assert (
        int(fp_params["ecfp_radius"]) == ecfp_radius
    ), "Queried and remote ecfp radii don't match"
    nbits = int(fp_params["bit_size"])
    return DummyFingerprinter(ecfp_radius, nbits)


if FIPRI_SERVER_PORT and FIPRI_SERVER_PORT:
    client = zerorpc.Client(timeout=None, passive_heartbeat=True)
    client.connect("tcp://{}:{}".format(FIPRI_SERVER_HOST, FIPRI_SERVER_PORT))
    trainable_fps["optics4_1"] = lambda training_smiles: RemoteFingerprinter(
        client,
        "optics",
        training_smiles,
        folded_region_expansion_factor=1.0,
        ecfp_radius=2,
    )
    trainable_fps["optics6_1"] = lambda training_smiles: RemoteFingerprinter(
        client,
        "optics",
        training_smiles,
        folded_region_expansion_factor=1.0,
        ecfp_radius=3,
    )
    trainable_fps[
        "ecfp4_1_dyn"
    ] = lambda training_smiles: query_remote_bit_size_and_construct_ecfp(
        RemoteFingerprinter(
            client,
            "optics",
            training_smiles,
            folded_region_expansion_factor=1.0,
            ecfp_radius=2,
        ),
        2,
    )
    trainable_fps[
        "ecfp6_1_dyn"
    ] = lambda training_smiles: query_remote_bit_size_and_construct_ecfp(
        RemoteFingerprinter(
            client,
            "optics",
            training_smiles,
            folded_region_expansion_factor=1.0,
            ecfp_radius=3,
        ),
        3,
    )


def TrainFP(fp_name, all_train_smiles):
    """
    Only when this is called from the script with all the training smiles strings,
    it will be accessible by CalculateFP!
    This should be called before every change in dataset, because it constructs a trained fingerprinter
    for this dataset and makes it accessible from the dict that CalculateFP uses!
    """

    fingerprinters = []

    # fiprihash_corex_2_1024_series_128
    # would be corex initialized with 1024 bit,
    # and then varied k (nof_clusters) in range(0,<nof initial clusters>,128) intervals.
    elements = fp_name.split("_")
    # fiprihash prefix is required
    if elements[0] != "fiprihash":
        # we don't have to train this, it's not trainable
        return

    fingerprinter_type = elements[1]
    ecfp_radius = int(elements[2])
    bit_size = int(elements[3])

    if ecfp_radius > 10:
        raise ValueError("Unreasonable ecfp size!")
    # train the fingerprinter:

    print (
        "Training fingerprinter '{}' with {} molecules".format(
            fp_name, len(all_train_smiles)
        )
    )
    seed_fp = RemoteFingerprinter.create_remote(
        client,
        fingerprinter_type,
        all_train_smiles,
        bit_size=bit_size,
        ecfp_radius=ecfp_radius,
    )

    base_name = "_".join(elements[1:4])

    if len(elements) > 4:
        # this is a series fingerprint
        if elements[4] == "series":
            interval_step = elements[4]
            k_values = range(0, bit_size + 1, interval_step)
            series_fps = make_series_fingerprinter(seed_fp, k_values)

            names = []
            for k_val, fp in series_fps.items():
                name = base_name + "_" + k_val
                fpdict[name] = fp.hash
                names.append(name)
            return names

    else:
        fpdict[base_name] = seed_fp.hash
        return [base_name]


def CalculateFP(fp_name, smiles):
    m = smiles
    # only convert to a rdkit molecule when we don't call a remote method,
    # since the mols can't be serialized easily
    if not fp_name in trainable_fps.keys():
        m = Chem.MolFromSmiles(m)
        if m is None:
            raise ValueError("SMILES cannot be converted to a RDKit molecules:", smiles)
    return fpdict[fp_name](m)
