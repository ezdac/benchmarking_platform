#
# $Id$
#
# module to calculate a fingerprint from SMILES

import subprocess
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

# FIXME don't hardcode this
FIPRIHASH_PATH = '/Users/Max/coding/master/main.py'
PYTHON3_VERSION = '/Users/Max/.pyenv/versions/miniconda3-latest/envs/masterthesis/bin/python3'

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


# TODO pass bitsize from the run config
nbits = 2048
# TODO add longbits option to run config
longbits = 16384

# dictionary
fpdict = {}
fpdict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpdict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpdict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpdict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpdict['ecfc0'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
fpdict['ecfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
fpdict['ecfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
fpdict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpdict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpdict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpdict['fcfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
fpdict['fcfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
fpdict['fcfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
fpdict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpdict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpdict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpdict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpdict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpdict['ap'] = lambda m: Pairs.GetAtomPairFingerprint(m)
fpdict['tt'] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
fpdict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpdict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpdict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpdict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpdict['mh6432'] = lambda m: minhash(m, bitsize=nbits, radius=2)


def minhash(mol, bitsize, radius):
    smiles = Chem.MolToSmiles(mol)
    # TODO this is VERY slow, repeat(7, 10) with this function takes 5.34851908684, while the inprocess python3
    # takes 0.00707451700003503, as calculated by this snippet:
    # https://stackoverflow.com/questions/8220801/how-to-use-timeit-module

    # TODO port the scoring part to python3, if too much work use the other parts with python2
    output = subprocess.check_output([PYTHON3_VERSION, FIPRIHASH_PATH, str(smiles), '--radius', str(radius),
                                      '--bitsize', str(bitsize), '--serialize'])
    return UIntSparseIntVect(output)


def CalculateFP(fp_name, smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('SMILES cannot be converted to a RDKit molecules:', smiles)

    return fpdict[fp_name](m)


def getFPDict(fp_names, smiles):
    '''Gets the fingerprints from the fingerprint library
    and stores them in a dictioanry'''
    fp_dict = {}
    for fp in fp_names:
        fp_dict[fp] = fingerprint_lib.CalculateFP(fp, smiles)
    return fp_dict


def printFPs(fps, fpname):
    '''Prints a list of fingerprints'''
    print("-------------------------------")
    print("FUSION DONE FOR:")
    for fp in fps:
        print(fp,)
    print("")
    print("Name of fusion:", fpname)
    print("-------------------------------")


def checkFPFile(filepath):
    '''Checks if file containing fingerprint names exists
    and reads the fingerprints'''
    try:
        myfile = open(filepath, 'r')
    except:
        raise IOError('file does not exist:', filepath)
    else:
        fp_names = []
        for line in myfile:
            line = line.rstrip().split()
            fp_names.append(line[0])
        return fp_names


def readFPs(filepath):
    '''Reads a list of fingerprints from a file'''
    try:
        myfile = open(filepath, 'r')
    except:
        raise IOError('file does not exist:', filepath)
    else:
        fps = []
        for line in myfile:
            if line[0] != "#": # ignore comments
                line = line.rstrip().split()
                fps.append(line[0])
        return fps


def getName(fp, fp_names):
    '''Determines the new name of a fingerprint in case
    multiple fingerprints with the same name'''
    # check if fp already exists. if yes, add a number
    if fp in fp_names:
        suffix = 2
        tmp_name = fp+'_'+str(suffix)
        while tmp_name in fp_names:
            suffix += 1
            tmp_name = fp+'_'+str(suffix)
        return tmp_name
    else:
        return fp