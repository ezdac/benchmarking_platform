import gzip
import pickle
from collections import namedtuple
from fipribench.globals import INPATH_CMP, INPATH_LIST
import numpy as np
import pandas as pd

MoleculeData = namedtuple('MoleculeData', 'internal_id external_id smiles')


class DataSet:
    """
    Represents the dataset the scoring is performed over
    TODO make mapping dynamic, lookup path for dataset identifier
    """

    def __init__(self):
        pass

    def get_data(self, path) -> 'MoleculeData':
        lines = gzip.open(path, 'rt', encoding='utf-8')
        for line in lines:
            # TODO error handling/ filtering of other lines
            if line[0] == '#':
                continue
            else:
                # structure of line: [external ID, internal ID, SMILES]]
                line = line.rstrip().split()
            yield MoleculeData(*line[:3])

    def get_actives(self, dataset, target):
        path = INPATH_CMP.joinpath(f'{dataset}/cmp_list_{dataset}_{str(target)}_actives.dat.gz',)
        return self.get_data(path)

    def get_decoys(self, dataset_name, target_name):
        if dataset_name is 'ChEMBL':
            # FIXME edgecase, here this is not intended to be called target specific
            # if neccessary, defer to this class and save in memory (if feasible)
            path = INPATH_CMP.joinpath(f'{dataset_name}/cmp_list_{dataset_name}_zinc_decoys.dat.gz')
        else:
            path = INPATH_CMP.joinpath(f'{dataset_name}/cmp_list_{dataset_name}_{str(target_name)}_decoys.dat.gz')
        return self.get_data(path)

    def get_dataframe(self, dataset_name, target_name, num_query_mols):
        """
        returns a DataFrame with all relevant information
        DataFrame will look like this:
            smiles    internal_id     external_id  active  is_training  is_fp_training
            O=C(O)c1c(O)cc([N+](=O)[O-])cc1[N+](=O)[O-] CHEMBL447810 ChEMBL_15_A_1 True False True
        """
        training_list = list(self.get_training_list(dataset_name, target_name, num_query_mols))
        actives = list(self.get_actives(dataset_name, target_name))
        decoys = list(self.get_decoys(dataset_name, target_name))

        data = dict()
        data['smiles'] = np.array([mol.smiles for mol in actives + decoys])
        data['internal_id'] = np.array([mol.internal_id for mol in actives + decoys])
        data['external_id'] = np.array([mol.external_id for mol in actives + decoys])
        data['is_active'] = np.concatenate((np.full(len(actives), True), np.full(len(decoys), False)))

        # active/decoy indices are not unique, and are defined per type (active, decoy)
        # this is why we have to separate them
        training_indices_actives = training_list[:num_query_mols]
        training_indices_decoys = training_list[num_query_mols:]

        data['is_training'] = np.concatenate((
                np.isin(np.where(data['is_active'])[0], training_indices_actives),
                np.isin(np.where(np.logical_not(data['is_active']))[0], training_indices_decoys),
            )
        )

        # TODO load from actual list!
        fp_training_list = self.get_fp_training_list(dataset_name, target_name, num_query_mols)
        # for now, just define all molecules as fp-training input
        data['is_fp_training'] = np.full_like(data['is_training'], True)
        return data

    def get_fp_training_list(self, dataset_name, target_name, num_query_mols):
        # TODO
        return []

    def get_training_list(self, dataset_name, target_name, num_query_mols):
        # why didn't they work with internal ids???
        training_input = open(
            INPATH_LIST.joinpath(
                f'{dataset_name}/training_{dataset_name}_{str(target_name)}_{str(num_query_mols)}.pkl'
            ),
            'rb'
        )
        return pickle.load(training_input)
