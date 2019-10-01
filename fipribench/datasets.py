import gzip
import pickle
from collections import namedtuple
from fipribench.globals import INPATH_CMP, INPATH_LIST


MoleculeData = namedtuple('MoleculeData', 'internal_id external_id smiles')


class DataSet():
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
            if line[0] != '#':
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

    def get_training_list(self, dataset_name, target_name, num_query_mols):
        training_input = open(
            INPATH_LIST.joinpath(
                f'{dataset_name}/training_{dataset_name}_{str(target_name)}_{str(num_query_mols)}.pkl'
            ),
            'rb'
        )
        return pickle.load(training_input)
