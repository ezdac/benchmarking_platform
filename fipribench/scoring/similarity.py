# dictionary for similarity measures
from rdkit import DataStructs


simil_dict = {}
simil_dict['Dice'] = lambda x, y: sorted(DataStructs.BulkDiceSimilarity(x, y), reverse=True)
simil_dict['Tanimoto'] = lambda x, y: sorted(DataStructs.BulkTanimotoSimilarity(x, y), reverse=True)
simil_dict['Cosine'] = lambda x, y: sorted(DataStructs.BulkCosineSimilarity(x, y), reverse=True)
simil_dict['Russel'] = lambda x, y: sorted(DataStructs.BulkRusselSimilarity(x, y), reverse=True)
simil_dict['Kulczynski'] = lambda x, y: sorted(DataStructs.BulkKulczynskiSimilarity(x, y), reverse=True)
simil_dict['McConnaughey'] = lambda x, y: sorted(DataStructs.BulkMcConnaugheySimilarity(x, y), reverse=True)
simil_dict['Manhattan'] = lambda x, y: sorted(DataStructs.BulkAllBitSimilarity(x, y), reverse=True)
simil_dict['RogotGoldberg'] = lambda x, y: sorted(DataStructs.BulkRogotGoldbergSimilarity(x, y), reverse=True)


def getBulkSimilarity(fp, fp_list, simil):
    '''Calculate the bulk similarity for a given list of fingerprints'''
    return simil_dict[simil](fp, fp_list)


# TODO remove, this should be done in cfg parsing time
def checkSimil(simil):
    '''Checks if the chosen similarity metric is supported'''
    simil_list = ['Dice', 'Tanimoto', 'Cosine', 'Russel', 'Kulczynski', 'McConnaughey', 'Manhattan', 'RogotGoldberg']
    if simil not in simil_list:
        raise ValueError('provided similarity metric not supported:', simil)
