#
# calculates fingerprints and scores lists
# based on the predicted probability
#
# INPUT
# required:
# -n [] : number of query mols
# -f [] : fingerprint to build the logistic regression with
# optional:
# -o [] : relative output path (default: pwd)
# -a : append to the output file (default: overwrite)
# -s [] : similarity metric (default: Dice, 
#         other options: Tanimoto, Cosine, Russel, Kulczynski, 
#         McConnaughey, Manhattan, RogotGoldberg)
# -r [] : file containing the logistic regression info
#          default parameters: penalty='l2', dual=0 (false), C=1.0,
#          fit_intercept=1 (true), intercept_scaling=1.0,
#          class_weight=None, tol=0.0001
# --help : prints usage
#
# OUTPUT: for each target in each data set
#         a file with a list (1 element) of LR prediction
#         per LR prediction: [name, list of 50 scored lists]
#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import fipribench.fingerprint.__init__
import fipribench.scoring.similarity
import fipribench.scoring.utils
from rdkit import Chem, DataStructs
import pickle, gzip, sys, os, os.path, numpy
from collections import defaultdict
from optparse import OptionParser 
from sklearn.linear_model import LogisticRegression
import logging

from pathlib import Path
from collections import namedtuple

log = logging.getLogger("scoring")
# import fingerprint library
from .. import fingerprint_lib



from ... import configuration_file_I as conf

from .. import utils as scor

# import ML functions
from .. import ml_functions_13 as ml_func


# paths
path = Path(os.path.realpath(__file__)).parent
# this is root path of the package
parentpath = path.parent.parent
INPATH_CMP = parentpath.joinpath('compounds')
INPATH_LIST = parentpath.joinpath('query_lists/data_sets_I')

# dictionary for readMLFile()
read_dict = {}
read_dict['penalty'] = lambda x: x
read_dict['dual'] = lambda x: bool(x)
read_dict['C'] = lambda x: float(x)
read_dict['fit_intercept'] = lambda x: bool(x)
read_dict['intercept_scaling'] = lambda x: float(x)
read_dict['class_weight'] = lambda x: x
read_dict['tol'] = lambda x: float(x)

# prepare command-line option parser
usage = "usage: %prog [options] arg"
parser = OptionParser(usage)
parser.add_option("-n", "--num", dest="num", type="int", metavar="INT", help="number of query mols")
parser.add_option("-f", "--fingerprint", dest="fp", help="fingerprint to train logistic regression with")
parser.add_option("-o", "--outpath", dest="outpath", metavar="PATH", help="relative output PATH (default: pwd)")
parser.add_option("-s", "--similarity", dest="simil", type="string", metavar="NAME", help="NAME of similarity metric to use (default: Dice, other options are: Tanimoto, Cosine, Russel, Kulczynski, McConnaughey, Manhattan, RogotGoldberg")
parser.add_option("-m", "--ml", dest="ml", metavar="FILE", help="file containing the logistic regression info (default parameters: penalty=l2, dual=0 (false), C=1.0, fit_intercept=1 (true), intercept_scaling=1.0, class_weight=None, tol=0.0001)")
parser.add_option("-a", "--append", dest="do_append", action="store_true", help="append to the output file (default: False)")

############# MAIN PART ########################

from abc import ABC


class ScoringModel(ABC):
    """
    Abstract class that defines the interface for different scoring methods
    """

    def __init__(self):

        pass


    def train(self, fingerprints, classifications):
        """

        :param fingerprints: nparray of fingerprints
        :param classifications: nparray of binary classifications (active/inactive),
            must map to the 'fingerprints' elements
        :return:
        """
        raise NotImplementedError

    def predict(self, fingerprints):
        """
        After an eventual training, predict the
        :param fingerprint:
        :return: The scores of the input fingerprints
        """

        raise NotImplementedError


class LogisticScoring(ScoringModel):

    # TODO find more general "options" scheme
    def __init__(self, ml_dict):
        self._ml = LogisticRegression(
            penalty=ml_dict['penalty'],
            dual=ml_dict['dual'],
            C=ml_dict['C'],
            fit_intercept=ml_dict['fit_intercept'],
            intercept_scaling=ml_dict['intercept_scaling'],
            class_weight=ml_dict['class_weight'],
            tol=ml_dict['tol']
        )

    def train(self, fingerprints, classifications):
        self._ml.fit(fingerprints, classifications)

    def predict(self, fingerprints):
        return self._ml.predict_proba(fingerprints)


class Filesystem(ABC):
    """
    Abstract class that reads and writes information to the filesystem,
    or maybe later to a database
    """
    def __init__(self):
        pass


MoleculeData = namedtuple('MoleculeData', 'internal_id external_id smiles')


class DataSet():
    """
    Represents the dataset the scoring is performed over
    TODO make mapping dynamic, lookup path for dataset identifier
    """

    def __init__(self):
        pass

    def get_data(self, path) -> MoleculeData:
        lines = gzip.open(path, 'r')
        for line in lines:
            # TODO error handling/ filtering of other lines
            if line[0] != '#':
                # structure of line: [external ID, internal ID, SMILES]]
                line = line.rstrip().split()
            yield MoleculeData(*line[:3])

    def get_actives(self, dataset, target):
        path = INPATH_CMP.joinpath(f'{dataset}/cmp_list_{dataset}_{str(target)}_actives.dat.gz')
        # TODO yield or return?
        yield self.get_data(path)

    def get_decoys(self, dataset_name, target_name):
        if dataset_name is 'ChEMBL':
            # FIXME edgecase, here this is not intended to be called target specific
            path = INPATH_CMP.joinpath(f'{dataset_name}/cmp_list_{dataset_name}_zinc_decoys.dat.gz')
        else:
            path = INPATH_CMP.joinpath(f'{dataset_name}/cmp_list_{dataset_name}_{str(target_name)}_decoys.dat.gz')
        yield self.get_data(path)

    def get_training_list(self, dataset_name, target_name, num_query_mols):
        training_input = open(
            INPATH_LIST.joinpath(
                f'{dataset_name}/training_{dataset_name}_{str(target_name)}_{str(num_query_mols)}.pkl'
            ),
            'r'
        )
        return pickle.load(training_input)


def parse_args():

    # read in command line options
    (options, args) = parser.parse_args()
    # required arguments
    if options.num and options.fp:
        num_query_mols = options.num
        fp_build = options.fp
    else:
        raise RuntimeError('One or more of the required options was not given!')

    # optional arguments
    do_append = False
    if options.do_append: do_append = options.do_append
    simil_metric = 'Dice'
    if options.simil: simil_metric = options.simil
    outpath = path
    outpath_set = False
    if options.outpath:
        outpath_set = True
        outpath = path+options.outpath

    # check for sensible input
    if outpath_set: scor.checkPath(outpath, 'output')
    fipribench.scoring.similarity.checkSimil(simil_metric)
    scor.checkQueryMols(num_query_mols, conf.list_num_query_mols)

    # default machine-learning method variables
    ml_dict = dict(penalty='l2', dual=False, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None, tol=0.0001)
    if options.ml:
        ml_dict = fipribench.scoring.utils.readMLFile(ml_dict, read_dict, path + options.ml)

    # initialize machine-learning method
    ml = LogisticRegression(penalty=ml_dict['penalty'], dual=ml_dict['dual'], C=ml_dict['C'], fit_intercept=ml_dict['fit_intercept'], intercept_scaling=ml_dict['intercept_scaling'], class_weight=ml_dict['class_weight'], tol=ml_dict['tol'])

    return None


from sklearn.tree import tree
def fp_vector_to_nparray(vector):
    arr = numpy.zeros((3,), tree.DTYPE)
    DataStructs.ConvertToNumpyArray(vector, arr)
    return arr

def calculate_scored_lists(num_query_mols, fingerprint_method, outpath, simil_metric, scoring_model, append,
                           filesystem, datasets: DataSet):

    # loop over data-set sources
    for dataset_name in conf.set_data.keys():
        log.info(dataset_name)
        # loop over targets
        firstchembl = True
        for target_name in conf.set_data[dataset_name]['ids']:
            log.info(target_name)

            # read in actives and calculate fps
            actives_ids = []
            np_fps_act = []
            for mol_data in datasets.get_actives(dataset_name, target_name):
                    fp_vector = fipribench.fingerprint.__init__.getFP(fingerprint_method, mol_data.smiles)
                    actives_ids.append(mol_data.internal_id)
                    np_fps_act.append(fp_vector_to_nparray(fp_vector))
            num_actives = len(actives_ids)
            num_test_actives = num_actives - num_query_mols

            # read in decoys and calculate fps
            decoys_ids = []
            np_fps_dcy = []
            if not (firstchembl is False and dataset_name == "ChEMBL"):
                for mol_data in datasets.get_decoys(dataset_name, target_name):
                    fp_vector = fipribench.fingerprint.__init__.getFP(fingerprint_method, mol_data.smiles)
                    decoys_ids.append(mol_data.internal_id)
                    np_fps_dcy.append(fp_vector_to_nparray(fp_vector))
                if dataset_name == 'ChEMBL':
                    firstchembl = False

            num_decoys = len(decoys_ids)
            log.info("Molecules read in and fingerprints calculated")

            # open training lists
            # this is a list of some IDs, they seem to be "randomly" appended to the list
            training_list = datasets.get_training_list(dataset_name, target_name, num_query_mols)
            # to store the scored lists
            scores = defaultdict(list)

            # loop over repetitions
            for q in range(conf.num_reps):
                log.info(q)

                # this exludes some IDs, which are present in the training list, those will be used later in the
                # train_fps to train the models
                test_list = [i for i in range(num_actives) if i not in training_list[:num_query_mols]]
                test_list += [i for i in range(num_decoys) if i not in training_list[num_query_mols:]]

                # list with active/inactive info
                # XXX they construct a list [1, 1, 1,..., 0,0,...] (class membership active/inactive)
                ys_fit = [1]*num_query_mols + [0]*(len(training_list)-num_query_mols)

                # training fps
                # XXX they construct a list [active_fp, active_fp, active_fp, ..., inactive_fp, inactive_fp,...]
                train_fps = [np_fps_act[i] for i in training_list[:num_query_mols]]
                train_fps += [np_fps_dcy[i] for i in training_list[num_query_mols:]]

                # XXX now they use the training set and the ys_fit to train a model of some kind
                # fit logistic regression
                scoring_model.train(train_fps, ys_fit)

                # test fps and molecule info
                test_fps = [np_fps_act[i] for i in test_list[:num_test_actives]]
                test_fps += [np_fps_dcy[i] for i in test_list[num_test_actives:]]

                test_mols = [[actives_ids[i], 1] for i in test_list[:num_test_actives]]
                test_mols += [[decoys_ids[i], 0] for i in test_list[num_test_actives:]]

                # rank based on probability
                predicted_scores = scoring_model.predict(test_fps)
                # returns: array - like, shape = [n_samples, n_classes]

                # store: [probability, internal ID, active/inactive]
                # TODO check if the probablity is stored in single_score
                single_score = [[s[1], m[0], m[1]] for s, m in zip(predicted_scores, test_mols)]
                single_score.sort(reverse=True)
                scores['lr_'+fingerprint_method].append(single_score)

            # write scores to file
            if append is True:
                # binary format
                outfile = gzip.open(outpath.joinpath(f'list_{dataset_name}_{str(target_name)}.pkl.gz'), 'ab+')
            else:
                # binary format
                outfile = gzip.open(outpath.joinpath(f'list_{dataset_name}_{str(target_name)}.pkl.gz'), 'wb+')
            for fp in ['lr_'+fingerprint_method]:
                pickle.dump([fp, scores[fp]], outfile, 2)
            outfile.close()
            log.info("Scoring done and scored lists written")
