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

# TODO refactor global configuration
import fipribench.configuration_file_I as conf
from fipribench.datasets import DataSet
from fipribench.scoring.utils import fp_vector_to_nparray
from fipribench.fingerprint import CalculateFP
import pickle
import gzip
from collections import defaultdict
import logging

log = logging.getLogger(__name__)


def calculate_scored_lists(num_query_mols, fingerprint_method, outpath, simil_metric, scoring_model, append,
                           filesystem, datasets: DataSet):

    # TODO this check should be conducted by comparing global config and run_config
    assert num_query_mols in (5, 10, 20)

    # loop over data-set sources
    # TODO refactor the datasetloop outside of function
    for dataset_name in conf.set_data.keys():
        log.info(dataset_name)
        # loop over targets
        firstchembl = True
        # TODO refactor the target loop outside of function??
        for target_name in conf.set_data[dataset_name]['ids']:
            log.info(target_name)

            # read in actives and calculate fps
            actives_ids = []
            np_fps_act = []
            for mol_data in datasets.get_actives(dataset_name, target_name):
                    fp_vector = CalculateFP(fingerprint_method, mol_data.smiles)
                    actives_ids.append(mol_data.internal_id)
                    np_fps_act.append(fp_vector_to_nparray(fp_vector))
            num_actives = len(actives_ids)
            num_test_actives = num_actives - num_query_mols

            # read in decoys and calculate fps
            decoys_ids = []
            np_fps_dcy = []
            if not (firstchembl is False and dataset_name == "ChEMBL"):
                for mol_data in datasets.get_decoys(dataset_name, target_name):
                    fp_vector = CalculateFP(fingerprint_method, mol_data.smiles)
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
            #     FIXME this iterates over one element?
            for fp in ['lr_'+fingerprint_method]:
                pickle.dump([fp, scores[fp]], outfile, 2)
                # TODO log filename etc
                log.info("Written file")
            outfile.close()
            log.info("Scoring done and scored lists written")
