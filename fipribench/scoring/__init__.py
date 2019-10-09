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

from fipribench.datasets import DataSet
from fipribench.utils import fp_vector_to_nparray
from fipribench.fingerprint import CalculateFP
import pandas as pd
import numpy as np
import enlighten

import logging

LOGGER = logging.getLogger(__name__)


def calculate_scored_lists(num_query_mols, fingerprint_method, outpath, simil_metric, scoring_model,
                           target_name, dataset_name, datasets: DataSet, number_of_repetitions):

            # TODO use the get_dataframe instead of all this np stuff in this function !!!
            LOGGER.info(f"Start scoring for: target={target_name}, dataset={dataset_name}, similarity_metric={simil_metric},"
                     f" fingerprint={fingerprint_method.name}")
            data = datasets.get_dataframe(dataset_name, target_name, num_query_mols)

            # the last to columns' values are not shown in the debug output
            LOGGER.debug(data[:5])

            # train the fingerprint method with all molecules marked for fingerprint training
            fp_training_data = data[data['is_training'] == True]
            LOGGER.info(f"Start training phase with {fp_training_data.shape[1] / data.shape[1] * 100:.2f}% of scoring samples")
            fingerprint_method.train(fp_training_data['smiles'], fp_training_data['is_active'].astype(int))
            # TODO print info on training phasse, #cluster blabla
            LOGGER.info(f"Finished training phase.")

            # now calculate all fingerprints
            data['fingerprint'] = data['smiles'].apply(fingerprint_method.calculate_fingerprint)
            LOGGER.info("Calculated all fingerprints.")

            test_data = data[data['is_training'] == False]
            training_data = data[data['is_training'] == False]

            # to store the scored lists
            # FIXME check correct shape
            scores = np.empty((data.shape[0], 1))

            # loop over repetitions
            progress_bar_manager = enlighten.get_manager()
            scoring_progress_bar = progress_bar_manager.counter(
                total=number_of_repetitions,
                desc='Scoring iterations',
                unit='iterations',
                leave=False
            )
            scoring_progress_bar.refresh()
            for q in range(number_of_repetitions):
                LOGGER.info(f"Training scoring model {scoring_model.name}, iteration={q + 1}")
                # XXX now they use the training set and the ys_fit to train a model of some kind
                # fit logistic regression
                fingerprints = training_data['fingerprint'].to_numpy()
                LOGGER.debug(fingerprints)
                scoring_model.train(training_data['fingerprint'].values, training_data['is_active'].values)

                # rank test molecules based on probability
                LOGGER.info(f"Predicting scores for test data, iteration={q + 1}")
                predicted_scores = scoring_model.predict(test_data['fingerprint'])
                # returns: array - like, shape = [n_samples, n_classes]

                # store: [probability, internal ID, active/inactive]
                # single_score = [[s[1], m[0], m[1]] for s, m in zip(predicted_scores, test_mols)]
                scores = np.concatenate((predicted_scores[:, 1], scores), axis=1)
                scoring_progress_bar.update()

            scoring_progress_bar.close()
            # FIXME does this do "hstacking" on the dataframe?
            LOGGER.info(f"Finish scoring for: target={target_name}, dataset={dataset_name}, similarity_metric={simil_metric},"
                     f"fingerprint={fingerprint_method.name}")
            data['scores'] = scores


            # FIXME they sort one single score per run
            # we can't do that if we append scores to our df, because we have to associate the scores to the testmols.
            # FIXME is this equivalent to: single_score.sort(reverse=True)
            # active_proba = np.flip(np.sort(predicted_scores[:, 1]))
            # I think we should sort outside, when we actually need this!

            return scores


