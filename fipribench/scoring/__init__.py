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

import numpy as np
import enlighten
import time

import logging

LOGGER = logging.getLogger(__name__)


def calculate_scored_lists(data, fingerprint_method, simil_metric, scoring_model, number_of_repetitions):
    # FIXME why don't we use similarity metric??
    # train the fingerprint method with all molecules marked for fingerprint training
    nof_samples = data['smiles'].shape[0]
    fp_training_indices = data['is_fp_training'].nonzero()
    LOGGER.info(f"Start fingerprint training phase with "
                f"{fp_training_indices[0].shape[0] / data['is_fp_training'].shape[0] * 100:.2f}% "
                f"of scoring samples")

    fingerprint_method.train(data['smiles'][fp_training_indices], data['is_active'][fp_training_indices].astype(int))
    # TODO print info on training phase, #cluster blabla
    LOGGER.info(f"Finished fingerprint training phase.")

    # now calculate all fingerprints
    data['fingerprint'] = np.array(list(map(fingerprint_method.calculate_fingerprint, data['smiles'])))
    LOGGER.info("Calculated all fingerprints.")

    test_data_indices = (~data['is_training']).nonzero()
    training_data_indices = data['is_training'].nonzero()

    # to store the scored lists
    # FIXME check correct shape

    # loop over repetitions
    # progress_bar_manager = enlighten.get_manager()
    # scoring_progress_bar = progress_bar_manager.counter(
    #     total=number_of_repetitions,
    #     desc='Iterate',
    #     unit='iterations',
    #     leave=False
    # )
    # scoring_progress_bar.refresh()

    scoring_list = []
    # LOGGER.info(f"Training scoring model {scoring_model.name}, iteration={q + 1}")
    # LOGGER.info(f"Predicting scores for test data, iteration={q + 1}")
    LOGGER.info(f"Starting scoring, with {number_of_repetitions} iterations.")
    for q in range(number_of_repetitions):
        # XXX now they use the training set and the ys_fit to train a model of some kind
        # fit logistic regression

        scoring_model.train(data['fingerprint'][training_data_indices], data['is_active'][training_data_indices])

        # rank test molecules based on probability
        predicted_scores = scoring_model.predict(data['fingerprint'][test_data_indices])
        # returns: array - like, shape = [n_samples, n_classes]

        # store: [probability, internal ID, active/inactive]
        # single_score = [[s[1], m[0], m[1]] for s, m in zip(predicted_scores, test_mols)]
        scoring_list.append(predicted_scores[:, 1])
        # time.sleep(0.1)
        # scoring_progress_bar.update()
    LOGGER.info(f"Finished scoring.")

    # scoring_progress_bar.close()
    # time.sleep(0.1)

    # FIXME they sort one single score per run
    # we can't do that if we append scores to our df, because we have to associate the scores to the testmols.
    # FIXME is this equivalent to: single_score.sort(reverse=True)
    # active_proba = np.flip(np.sort(predicted_scores[:, 1]))
    # I think we should sort outside, when we actually need this!

    # returns scores list, and corresponding internal_id array
    return np.vstack(scoring_list), data['internal_id'][test_data_indices]
