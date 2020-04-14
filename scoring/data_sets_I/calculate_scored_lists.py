#
# calculates fingerprints and scores lists
# based on the predicted probability
#
# INPUT
# required:
# -n [] : number of query mols
# -f [] : fingerprint to build the random forest with
# optional:
# -o [] : relative output path (default: pwd)
# -a : append to the output file (default: overwrite)
# -s [] : similarity metric (default: Dice, 
#         other options: Tanimoto, Cosine, Russel, Kulczynski, 
#         McConnaughey, Manhattan, RogotGoldberg)
# -r [] : file containing the random forest info
#          default parameters: criterion=gini, max_depth=10,
#          max_features=auto (=sqrt), num_estimators=100,
#          min_samples_split=2, min_samples_leaf=1, n_jobs=1
# --help : prints usage
#
# OUTPUT: for each target in each data set
#         a file with a list (1 element) of RF prediction
#         per RF prediction: [name, list of 50 scored lists]
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


from calculate_scored_lists_ML import run_scoring, get_cwd_from_file_path, cli_parser
from sklearn.base import BaseEstimator

from scoring.data_sets_I.models import NearestNeighbourModel

if __name__ == '__main__':
    (options, args) = cli_parser.parse_args()

    path = get_cwd_from_file_path() + '/'
    ml = NearestNeighbourModel()
    run_scoring(options, ml, path, 'nn')