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

from sklearn.linear_model import LogisticRegression


import scoring.ml_functions_13 as ml_func
from calculate_scored_lists_ML import run_scoring, get_cwd_from_file_path, cli_parser



read_dict = {
    'penalty': lambda x: x,
    'dual': lambda x: bool(x),
    'C': lambda x: float(x),
    'fit_intercept': lambda x: bool(x),
    'intercept_scaling': lambda x: float(x),
    'class_weight': lambda x: x,
    'tol': lambda x: float(x)
}


if __name__ == '__main__':

    (options, args) = cli_parser.parse_args()
    ml_dict = dict(penalty='l2', dual=False, C=1.0, fit_intercept=True, intercept_scaling=1.0, class_weight=None,
                   tol=0.0001)

    path = get_cwd_from_file_path() + '/'
    # default machine-learning method variables
    if options.ml:
        ml_dict = ml_func.readMLFile(ml_dict, read_dict, path+options.ml)
    ml = LogisticRegression(**ml_dict)
    run_scoring(options, ml, path, 'lr')