#
# $Id$
#
# file containing the functions for the scoring step
#
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

import operator


# helper functions for the fusion
import numpy
from rdkit import DataStructs
from sklearn.tree import tree


def getRanks(probas):
    '''Add the ranks for a ranked list'''
    num_mol = len(probas)
    ranks = [[num_mol-i] + j for i,j in enumerate(probas)]
    # sort based on internal ID
    ranks.sort(key=operator.itemgetter(-2))
    return ranks


def getNumpy(inlist):
    # input is array[ [internal_id, RDKITcDatastruct], [internal_id, RDKITcDatastruct], ... ]
    outlist = []
    for i in inlist:
        arr = numpy.zeros((3,), tree.DTYPE)
        DataStructs.ConvertToNumpyArray(i[1], arr)
        outlist.append(arr)
    return outlist


def readMLFile(ml_dict, read_dict, filepath):
    '''Reads file with the parameters of the machine-learning method
    and stores it in a dictionary'''
    try:
        myfile = open(filepath, 'r')
    except:
        raise IOError('file does not exist:', filepath)
    else:
        for line in myfile:
            l = line.rstrip().split()
        if len(l) != 2:
            raise ValueError('Wrong number of arguments in ML file:', line)
        if l[0] in read_dict:
            ml_dict[l[0]] = read_dict[l[0]](l[1])
        else:
            raise KeyError('Wrong parameter in naive Bayes file:', line)
    return ml_dict


def fp_vector_to_nparray(vector):
    arr = numpy.zeros((3,), tree.DTYPE)
    DataStructs.ConvertToNumpyArray(vector, arr)
    return arr