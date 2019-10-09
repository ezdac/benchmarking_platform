import inspect

import numpy
from rdkit import DataStructs
from sklearn.tree import tree


def get_classes_for_module(module):
    return [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == module.__name__]


def fp_vector_to_nparray(vector):
    # FIXME This was how it was before, i don't understand at all why you would do this?
    # arr = numpy.zeros((3,), tree.DTYPE)
    arr = numpy.zeros((1,), bool)
    DataStructs.ConvertToNumpyArray(vector, arr)
    return arr
