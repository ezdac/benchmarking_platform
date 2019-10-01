import os
from pathlib import Path

# paths
module_path = Path(os.path.realpath(__file__)).parent
# this is root path of the package
# parentpath = path.parent.parent
INPATH_CMP = module_path.joinpath('compounds')
INPATH_LIST = module_path.joinpath('query_lists/data_sets_I')
# TODO add data set II
