from abc import ABC, abstractmethod
import gzip
import pickle
import logging
import numpy
from pathlib import PosixPath

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


class FileSystem(ABC):
    """
    Abstract class that reads and writes information to the filesystem,
    or maybe later to a database
    """

    def __init__(self, root_path: 'PosixPath', append=False):
        self.root_path = root_path
        if append is True:
            raise NotImplementedError("Append behaviour for file persistance is not implemented yet")
        self.append = append

    @abstractmethod
    def _save_file(self, np_dict: 'Dict[str, numpy.ndarray]', relative_path: 'PosixPath'):
        pass

    @abstractmethod
    def _load_file(self, relative_path: 'PosixPath'):
        pass

    def save_scores(self, scores_array: 'numpy.ndarray', internal_ids: 'numpy.ndarray', dataset_name: str,
                    target_name: str, fingerprint_method_name: str):
        data = dict()
        data['scores'] = scores_array
        data['internal_id'] = internal_ids
        # FIXME probably also include the fingerprint_method hash, so that we know once the fingerprint training set
        # is changed?
        relative_path = PosixPath(f'scores/{dataset_name}_{target_name}_{fingerprint_method_name}')
        self._save_file(data, relative_path)

    def load_scores(self, dataset_name: str, target_name: str, fingerprint_method_name: str):
        # FIXME probably also include the fingerprint_method hash, so that we know once the fingerprint training set
        # is changed?
        relative_path = PosixPath(f'scores/{dataset_name}_{target_name}_{fingerprint_method_name}')
        loaded = self._load_file(relative_path)
        return loaded['scores'], loaded['internal_id']


class PickleFilesystem(FileSystem):
    """
    TODO get rid of this class (probably)
    """

    def _save_file(self, np_dict: 'Dict[str, numpy.ndarray]', relative_path: 'PosixPath'):
        raise NotImplementedError
        # TODO
        open_flag = 'wb+'
        if append is True:
            open_flag = 'ab+'
        with gzip.open(self.root_path.joinpath(relative_path).joinpath('pkl.gz'), open_flag) as outfile:
            # not fixing the protocol is fine, because it is encoded in the file
            pickle.dump(pickleable, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            LOGGER.info(f'Written file {outfile.filename}')

    def _load_file(self, relative_path: 'PosixPath'):
        raise NotImplementedError
        with gzip.open(self.root_path.joinpath(relative_path).joinpath('pkl.gz'), 'rb+') as outfile:
            object_ = pickle.load(outfile)
            LOGGER.info(f'Loaded file {outfile.filename}')
        return object_


class NFZFilesystem(FileSystem):
    def _save_file(self, np_dict: 'Dict[str, numpy.ndarray]', relative_path: 'PosixPath'):
        path = self.root_path.joinpath(relative_path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        numpy.savez_compressed(str(path), **np_dict)
        LOGGER.info(f'Written file {path}, with data={list(np_dict.keys())}.')

    def _load_file(self, relative_path: 'PosixPath'):
        path = str(self.root_path.joinpath(relative_path).joinpath('.npz'))
        return numpy.load(path)


class DummyFileSystem(FileSystem):
    """
    Class that doesn't pass the files to the filesystem and does nothing to persist them.
    This class is used, when no data persistence is present in the user settings
    """

    def _save_file(self, np_dict: 'Dict[str, numpy.ndarray]', relative_path: 'PosixPath'):
        return None

    def _load_file(self, relative_path: 'PosixPath'):
        return None
