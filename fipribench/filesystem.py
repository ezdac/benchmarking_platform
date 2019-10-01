from abc import ABC


class Filesystem(ABC):
    """
    Abstract class that reads and writes information to the filesystem,
    or maybe later to a database
    """
    def __init__(self):
        pass