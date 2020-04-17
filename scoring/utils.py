import os


def get_cwd_from_file_path():
    return os.path.dirname(os.path.realpath(__file__))


