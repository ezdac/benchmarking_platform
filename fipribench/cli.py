import itertools
import logging
import sys
from pathlib import PosixPath
import click
import json
import time

from fipribench.fingerprint import FingerprintInterface

from fipribench.config_schema import (
    RandomForest,
    Config,
    read_settings_dict,
    dict_convert_enums_to_values_recursive,
    list_convert_enums_to_values
)
from fipribench.fingerprint import fpdict as fingerprint_dict
from fipribench.scoring.similarity import simil_dict as similarity_dict
from fipribench.datasets import DataSet
from fipribench.scoring import calculate_scored_lists
from fipribench.scoring.model import LogisticScoring
from fipribench.utils import get_classes_for_module
from fipribench.fingerprint import implementations as fingerprint_implementations
from fipribench.fingerprint import class_implements_fingerprint_interface

import fipribench.configuration_file_I as conf
import enlighten


LOGGER = logging.getLogger(__name__)

allowed_fingerprints = list(fingerprint_dict.keys())
allowed_similarity_measures = list(similarity_dict.keys())


@click.command()
@click.option('--config', '-c', required=True,
              type=click.Path(readable=True, resolve_path=True, dir_okay=False, file_okay=True),
              help='Path to the config file.')
@click.option('--logfile', '-l',
              type=click.Path(readable=True, resolve_path=True, dir_okay=False, file_okay=True),
              help='Optional file to save logging output in.')
@click.option('--verbose', '-v', count=True,
              help='Set the verbosity level, -v = WARNING -vv = DEBUG')
# TODO set default path!
@click.option('--outpath', '-o',
              type=click.Path(writable=True, resolve_path=True, dir_okay=True, file_okay=False),
              help='The path the output data should be written to')
# @click.argument('runfile', required=True,
#                 type=click.Path(readable=True, resolve_path=True, dir_okay=False, file_okay=True))
def cli(config: click.Path, outpath: click.Path, logfile: click.Path, verbose: int):

    input_path = PosixPath(config).resolve()
    settings_dict = read_settings_dict(input_path)
    settings = Config(**settings_dict)

    progress_bar_manager = enlighten.get_manager()
    initializing_progress_bar = progress_bar_manager.counter(
        total=4, desc='Initializing', unit='actions')
    initializing_progress_bar.update()

    logging_level = logging.INFO
    if verbose is 1:
        logging_level = logging.WARNING
    elif verbose > 1:
        logging_level = logging.DEBUG

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging_level)

    LOGGER.debug(f"Initialized program with settings: \n {json.dumps(json.loads(settings.json()), indent=2)}")

    dataset = DataSet()
    ml_models = []

    if settings.ml_models.logistic_regression is not None:
        kwargs = dict_convert_enums_to_values_recursive(settings.ml_models.logistic_regression.dict())

        model = LogisticScoring(sklearn_kwargs=kwargs)
        ml_models.append(model)

    initializing_progress_bar.update()
    if settings.ml_models.random_forest is not None:
        kwargs = dict_convert_enums_to_values_recursive(settings.ml_models.logistic_regression.dict())

        model = RandomForest(sklear_kwargs=kwargs)
        ml_models.append(model)
    initializing_progress_bar.update()

    # TODO Naive bayes
    # TODO Nearest Neighbor Similarity

    custom_fingerprint_classes = set(get_classes_for_module(fingerprint_implementations))
    fp_class_name_dict = {class_name: getattr(fingerprint_implementations, class_name) for class_name in
                          custom_fingerprint_classes}

    fp_method_instances = []
    for fp_method_settings in settings.fingerprints:
        try:
            fp_klass = fp_class_name_dict[fp_method_settings.method]
        except KeyError:
            raise KeyError("Provided fingerprint method not found in \'implementations.py\'")
        try:
            # FIXME here we need some checking and conversion (e.g. possible classifications?)
            # also, most of those should probably global parameters
            # "possible_classifications": [0, 1],
            # "ecfp_radius": 2
            fp_instance = fp_klass(**fp_method_settings.kwargs)
        except TypeError as e:
            raise TypeError("Provided keyword arguments do not match signature in \'implementations.py\': \n"
                            + str(e)).with_traceback(sys.exc_info()[2])
        fp_method_instances.append(fp_instance)

    verified_fp_method_instances = []
    for fp_method in fp_method_instances:
        if FingerprintInterface.provided_by(fp_method) is False:
            LOGGER.warning(f"Custom fingerprinter class {fp_method.__class__, __name__} does not to the interface"
                        f"as specified in \"fingerprint.__init__.py\"")
        else:
            verified_fp_method_instances.append(fp_method)

    if len(verified_fp_method_instances) == 0:
        # TODO better exception class, better message
        raise SystemExit("There was no custom fingerprinter method invoked")

    initializing_progress_bar.update()
    dataset_names = list_convert_enums_to_values(settings.datasets)

    initializing_progress_bar.close()

    # terminal = progress_bar_manager.term
    # bar_format = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
    #              u'S:' + terminal.green(u'{count_0:{len_total}d}') + u' ' + \
    #              u'F:' + terminal.red(u'{count_2:{len_total}d}') + u' ' + \
    #              u'E:' + terminal.yellow(u'{count_1:{len_total}d}') + u' ' + \
    #              u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

    # offset = len(terminal.green('')) + len(terminal.red('')) + len(terminal.yellow(''))


    overall_progress_bar = progress_bar_manager.counter(
        total=len(ml_models) * len(fp_method_instances) * len(dataset_names),
        desc='Scoring',
        unit='combinations',
    )
    overall_progress_bar.refresh()
    # create all combinations of parameters (equivalent to 3-fold nested loops
    for model, fp_method, dataset_name in itertools.product(
        ml_models,
        fp_method_instances,
        dataset_names
    ):

        firstchembl = True
        for target_name in conf.set_data[dataset_name]['ids']:
            # FIXME ignores the "firstchembl", this results in recalculating ChEMBL decoy fingerprints,
            # although this has to be done only once
            data_frame = calculate_scored_lists(
                num_query_mols=settings.number_query_molecules.value,
                fingerprint_method=fp_method,
                outpath=outpath,
                simil_metric=settings.similarity_measure.value,
                scoring_model=model,
                target_name=target_name,
                dataset_name=dataset_name,
                datasets=dataset,
                number_of_repetitions=conf.num_reps,
            )

            #
            # # write scores to file
            # append = True
            # # TODO add in config
            # if append is True:
            #     # binary format
            #     outfile = gzip.open(outpath.joinpath(f'list_{dataset_name}_{str(target_name)}.pkl.gz'), 'ab+')
            # else:
            #     # binary format
            #     outfile = gzip.open(outpath.joinpath(f'list_{dataset_name}_{str(target_name)}.pkl.gz'), 'wb+')
            # #     FIXME this iterates over one element?
            # for fp in ['lr_'+fingerprint_method]:
            #     pickle.dump([fp, scores[fp]], outfile, 2)
            #     # TODO log filename etc
            #     log.info("Written file")
            # outfile.close()
            # log.info("Scoring done and scored lists written")
            time.sleep(0.2)
            overall_progress_bar.update()

    overall_progress_bar.close()
    progress_bar_manager.stop()
