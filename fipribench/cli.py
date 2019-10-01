import itertools
import logging
import sys
import json

import click
import yaml
from jsonschema import validate as validate_schema

from fipribench.fingerprint import fpdict as fingerprint_dict
from fipribench.scoring.similarity import simil_dict as similarity_dict
from fipribench.datasets import DataSet
from fipribench.scoring import calculate_scored_lists
from fipribench.scoring.model import LogisticScoring

log = logging.getLogger(__name__)


allowed_fingerprints = list(fingerprint_dict.keys())
allowed_similarity_measures = list(similarity_dict.keys())

# TODO
general_config_schema = {}

run_config_schema = {
    "title": "Run Configuration",
    "type": "object",
    "fingerprint": {
        "type": "object",
        "methods": {
            "type": "array",
            'items': {
                "type": "string",
                "enum": allowed_fingerprints
            },
        },
        "bitsize": {"type": "number"},
    },
    "datasets": {
        "type": "object",
        "nof_query_molecules": {
            "type": "number",
            "enum": ["5", "10", "20"]
        }
    },
    "scoring": {
        "type": "object",
        "similarity_measure": {
            "type": "string",
            "enum": allowed_similarity_measures,
        }
    },
    "training": {
            "type": "object",
            "logistic_regression": {
                "type": "object",
                "penalty": {
                    "type": "string",
                    "enum": ["l1", "l2", "elasticnet", "none"]
                },
                "dual": {
                    "type": "number",
                },
                "fit_intercept": {
                    "type": "boolean",
                },
                "intercept_scaling": {
                    "type": "number",
                },
                "class_weight": {
                    # TODO allow this, this can be a dictionary
                    "type": "string",
                    "enum": ["l1", "l2", "elasticnet", "none"]
                    # TODO optional
                },
                "tol": {
                    "type": "number",
                },
                "solver": {
                    "type": "string",
                    "enum": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                },
            },
            "random_forest": {
                "type": "object",
                "criterion": {
                    "type": "string",
                    # TODO what else is possible?
                    "enum": ["gini"]
                },
                "max_features": {
                    "type": "string",
                    # TODO can be int, float, string or None,
                },
                "n_jobs": {
                    "type": "number",
                },
                "max_depth": {
                    "type": "number",
                },
                "min_samples_split": {
                    "type": "number",
                },
                "min_samples_leaf": {
                    "type": "number",
                },
                "num_estimators": {
                    "type": "number",
                },
            }
        },
}


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
@click.argument('runfile', required=True,
                type=click.Path(readable=True, resolve_path=True, dir_okay=False, file_okay=True))
def cli(config: click.Path, runfile: click.Path, outpath: click.Path, logfile: click.Path, verbose: int):

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

    # # TODO parse general config
    # with open(config, 'r') as f:
    #     # use safe_load instead load
    #     config_args = yaml.safe_load(f)
    #     validate_schema(instance=config_args, schema=general_config_schema)

    ml_models = []
    training_config = {}
    scoring_config = {}
    dataset_config = {}
    fingerprint_config = {}

    with open(runfile, 'r') as f:
        run_args = yaml.safe_load(f)
        # validate if the config is given properly, then we can parse it without any further
        # type checking or being afraid of errors
        validate_schema(instance=run_args, schema=run_config_schema)
        log.debug(f"Runfile validation accepted. Parameters: {json.dumps(run_args, indent=4)}")

        fingerprint_config.update(run_args['fingerprint'])
        dataset_config.update(run_args['datasets'])
        scoring_config.update(run_args['scoring'])
        training_config.update(run_args['training'])

    dataset = DataSet()

    if 'logistic_regression' in training_config:
        # TODO check options beforehand and print error
        model = LogisticScoring(training_config['logistic_regression'])
        ml_models.append(model)

    # if 'random_forest' in training_config:
    # ml_dict = dict(criterion='gini', max_features='auto', n_jobs=1, max_depth=10, min_samples_split=2,
    #                min_samples_leaf=1, num_estimators=100

    # calculate the scoring for every combination of scoring model and fingerprint method
    for model, fp_method in itertools.product(ml_models, fingerprint_config['methods']):
        calculate_scored_lists(
            num_query_mols=dataset_config['nof_query_molecules'],
            fingerprint_method=fp_method,
            outpath=outpath,
            simil_metric=scoring_config['similarity_measure'],
            scoring_model=model,
            # TODO add in config
            append=False,
            filesystem=None,
            datasets=dataset,
        )

