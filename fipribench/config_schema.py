import json
from enum import Enum
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, BaseSettings

from fipribench.datasets import DataSet


class FingerprintMethod(BaseModel):
    method: str = None  # Class name as defined in the implementations.py
    name: str = None  # Custom name, that will be used to describe the method (should be unique)
    kwargs: Dict[str, Any] = {}  # arguments to be passed to the initilizer


class DataSetName(Enum):
    chembl = 'ChEMBL'
    dud = 'DUD'
    muv = 'MUV'


class NofQueryMolecules(Enum):
    five = 5
    ten = 10
    twenty = 20

class SimilarityMeasure(Enum):
    DICE = 'Dice'
    TANIMOTO = 'Tanimoto'
    COSINE = 'Cosine'
    RUSSEL = 'Russel'
    KULCZYNSKI = 'Kulczynski'
    MCCONNAUGHEY = 'McConnaughey'
    MANHATTAN = 'Manhattan'
    ROGOTGOLDBERG = 'RogotGoldberg'


class LogisticRegressionPenalty(Enum):
    l1 = 'l1'
    l2 = 'l2'
    elasticnet = 'elasticnet'
    none = 'none'


class LogisticRegressionSolver(Enum):
    newton_cg = 'newton_cg'
    lbfgs = 'lbfgs'
    liblinear = 'liblinear'
    sag = 'sag'
    saga = 'saga'


class LogisticRegression(BaseModel):
    penalty: Optional[LogisticRegressionPenalty] = None
    dual: Optional[bool] = None
    fit_intercept: Optional[bool] = None
    intercept_scaling: Optional[float] = None
    class_weight:  Union[Dict[str, float], str] = None
    solver: LogisticRegressionSolver = None


class RandomForestCriterion(Enum):
    gini = 'gini'
    entropy = 'entropy'


class RandomForestMaxFeaturesString(Enum):
    auto = 'auto'
    sqrt = 'sqrt'
    log2 = 'log2'


class RandomForest(BaseModel):
    criterion: RandomForestCriterion = None
    max_features: Optional[Union[int, float, RandomForestMaxFeaturesString]] = None
    n_jobs: Optional[int] = None
    max_depth: Optional[int] = None
    min_samples_split: Optional[Union[int, float]] = None
    max_samples_split: Optional[Union[int, float]] = None
    min_samples_leaf: Optional[Union[int, float]] = None
    max_samples_leaf: Optional[Union[int, float]] = None
    num_estimators: Optional[int] = None


class MLModels(BaseModel):
    logistic_regression: LogisticRegression = None
    random_forest: RandomForest = None
    # naive_bayes = None  # TODO


class Config(BaseSettings):
    fingerprints: List[FingerprintMethod] = ...
    # TODO should this be global parameter or can you just define this by providing another FingerprintMethod whith a \
    #  (different) bitsize parameter to the list above?
    bitsize: int = 1024
    datasets: List[DataSetName] = ...
    number_query_molecules: NofQueryMolecules = ...
    number_scoring_repetitions: int = 50
    # FIXME if this is missing, we get an error that's not informative for the user
    ml_models: MLModels = None
    similarity_measure : SimilarityMeasure = SimilarityMeasure.TANIMOTO


def read_settings_dict(path):
    settings_dict = {}
    with open(path, 'r') as f:
        json_dict = json.load(f)
        settings_dict.update(json_dict)
    return settings_dict


def list_convert_enums_to_values(list_):
    return [elem.value for elem in list_]


def dict_convert_enums_to_values_recursive(dict_):
    new_dict = {}
    for key, val in dict_.items():
        if isinstance(val, Enum):
            new_dict[key] = val.value
        elif isinstance(val, dict):
            new_dict[key] = dict_convert_enums_to_values_recursive(val)
        elif isinstance(val, list):
            new_dict[key] = list_convert_enums_to_values(val)
        else:
            new_dict[key] = val
    return new_dict