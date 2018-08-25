#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import random
import math
from collections import defaultdict, OrderedDict
from functools import cmp_to_key
import logging
import warnings
import json
import traceback
import datetime
import psutil
import copy
import errno

import numpy as np
from scipy.stats import pearsonr

from sklearn import preprocessing, svm, neighbors
import sklearn.metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, average_precision_score, log_loss
from sklearn.metrics.scorer import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.utils import check_X_y, column_or_1d, check_consistent_length
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.exceptions import DataConversionWarning

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, GetHashedTopologicalTorsionFingerprintAsBitVect

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mkdir_p(path):
    """Make directory with parents"""
    try:
        os.makedirs(path)
        return 0
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return 1
        else:
            raise


def csv_list(init):
    """auxiliary function to parse comma-delimited command line arguments"""
    field = init.split(',')
    return [x.strip() for x in field if len(x)]


def mol_to_fp_fun(fp_type, nBits=2048):
    """instantiate fingerprinting function from string"""
    if fp_type == 'ecfp4':
        return lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=nBits)
    if fp_type == 'ecfp6':
        return lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=nBits)
    if fp_type == 'ecfp12':
        return lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 6, nBits=nBits)
    if fp_type == 'maccs':
        if(nBits != 166):
            logging.warning('maccs FP is of a fix size (166). Ignoring nBits=' + str(nBits))
        return lambda x: Chem.MACCSkeys.GenMACCSKeys(x)
    if fp_type == 'daylight' or fp_type == "path5":
        return lambda x: FingerprintMols.FingerprintMol(x, minPath=1, maxPath=5, fpSize=nBits, bitsPerHash=2,
                                                        useHs=0, tgtDensity=0.0, minSize=nBits)
    if fp_type == 'ap':
        return lambda x: GetHashedAtomPairFingerprintAsBitVect(x, nBits=nBits)
    if fp_type == 'torsion':
        return lambda x: GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=nBits)
    else:
        logging.error('unknown fingerprint type: ' + fp_type)
        sys.exit(1)


warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
#warnings.simplefilter("ignore", DeprecationWarning)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def exceptions_str():
    return traceback.format_exception_only(
        sys.exc_info()[0], sys.exc_info()[1])[0].strip()


# grid of possible hyperparameter settings for optimization
# note: order is significant for 2nd stage local optimization

PARAM_GRID = {
    'rf': OrderedDict([
        ('class_weight', [None, 'balanced']),
        ('bootstrap', [False, True]),
        ('max_features', [None, 'sqrt', 'log2']),
        ('criterion', ['gini', 'entropy']),
        ('max_depth', [None, 10, 20, 50, 100]),
        ('min_samples_split', [2, 5, 10, 20, 50]),
        ('min_samples_leaf', [1, 5, 10, 20, 50]),
        ('n_estimators', [10, 50, 100, 200, 500]),
    ]),
    'gbm': OrderedDict([
        ('max_depth', [3, 4, 5, 6, 8]),
        ('min_samples_split', [2, 5, 10, 20, 50]),
        ('min_samples_leaf', [1, 5, 10, 20]),
        ('max_features', [None, 'sqrt', 'log2']),
        ('criterion', ['mse', 'friedman_mse', 'mae']),
        ('subsample', [0.3, 0.5, 0.8, 1.0]),
        ('learning_rate', [0.001, 0.01, 0.05, 0.1, 0.2]),
        ('n_estimators', [10, 50, 100, 200, 500]),
    ]),
    'xgb': OrderedDict([
        ('scale_pos_weight', [1, 5, 10, 20]),
        ('max_depth', [3, 4, 5, 6, 8, 12]),
        ('min_child_weight', [1, 2, 5, 10]),
        ('gamma', [0.0, 0.2, 0.5, 1.0]),
        ('max_delta_step', [0, 1, 5, 10]),
        ('subsample', [0.5, 0.8, 1.0]),
        ('colsample_bytree', [0.5, 0.8, 1.0]),
        ('reg_alpha', [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
        ('reg_lambda', [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
        ('learning_rate', [0.001, 0.01, 0.05, 0.1, 0.2]),
        ('n_estimators', [10, 50, 100, 200, 500]),
    ]),
    'lr': OrderedDict([
        ('class_weight', [None, 'balanced']),
        ('penalty', ['l1', 'l2']),
        ('C', [0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('tol', [1e-5, 1e-4, 1e-3]),
        #('max_iter', [200, 1000]),
    ]),
    'svm_lin': OrderedDict([
        ('class_weight', [None, 'balanced']),
        ('penalty', ['l1', 'l2']),
        ('C', [0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('tol', [1e-5, 1e-4, 1e-3]),
        # Unsupported set of arguments: The combination of penalty='l1' and loss='hinge' is not supported
        # 'loss': ['hinge', 'squared_hinge']),
    ]),
    'svm_poly': OrderedDict([
        # note: with sigmoid kernel and coef0=10 (and greater), decision function values seemed to
        # have offset between cross folds - weird!
        ('kernel', ['poly']),
        ('class_weight', [None, 'balanced']),
        ('degree', [2, 3, 4]),
        ('coef0', [0.0, 0.1, 1]),
        ('C', [0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('gamma', ['auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('shrinking', [False, True]),
        ('tol', [1e-4, 1e-3, 1e-2]),
    ]),
    'svm_rbf': OrderedDict([
        ('kernel', ['rbf']),
        ('class_weight', [None, 'balanced']),
        ('C', [0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('gamma', ['auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('shrinking', [False, True]),
        ('tol', [1e-4, 1e-3, 1e-2]),
    ]),
    'svm_sig': OrderedDict([
        # note: with sigmoid kernel and coef0=10 (and greater), decision function values seemed to
        # have offset between cross folds - weird!
        ('class_weight', [None, 'balanced']),
        ('kernel', ['sigmoid']),
        ('coef0', [0.0, 0.1, 0.5, 1]),
        ('C', [0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('gamma', ['auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]),
        ('shrinking', [False, True]),
        ('tol', [1e-4, 1e-3, 1e-2]),
    ]),
    'mlp': OrderedDict([
        ('alpha', [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0]),
        ('learning_rate_init', [0.001, 0.01, 0.1]),
        ('learning_rate', ['constant', 'invscaling', 'adaptive']),
        ('activation', ['logistic', 'tanh', 'relu']),
        ('depth', [1, 2, 3]),
        ('width', ['sqrt', 'log2', 'full']),
        ('tol', [1e-5, 1e-4, 1e-3]),
    ]),
}


def get_param_combinations(d):
    """cardinality of cross-product of all options"""
    sz = 1
    for v in d.values():
        sz *= len(v)
    return sz


def cmp_mixed(x, y):
    """compare function to sort mixed numeric/string values"""
    try:
        xf = float(x)
        yf = float(y)
        return xf - yf
    except:
        xs = str(x)
        ys = str(y)
        if xs < ys:
            return -1
        if xs > ys:
            return 1
        return 0


# note: did not get matplotlib cycler to work as desired
class PlotCycler(object):

    def __init__(self, what='color'):
        if what == 'color':
            self.cycle = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
        else:
            self.cycle = ['-', '--', '-.', ':', ':']

        self.pos = 0

    def next(self):
        pos_last = self.pos
        self.pos = (self.pos + 1) % len(self.cycle)
        return self.cycle[pos_last]


def core_basename(x):
    """strip off directories and all extensions from filename"""
    x = os.path.split(x)[1]
    ext = [x, 'dummy']
    while len(ext[1]) > 1:
        ext = os.path.splitext(ext[0])
    return ext[0]


def df_to_csv_append(df, csv_file_path, sep="\t", header=True, overwrite=False, **kwargs):
    """append pandas DataFrame to file"""
    if overwrite or not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False, sep=sep, header=True, **kwargs)
        return
    df_file = pd.read_csv(csv_file_path, nrows=1, sep=sep)

    if len(df.columns) != len(df_file.columns):
        raise ValueError("Columns do not match!! Dataframe has " + str(len(df.columns)) +
                         " columns. CSV file " + csv_file_path + " has " + str(len(df_file.columns)) + " columns.")
    elif not (df.columns == df_file.columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csv_file_path, mode='a', index=False, sep=sep, header=False, **kwargs)


class Fingerprinter(BaseEstimator, TransformerMixin):

    """transform a smiles string into one of several possible fingerprint types"""

    FP_TYPES = ['ecfp4', 'ecfp6', 'ecfp12', 'maccs', 'daylight', 'ap', 'torsion', 'prop']

    def __init__(self, fp_type='ecfp4', fp_size=2048):
        if fp_type not in Fingerprinter.FP_TYPES:
            raise ValueError('no such fingerprint type: %s' % fp_type)
        self.fp_type = fp_type
        self.fp_size = fp_size
        # avoid warnings from aw_common
        if fp_type == 'maccs':
            self.fp_size = 166
        self.features = None

    def get_feature_names(self):
        if self.features is None:
            raise ValueError('Fingerprinter: need to call fit() before getting feature names')
        return self.features

    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        logging.debug('generating %d %s fingerprints' % (len(X), self.fp_type))
        # if logging.getLogger().getEffectiveLevel() <= 10:
        #    for line in traceback.format_stack():
        #        logging.debug(line.strip())
        if self.fp_type == 'prop':
            properties = rdMolDescriptors.Properties()
            get_fp = properties.ComputeProperties
        else:
            get_fp = mol_to_fp_fun(self.fp_type, self.fp_size)
        fp = []
        smiles_used = []
        for i in range(len(X)):
            try:
                mol = Chem.MolFromSmiles(X[i], sanitize=True)
                fp.append(list(get_fp(mol)))
                smiles_used.append(X[i])
            except:
                logging.error('For smile: "%s":' % X[i])
                logging.error(exceptions_str())
                raise
                # pass

        if self.fp_type == 'prop':
            self.features = list(properties.GetPropertyNames())
        else:
            self.features = [self.fp_type + '_' + str(i) for i in range(len(fp[0]))]
        df = pd.DataFrame(fp, columns=self.features)
        logging.debug('done fingerprinting %s' % self.fp_type)
        return df


def clone_pipeline(pipe_in):
    """clone an sklearn Pipeline object"""
    return Pipeline(memory=pipe_in.memory,
                    steps=[(name, clone(est)) for name, est in pipe_in.steps])


def create_fp(fp_type, fp_size, mem):
    """create a single fingerprint transformer.
       - include scaling for molecular properties
    """
    if fp_type != 'prop':
        return Fingerprinter(fp_type, fp_size)
    else:
        pipe = Pipeline(memory=mem,
                        steps=[(fp_type, Fingerprinter(fp_type, fp_size))])

        # molecular descriptors can have very different ranges - normalize!
        pipe.steps.append(('scale', preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)))
        return pipe


def create_fp_pipe(fp_type, fp_size, mem):
    """instantiate a feature pipeline from string"""
    fps = fp_type.split('~')
    if len(fps) == 1:
        pipe = Pipeline(memory=mem,
                        steps=[(fp_type, Fingerprinter(fp_type, fp_size))])
        if fp_type == 'prop':
            # molecular descriptors can have very different ranges - normalize!
            pipe.steps.append(('features', preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)))
    else:
        combined_features = FeatureUnion([(fp, create_fp(fp, fp_size, mem)) for fp in fps], n_jobs=1)
        pipe = Pipeline(memory=mem, steps=[('features', combined_features)])
    return pipe, ('prop' not in fps)


def get_feature_names(pipe, pos=-1):
    """
    workaround: sklearn Pipeline, StandardScaler classes have not get_feature_names() method :-( ANNOYING
    """
    if not isinstance(pipe, (FeatureUnion, Pipeline)) and hasattr(pipe, 'get_feature_names'):
        return pipe.get_feature_names()
    el = pipe.steps[pos][1]
    if isinstance(el, preprocessing.StandardScaler):
        return get_feature_names(pipe, pos - 1)
    if isinstance(el, FeatureUnion):
        features = []
        for (name, est, weight) in el._iter():
            features.extend(get_feature_names(est))
        return features
    return get_feature_names(el)


def prepare_pipeline_for_scoring(pipe):
    """
    Recursively set all memory in pipe to None; if possible, parallelize classifiers and FeatureUnion
    """

    if hasattr(pipe, 'n_jobs'):
        pipe.n_jobs = -1
    if hasattr(pipe, 'nthread'):
        pipe.nthread = -1
    if isinstance(pipe, ThresholdTuner):
        prepare_pipeline_for_scoring(pipe.classifier)
    if isinstance(pipe, Pipeline):
        pipe.memory = None
        for s in pipe.steps:
            prepare_pipeline_for_scoring(s[1])

    if isinstance(pipe, FeatureUnion):
        for (name, est, weight) in pipe._iter():
            prepare_pipeline_for_scoring(est)


class LabelStratificationEncoder(BaseEstimator, TransformerMixin):
    """
    encode a target label and an additional stratification column into a fake 'multiclass' target.
    Auxiliary class to help with stratification when using sklearn CV classes.
    """

    def __init__(self):
        self.le = None

    def fit(self, strat=None):
        if strat is not None and len(strat) > 1:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(strat)
            logging.debug('LabelStratificationEncoder classes: %s, %s' % (self.le.classes_, np.bincount(strat)))
        return self

    def transform(self, y, strat=None):
        if self.le is None or strat is None:
            return y
        w_trans = self.le.transform(strat)
        # use lowest bit for target; assumption: y in [0,1]
        res = y + 2 * w_trans
        if logging.getLogger().getEffectiveLevel() <= 10:
            dist = np.bincount(res).astype(float)
            dist /= np.sum(dist)
            logging.debug('LabelStratificationEncoder distribution %s' % dist)

        return res

    def fit_transform(self, y, strat):
        self.fit(strat)
        return self.transform(y, strat)

    def inverse_transform(self, y):
        y_trans = np.mod(y, 2)
        strat = None
        if self.le is not None:
            strat = self.le.inverse_transform(y // 2)
        return y_trans, strat


class ThresholdTuner(BaseEstimator, ClassifierMixin):
    """wrapper for a classifier to additionally estimate a prediction threshold on held out validation set."""

    def __init__(self, classifier, thresh=None, valid_frac=0.0, max_false_positive_rate=.2, random_state=None, stratifier=None):
        """
        max_false_positive_rate - adjust threshold so as to have (at most) that many false positives.
        """
        self.classifier = classifier
        self.thresh = thresh
        self.valid_frac = valid_frac
        self.max_false_positive_rate = max_false_positive_rate
        self.random_state = random_state
        self.stratifier = stratifier
        if stratifier is not None and not isinstance(stratifier, LabelStratificationEncoder):
            raise ValueError('stratifier must be an instance of class LabelStratificationEncoder')

    def fit(self, X, y, sample_weight=None, test_filter=None):
        """
        Fit the embedded classifier, then estimate a threshold to attain max_false_positive_rate
        test_filter: subset index to compute threshold on
        """

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)

        w_train = None
        w_valid = None
        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)
            check_consistent_length(sample_weight, y)

        if test_filter is not None:
            test_filter = column_or_1d(test_filter)
            check_consistent_length(test_filter, y)

        # split train set further for validation holdout
        if self.valid_frac > 0.0:

            cv = StratifiedShuffleSplit(test_size=self.valid_frac, random_state=self.random_state)

            for train_idx, valid_idx in cv.split(X, y):
                break

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_valid = X[valid_idx][test_filter[valid_idx]]
            y_valid = y[valid_idx][test_filter[valid_idx]]

            if sample_weight is not None:
                w_train = sample_weight[train_idx]
                w_valid = sample_weight[valid_idx][test_filter[valid_idx]]

        else:
            # validate on train

            X_train = X
            y_train = y
            w_train = sample_weight

            X_valid = X
            y_valid = y
            w_valid = sample_weight

        if logging.getLogger().getEffectiveLevel() <= 10:
            dist_train = np.bincount(y_train).astype(float)
            dist_train /= np.sum(dist_train)
            dist_valid = np.bincount(y_valid).astype(float)
            dist_valid /= np.sum(dist_valid)
            logging.debug('distribution train %s - valid %s' % (dist_train, dist_valid))

        strat_train = None
        strat_valid = None
        if self.stratifier is not None:
            y_train, strat_train = self.stratifier.inverse_transform(y_train)
            y_valid, strat_valid = self.stratifier.inverse_transform(y_valid)

        if w_train is not None:
            self.classifier.fit(X_train, y_train, w_train)
        else:
            self.classifier.fit(X_train, y_train)

        if self.max_false_positive_rate is not None:
            score = score_classifier(self.classifier, X_valid)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_valid, score, sample_weight=w_valid)
            # note: fpr is increasing
            # round for few positives
            #pos = np.count_nonzero(y_valid)
            assert(fpr[0] <= fpr[-1])
            #fpr_target = math.floor(self.max_false_positive_rate * pos)/(1.0 * pos)
            self.thresh = np.interp(self.max_false_positive_rate, fpr, thresholds)

        return self

    def predict_proba(self, X):
        return score_classifier(self.classifier, X)

    def predict(self, X):
        """compute the score of the embedded classifier, then apply tuned threshold"""
        score = score_classifier(self.classifier, X)
        return (score > self.thresh).astype(int)


class MLPClassifierWrapper(MLPClassifier):
    """
    Wrapper for MLPClassifier.
    Purpose: ability to specify width of hidden layers with keywords instead of fixed numbers
    """

    def __init__(self, width='sqrt', depth=1,
                 activation="tanh",  # changed!
                 solver='adam', alpha=0.1,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.01, power_t=0.5, max_iter=10000,
                 shuffle=True, random_state=None, tol=1e-3,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8):
        self.width = width
        self.depth = depth
        sup = super(MLPClassifier, self)
        sup.__init__(hidden_layer_sizes=(100,), activation=activation,
                     solver=solver, alpha=alpha,
                     batch_size=batch_size, learning_rate=learning_rate,
                     learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, loss='log_loss',
                     shuffle=shuffle, random_state=random_state, tol=tol,
                     verbose=verbose, warm_start=warm_start, momentum=momentum,
                     nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                     validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                     epsilon=epsilon)

    def fit(self, X, y, sample_weight=None):

        if self.width not in ['sqrt', 'log2', 'full']:
            raise ValueError('invalid layer width: %s' % self.width)

        n = X.shape[1]
        if self.width == 'sqrt':
            n = int(math.ceil(math.sqrt(n)))
        else:
            n = int(math.ceil(math.log2(n)))
        self.hidden_layer_sizes = (n) * self.depth
        super(MLPClassifier, self).fit(X, y)


# properties of sklearn classifiers
def classifier_has_random_state(method):
    return not(method.startswith('knn') or method in ('dummy', 'nb', 'qda'))


def classifier_has_cache_size(method):
    return method in ('svm_rbf', 'svm_poly', 'svm_sig')


def classifier_allows_weights(method):
    return not(method.startswith('knn') or method in ('qda', 'mlp'))


def get_classifier(method, hyper_param_file=None, binary_features=True, **kwargs):
    """
    instantiate an sklearn classifier according to string.
    read hyper parameters from hyper_parm_file, if given.
    """

    classifier = None

    if ((not classifier_has_random_state(method)) or
        # workaround: xgb does not allow seed=None
            (method == 'xgb' and 'random_state' in kwargs and kwargs['random_state'] is None)):

        try:
            kwargs.pop('random_state')
        except:
            pass
    if not classifier_has_cache_size(method):
        try:
            kwargs.pop('cache_size')
        except:
            pass

    if method[:3] == 'knn':
        k = int(method[3:])
        if binary_features:
            metric = 'jaccard'
        else:
            metric = 'minkowski'
        classifier = neighbors.KNeighborsClassifier(k, metric=metric, weights='distance', **kwargs)
    elif method == 'lr':
        classifier = LogisticRegression(max_iter=200, class_weight='balanced', **kwargs)
    elif method == 'rf':
        classifier = RandomForestClassifier(n_estimators=200, class_weight='balanced', **kwargs)
    elif method == 'gbm':
        classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, **kwargs)
    elif method == 'svm_lin':
        classifier = svm.LinearSVC(max_iter=10000, dual=False, **kwargs)
    elif method == 'svm_poly':
        # note: probability computation for svm is expensive and not needed
        classifier = svm.SVC(probability=False, kernel='poly', **kwargs)
    elif method == 'svm_rbf':
        classifier = svm.SVC(probability=False, kernel='rbf', **kwargs)
    elif method == 'svm_sig':
        classifier = svm.SVC(probability=False, kernel='sigmoid', **kwargs)
    elif method == 'ada':
        classifier = AdaBoostClassifier(n_estimators=500, **kwargs)
    elif method == 'gp':
        classifier = GaussianProcessClassifier(**kwargs)
    elif method == 'mlp':
        classifier = MLPClassifierWrapper(**kwargs)
    elif method == 'nb':
        if binary_features:
            classifier = BernoulliNB(**kwargs)
        else:
            classifier = GaussianNB(**kwargs)
    elif method == 'qda':
        classifier = QuadraticDiscriminantAnalysis(**kwargs)
    elif method == 'xgb':
        from xgboost import XGBClassifier
        classifier = XGBClassifier(n_jobs=-1, nthread=-1, **kwargs)
    elif method == 'dummy':
        classifier = DummyClassifier(strategy='uniform')
    else:
        raise ValueError('unknown classifier: ' + str(method))

    if hyper_param_file is not None:
        if not os.path.exists(hyper_param_file):
            raise ValueError('requested hyperparameters file not found: %s' % hyper_param_file)
        logging.info('reading hyperparameters from file: %s' % hyper_param_file)
        hyper_params = json.load(open(hyper_param_file, 'r'))
        if 'comment' in hyper_params:
            hyper_params.pop('comment')
        logging.info(hyper_params)
        # remove possible pipeline prefix
        hyper_params = dict([(k.split('__')[-1], v) for k, v in hyper_params.items()])
        classifier.set_params(**hyper_params)

    return classifier


def plot_folds(label_score_weight, dir_out='images', tag='test', plot_type='pr', chosen_thresh=0.0):
    """plot precision-recall or roc curves from multiple cross validation folds"""

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b',  'm', 'c', 'y', 'k', 'w'])))
    cyc_line = PlotCycler('line')

    if plot_type == 'thresh':
        plt.title("chosen threshold: %.3f" % chosen_thresh)
        first = True
        for label, score, weight in label_score_weight:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, score, sample_weight=weight)
            label_t = None
            label_f = None
            if first:
                label_t = "tpr"
                label_f = "fpr"
                first = False
            l = cyc_line.next()
            plt.plot(thresholds, tpr, "b--", label=label_t, linestyle=l)
            plt.plot(thresholds, fpr, "g-", label=label_f, linestyle=l)
            if chosen_thresh is not None:
                plt.plot((chosen_thresh, chosen_thresh), (0.0, 1.0), 'k-')
        plt.ylabel("score")
        plt.xlabel("decision threshold")
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(dir_out, plot_type + '_' + tag + '.png'))
        plt.close()
        return

    elif plot_type == 'violin':
        score_max = -1
        score_min = 1e10

        for ls in label_score_weight:
            score_max = max(score_max, max(ls[1]))
            score_min = min(score_min, min(ls[1]))

        fig, ax = plt.subplots(1, len(label_score_weight))
        fold = -1
        for label, score, weight in label_score_weight:
            fold += 1
            ax[fold].violinplot([score[~label], score[label]], showmedians=True)
            ax[fold].grid(axis='both')
            pearson = pearsonr(label, score)[0]
            ax[fold].set_title('%.3f' % pearson)
            ax[fold].set_ylim([score_min, score_max])

        plt.suptitle('score distributions for %s, pearson =' % tag, fontsize=15)
        plt.savefig(os.path.join(dir_out, plot_type + '_' + tag + '.png'))
        plt.close()
        return

    elif plot_type == 'pr':
        analyze_func = precision_recall_curve
        agg_func = sklearn.metrics.average_precision_score
        agg_name = 'ap'
        plt.xlabel('recall')
        plt.ylabel('precision')
        title_prefix = 'precision-recall'
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    elif plot_type == 'roc':
        def analyze_func(y, x, sample_weight):
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, x, sample_weight=sample_weight)
            return tpr, fpr, thresholds
        agg_func = sklearn.metrics.roc_auc_score
        agg_name = 'area'
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        title_prefix = 'roc'
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:
        raise ValueError('unknown plot type: ' + plot_type)

    aps = []
    for label, score, weight in label_score_weight:
        y, x, thresholds = analyze_func(label, score, sample_weight=weight)
        aps.append(agg_func(label, score, sample_weight=weight))

        # sort and dedupe points!
        xys = [xy for xy in sorted(zip(x, y))]
        xys_new = []
        for i in range(len(xys) - 1):
            if xys[i][0] != xys[i + 1][0]:
                xys_new.append(xys[i])
        xys_new.append(xys[-1])
        xys = xys_new

        x = [xy[0] for xy in xys]
        y = [xy[1] for xy in xys]

        plt.step(x, y, alpha=0.5, color='k', linestyle=cyc_line.next(), where='post')
        plt.fill_between(x, y, step='post', alpha=0.1, color='b')

    plt.title('%s curves for %s: %s=%.3f' % (title_prefix, tag, agg_name, np.mean(aps)))
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(os.path.join(dir_out, plot_type + '_' + tag + '.png'))
    plt.close()


def score_classifier(classifier, X):
    """apply a classifier"""

    if hasattr(classifier, 'predict_proba'):
        score = classifier.predict_proba(X)[:, 1]
    else:
        # probability computation for svm is expensive and not needed
        score = classifier.decision_function(X)

    # important for evaluation metrics: add some random noise to break possible ties!
    score += 1e-10 * (random.random() - 0.5)
    return score


def enrichment(labels, score, at=0.1):
    """early enrichment metric"""
    score, labels = zip(*sorted(zip(score, labels)))
    score = [x for x in reversed(score)]
    labels = [x for x in reversed(labels)]
    avg = np.average(labels)
    if avg == 0.0:
        return float('NaN')
    head = min(max(1, int(round(at * len(labels)))), len(labels) - 2)
    return np.average(labels[:head]) / avg


def tpr_at_fpr(labels, score, at=0.2, **kwargs):
    """recall when tolerating a set level of false positives"""
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, score, **kwargs)
    return np.interp(at, fpr, tpr, left=0.0, right=1.0)


def fpr_tpr_at_thresh(labels, score, thresh, sample_weight=None):
    """false and true positive rates at given threshold"""

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, score, sample_weight=sample_weight)
    # note: thresholds are decreasing!
    thresholds = list(reversed(thresholds))
    fpr = list(reversed(fpr))
    tpr = list(reversed(tpr))
    assert(thresholds[0] <= thresholds[-1])
    fpr_at_thresh = np.interp(thresh, thresholds, fpr, left=1.0, right=0.0)
    tpr_at_thresh = np.interp(thresh, thresholds, tpr, left=1.0, right=0.0)
    return fpr_at_thresh, tpr_at_thresh


def wrap_score(y_true, y_pred, sample_weight=None, func=None, **kwargs):
    """auxiliary closure to pass on sample weights to scoring metric"""
    if sample_weight is not None:
        sample_weight = sample_weight.iloc[y_true.index.values].values.reshape(-1)
        check_consistent_length(sample_weight, y_true)
    return func(y_true, y_pred, sample_weight=sample_weight, **kwargs)


def eval_score(y, score, sample_weight=None, thresh=None, rec=None):
    """calculate several evaluation metrics for predictions; store results in rec dictionary."""

    if rec is None:
        rec = {}
    if 'auc' not in rec:
        rec['auc'] = []
        rec['pr'] = []
        rec['pearson'] = []
        rec['fpr'] = []
        rec['tpr'] = []
        rec['thresh'] = []

    auc = sklearn.metrics.roc_auc_score(y, score, sample_weight=sample_weight)
    rec['auc'].append(auc)

    average_precision = sklearn.metrics.average_precision_score(y, score, sample_weight=sample_weight)
    rec['pr'].append(average_precision)

    # TODO: sample weight?
    pearson = pearsonr(y, score)[0]
    rec['pearson'].append(pearson)

    if thresh is not None:
        fpr_at_thresh, tpr_at_thresh = fpr_tpr_at_thresh(y, score, thresh, sample_weight=sample_weight)
    else:
        fpr_at_thresh = None
        tpr_at_thresh = None

    rec['fpr'].append(fpr_at_thresh)
    rec['tpr'].append(tpr_at_thresh)
    rec['thresh'].append(thresh)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, score, sample_weight=sample_weight)
    return fpr, tpr, thresholds


def plot_optim_results(df, random_grid, method, dir='.', tag='', include_train=False):
    """plot results data frame generated by RandomizedSearchCV"""

    df = df.copy()
    random_grid = random_grid.copy()

    # note: values are actually objects, not numeric - see RandomizedSearchCV documentation
    # for comparability, convert all types to string
    # many value ranges are logarithmic, not much loss in uniform x-spacing

    pref = 'param_'
    for k, v in random_grid.items():
        random_grid[k] = [str(x) for x in v]
        #df[pref + k] = df[pref + k].astype(str)
        df[pref + k] = [str(x) if str(x) != 'nan' else 'None' for x in df[pref + k]]

    plot_names = [k for k in random_grid.keys() if pref + k in df.columns]

    # cross plots - these parameters are highly correlated
    cross_plot = None
    cross_name = None
    if method in ('gbm', 'xgb'):
        cross_plot = ['clf__n_estimators', 'clf__learning_rate']
    if method in ('svm_rbf', 'svm_poly', 'svm_sig'):
        cross_plot = ['clf__C', 'clf__gamma']
    if cross_plot is not None:
        cross_name = cross_plot[0] + '_' + cross_plot[1]
        df[pref + cross_name] = ['(%s,%s)' % (l, n) for l, n in zip(df[pref + cross_plot[0]], df[pref + cross_plot[1]])]
        plot_names.append(cross_name)

    for c in ['mean_train_score', 'mean_test_score']:
        df[c] = df[c].astype(float)
        if np.isnan(df[c]).any():
            logging.warning('%s contains NaNs!' % c)
            df[c].fillna(0.0, inplace=True)

    for p in plot_names:
        c = pref + p

        # sort parameters in the same way as in the grid
        if p != cross_name:
            sort_cols = [p]
        else:
            sort_cols = cross_plot

        for sort_col in sort_cols:
            u = np.unique(list(df[pref + sort_col]))
            u = sorted(u, key=cmp_to_key(cmp_mixed))
            df['order'] = [u.index(x) for x in df[pref + sort_col]]
            df.sort_values('order', inplace=True, kind='mergesort')  # note: must be stable sort

        u, i, x = np.unique(df[c], return_index=True, return_inverse=True)
        if include_train:
            x_jitter = [xx + 0.1 * (random.random() - 0.5) for xx in x]
            y_jitter = [yy + 0.01 * (random.random() - 0.5) for yy in df['mean_train_score']]
            plt.plot(x_jitter, y_jitter, "b.", label="train")
        x_jitter = [xx + 0.25 * (random.random() - 0.5) for xx in x]
        y_jitter = [yy + 0.01 * (random.random() - 0.5) for yy in df['mean_test_score']]
        plt.plot(x_jitter, y_jitter, "r.", label="test")

        if p == cross_name:
            # accomodate long tick labels!
            plt.xticks(range(len(u)), u[np.argsort(i)], rotation=90)
            plt.gcf().subplots_adjust(bottom=0.2)
        else:
            plt.xticks(range(len(u)), u[np.argsort(i)])

        # mean line
        if include_train:
            med = df.fillna(-1).groupby(c)[['mean_train_score']].mean()
            plt.plot(range(len(med.index)), med, "b:")
        med = df.fillna(-1).groupby(c)[['mean_test_score']].mean()
        plt.plot(range(len(med.index)), med, "r:")

        plt.margins(x=0.5)
        plt.grid(True)
        plt.title(p)
        plt.legend(loc='best')
        plt.savefig(os.path.join(dir, 'optim_%s_%s.png' % (tag, p)))
        plt.close()


def _fit_and_score(pipe, X, y, y_enc, sample_weight, test_filter, train_idx, test_idx):
    """auxiliary inner loop function to pass to Parallel"""
    X_train = X.iloc[train_idx]
    y_train_enc = y_enc.iloc[train_idx]
    f_train = test_filter[train_idx]

    X_test = X.iloc[test_idx][test_filter[test_idx]]
    y_test = y.iloc[test_idx][test_filter[test_idx]]

    if sample_weight is not None:
        w_train = sample_weight.iloc[train_idx]
        w_test = sample_weight.iloc[test_idx][test_filter[test_idx]]
    else:
        w_train = None
        w_test = None

    pipe.fit(X_train, y_train_enc, clf__sample_weight=w_train, clf__test_filter=f_train)

    score = pipe.predict_proba(X_test)

    rec = {}
    eval_score(y_test, score, w_test, pipe.named_steps['clf'].thresh, rec)
    return y_test, score, w_test, rec


def optimize_subset_params(estimator, X, y, fit_params, scoring, param_grid, param_subset, cv, n_iter, random_state, verbose, n_jobs):
    """single pass of hyper-parameter optimization.
    if n_iter is less than 70% of possible parameter space, RandomizedSearchCV is used, else GridSearchCV.
    all params not listed in param_subset are kept fixed.
    returns best parameters, score, dataframe with all results, and type of search.
    """

    if param_subset is None:
        subset_grid = param_grid
    else:
        current_params = dict([('clf__' + k, v) for k, v in estimator.named_steps['clf'].get_params().items()])

        subset_grid = {}
        for k in param_grid:
            subset_grid[k] = [current_params[k]]

        for k in param_subset:
            subset_grid[k] = param_grid[k]

    # note: if there are multiple calls to this function for single-parameter optimization, best to make results comparable!

    if random_state is not None:
        try:
            estimator.set_params(clf__random_state=random_state)
        except:
            pass

    if n_iter >= 0.7 * get_param_combinations(subset_grid):
        # if n_iter is greater than parameter space, RandomizedSearchCV throws an error - just switch to GridSearchCV
        search = GridSearchCV(estimator=estimator, param_grid=subset_grid, scoring=scoring, fit_params=fit_params,
                              cv=cv, verbose=verbose, n_jobs=n_jobs, pre_dispatch='2*n_jobs', return_train_score=True, refit=True)
    else:
        search = RandomizedSearchCV(estimator=estimator, param_distributions=subset_grid, scoring=scoring, fit_params=fit_params, n_iter=n_iter,
                                    cv=cv, verbose=verbose, random_state=random_state, n_jobs=n_jobs, pre_dispatch='2*n_jobs', return_train_score=True, refit=True)

    # most efficient to add search as a step in the pipeline! See https://stackoverflow.com/questions/43366561/use-sklearns-gridsearchcv-with-a-pipeline-preprocessing-just-once
    # however, to include pipeline parameters other than in the classifier, pipe would have to be an argument.
    #estimator.steps.append(('grid', search))
    # estimator.fit(X, y)

    logging.debug('using %s ' % (type(search).__name__))
    # logging.debug(search)
    search.fit(X, y)

    estimator.set_params(**search.best_params_)

    res = pd.DataFrame(search.cv_results_)
    # res = res.drop(['mean_fit_time', 'mean_score_time', 'std_fit_time', 'std_score_time'], axis=1)
    res['mean_test_score'] = res['mean_test_score'].astype(float)
    res.sort_values('mean_test_score', ascending=False, inplace=True)

    return search.best_params_, search.best_score_, res, type(search).__name__


# note: despite a lot of trying, I could not fit stratification by target *and* weight into this framework.

def hyper_optim(pipe, method, X, y, sample_weight=None, binary_features=True, scoring='roc_auc', fine_tune=False, n_iter=100, cv=3, seed=None, n_jobs=-1, dir_out='.', experiment='exp', overwrite=0, svm_cache_size=5000, verbose=1):
    """
    run hyperparameter optimization.
    first stage is randomized or grid search; second stage is local parameter-wise optimization until convergence (with fine_tune=True).
    Special cases for gbm, xgboost, and rf: tune tree and regularization parameters with fixed number of trees (and learn rate), then optimize
    the latter ones.
    Write output json file with best parameters (subject to parameter overwrite); filename is return value.
    Write tsv file with optimization results.
    Plot optimization results, separately for the two stages.
    """

    tag = '%s_%s' % (experiment, scoring)
    now = datetime.datetime.now()
    comment = '%s, created %s' % (tag, now.strftime('%c'))

    if method in PARAM_GRID:
        random_grid = PARAM_GRID[method]
        random_grid = dict([('clf__' + k, v) for k, v in random_grid.items()])
    else:
        raise ValueError('hyperparameter optimization not supported for %s' % method)

    #hp_file = os.path.join(dir_out, 'best_%s_%s.json' % (experiment, scoring))
    hp_file = os.path.join(dir_out, 'hyper_params_%s.json' % method)
    hp_file_b = 'hyper_params_%s.json' % method

    if not os.path.exists(hp_file):
        # backup to current directory
        hp_file_r = hp_file_b
    else:
        hp_file_r = hp_file

    hp_file_v2 = os.path.join(dir_out, 'hyper_params_v2_%s.json' % method)
    hp_file_v2_b = 'hyper_params_v2_%s.json' % method
    if not os.path.exists(hp_file_v2):
        hp_file_v2_r = hp_file_v2_b
    else:
        hp_file_v2_r = hp_file_v2

    if not overwrite:
        if fine_tune and os.path.exists(hp_file_v2_r):
            logging.info('using previously generated hyperparameters from file %s' % hp_file_v2_r)
            return hp_file_v2_r
        if (not fine_tune) and os.path.exists(hp_file_r):
            logging.info('using previously generated hyperparameters from file %s' % hp_file_r)
            return hp_file_r

    # workaround to allow for sample weights in score functions:
    # wrap target (y) in pandas DataFrame, pass the whole weight column statically as scorer argument; then, at call
    # time, select fold subset based on the preserved *index* of y DataFrame for fold.
    # see: https://stackoverflow.com/questions/49581104/sklearn-gridsearchcv-not-using-sample-weight-in-score-function

    # translate string to scorer for custom metrics and to pass sample weights
    scorer = scoring
    if scoring == 'tpr_at_fpr':
        scorer = make_scorer(wrap_score, func=tpr_at_fpr, greater_is_better=True, sample_weight=sample_weight)
    elif sample_weight is not None:
        if scoring == 'accuracy':
            scorer = make_scorer(wrap_score, func=accuracy_score, sample_weight=sample_weight)
        elif scoring == 'roc_auc':
            scorer = make_scorer(wrap_score, func=roc_auc_score, needs_threshold=True, sample_weight=sample_weight)
        elif scoring == 'average_precision':
            scorer = make_scorer(wrap_score, func=average_precision_score,
                                 needs_threshold=True, sample_weight=sample_weight)
        elif scoring == 'log_loss':
            scorer = make_scorer(wrap_score, func=log_loss, greater_is_better=False,
                                 needs_proba=True, sample_weight=sample_weight)
        else:
            raise ValueError('weighted hyperparameter optimization not implemented for %s' % scoring)

    if sample_weight is not None:
        y = pd.DataFrame(y)

    search_type = None
    best_score = -1e20
    best_params = None
    res = None

    cv_local = 5

    if fine_tune and os.path.exists(hp_file):
        # read first-stage model from file
        classifier = get_classifier(method, hyper_param_file=hp_file_r,
                                    binary_features=binary_features, random_state=seed, cache_size=svm_cache_size)
        pipe.steps.append(('clf', classifier))
        best_params = json.load(open(hp_file_r, 'r'))
        if 'comment' in best_params:
            best_params.pop('comment')
        best_score = -1e10
    else:
        classifier = get_classifier(method, binary_features=binary_features,
                                    random_state=seed, cache_size=svm_cache_size)
        pipe.steps.append(('clf', classifier))

        logging.info('starting hyperparameter optimization for %s' % tag)

        if method in ('gbm', 'xgb'):
            # for gbm and xgboost, ideally, should be 1) tree params 2) regularization params
            # see: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

            # fix high learning rate
            pipe.set_params(clf__learning_rate=0.1, clf__n_estimators=100)

            # 1. main optimization
            subset = [k for k in random_grid if k not in ('clf__n_estimators', 'clf__learning_rate')]
            best_params, best_score, res, search_type = optimize_subset_params(pipe, X, y, fit_params={
                                                                               'clf__sample_weight': sample_weight}, scoring=scorer, param_grid=random_grid, param_subset=subset, cv=cv, n_iter=n_iter, random_state=seed, verbose=verbose, n_jobs=n_jobs)
            logging.info('best (non-lr) parameters after first stage of hyperparameter optimization (%s) for %s: (score %.3f)' %
                         (search_type, tag, best_score))
            logging.info(best_params)

            # 2. optimize lr and #trees
            logging.info('optimizing learn rate and number of trees for %s' % tag)
            best_params, best_score, res_step, _ = optimize_subset_params(pipe, X, y, fit_params={'clf__sample_weight': sample_weight}, scoring=scorer, param_grid=random_grid, param_subset=(
                'clf__n_estimators', 'clf__learning_rate'), cv=cv_local, n_iter=n_iter, random_state=seed, verbose=verbose, n_jobs=n_jobs)
            res = res.append(res_step, ignore_index=True, sort=True)
            res['mean_test_score'] = res['mean_test_score'].astype(float)
            res.sort_values('mean_test_score', ascending=False, inplace=True)
            logging.info('best number of trees for tuning %s at lr=%.3f: %d, score = %.3f' %
                         (tag, best_params['clf__learning_rate'], best_params['clf__n_estimators'], best_score))

        elif method == 'rf':
            # similarly as for gradient boosting, number of estimators might overpower all other signals - do it in two steps
            pipe.set_params(clf__n_estimators=100)

            # 1. optimize everything except n_estimators
            subset = [k for k in random_grid if k not in ('clf__n_estimators')]
            best_params, best_score, res, search_type = optimize_subset_params(pipe, X, y, fit_params={
                                                                               'clf__sample_weight': sample_weight}, scoring=scorer, param_grid=random_grid, param_subset=subset, cv=cv, n_iter=n_iter, random_state=seed, verbose=verbose, n_jobs=n_jobs)

            logging.info('best parameters after first stage of hyperparameter optimization (%s) for %s: (score %.3f)' % (
                search_type, tag, best_score))
            logging.info(best_params)

            # 2. optimize #trees
            logging.info('optimizing number of trees for %s' % tag)
            best_params, best_score, res_step, _ = optimize_subset_params(pipe, X, y, fit_params={'clf__sample_weight': sample_weight}, scoring=scorer, param_grid=random_grid, param_subset=(
                'clf__n_estimators',), cv=cv_local, n_iter=n_iter, random_state=seed, verbose=verbose, n_jobs=n_jobs)
            res = res.append(res_step, ignore_index=True, sort=True)
            res['mean_test_score'] = res['mean_test_score'].astype(float)
            res.sort_values('mean_test_score', ascending=False, inplace=True)
            logging.info('best number of trees for tuning %s: %d, score = %.3f' %
                         (tag, best_params['clf__n_estimators'], best_score))

        else:
            # method other than rf, gbm, xgb
            best_params, best_score, res, search_type = optimize_subset_params(pipe, X, y, fit_params={
                                                                               'clf__sample_weight': sample_weight}, scoring=scorer, param_grid=random_grid, param_subset=None, cv=cv, n_iter=n_iter, random_state=seed, verbose=verbose, n_jobs=n_jobs)
            logging.info('best parameters after first stage of hyperparameter optimization (%s) for %s: (score %.3f)' % (
                search_type, tag, best_score))
            logging.info(best_params)

        best_params['comment'] = comment
        json.dump(best_params, open(hp_file, 'w'))
        hp_file_r = hp_file
        # backup to current directory as well
        if not os.path.exists(hp_file_b):
            json.dump(best_params, open(hp_file_b, 'w'))
        best_params.pop('comment')

        res_file = os.path.join(dir_out, 'optim_%s.tsv' % tag)
        res.to_csv(res_file, sep='\t', index=False, float_format='%.3g')

        plot_optim_results(res, random_grid, method, dir_out, tag)

    if fine_tune:

        # parameter grid might have changed from previous optimization run!
        current_params = dict([('clf__' + k, v) for k, v in pipe.named_steps['clf'].get_params().items()])

        for k in random_grid:
            if k not in best_params:
                best_params[k] = current_params[k]

        it = 0
        if search_type == 'GridSearchCV':
            logging.info('no local optimization necessary - performed grid search in first step')
        else:
            res = None
            changed = True
            first = True
            while changed:
                logging.info('starting local optimization, round %d' % it)
                it += 1
                changed = False
                for param in random_grid:
                    if len(random_grid[param]) == 1:
                        continue
                    # fix everything except this one parameter
                    # this makes it easier to collect and align results afterwards
                    best_params_step, best_score_step, res_step, search_type = optimize_subset_params(pipe, X, y, fit_params={'clf__sample_weight': sample_weight}, scoring=scorer, param_grid=random_grid, param_subset=[
                                                                                                      param], cv=cv_local, n_iter=1e6, random_state=None, verbose=verbose, n_jobs=min(n_jobs, cv_local * len(random_grid[param])))
                    logging.debug('optimizing parameter %s, old (%s, %.5g) -> new (%s, %.5g)' %
                                  (param, best_params[param], best_score, best_params_step[param], best_score_step))
                    logging.debug(pipe.named_steps['clf'].get_params())
                    if best_params_step[param] == best_params[param]:
                        if best_score_step < best_score:
                            logging.warning('score decreased with same parameter value! parameter %s, old (%s, %.5g) -> new (%s, %.5g)' %
                                            (param, best_params[param], best_score, best_params_step[param], best_score_step))
                        if first:
                            best_score = best_score_step
                            first = False
                    else:
                        # best param value changed
                        if best_score_step <= best_score:
                            logging.warning('score decreased with different parameter value! parameter %s, old (%s, %.5g) -> new (%s, %.5g)' % (
                                param, best_params[param], best_score, best_params_step[param], best_score_step))
                        else:
                            changed = True
                            first = False
                            logging.info('changed parameter %s, old (%s, %.5g) -> new (%s, %.5g)' %
                                         (param, best_params[param], best_score, best_params_step[param], best_score_step))
                            best_score = best_score_step
                            best_params = best_params_step
                            pipe.set_params(**best_params)

                    if res is None:
                        res = res_step
                    else:
                        res = res.append(res_step, ignore_index=True, sort=True)

        best_params['comment'] = comment
        json.dump(best_params, open(hp_file_v2, 'w'))
        hp_file_v2_r = hp_file_v2

        # backup to current directory as well
        if not os.path.exists(hp_file_v2_b):
            json.dump(best_params, open(hp_file_v2_b, 'w'))
        best_params.pop('comment')

        logging.info('best parameters after 2nd stage of hyperparameter optimization for %s (score %.3f):' %
                     (tag, best_score))
        logging.info(best_params)

        if res is not None:
            # res = res.drop(['mean_fit_time', 'mean_score_time', 'std_fit_time', 'std_score_time'], axis=1)
            res['mean_test_score'] = res['mean_test_score'].astype(float)
            res.sort_values('mean_test_score', ascending=False, inplace=True)
            res_file = os.path.join(dir_out, 'optim_v2_%s.tsv' % tag)
            res.to_csv(res_file, sep='\t', index=False, float_format='%.3g')

            plot_optim_results(res, random_grid, method, dir_out, 'v2_' + tag)

    logging.info('done with hyperparameter optimization for %s' % tag)

    pipe.steps = pipe.steps[:-1]
    return hp_file_v2_r if fine_tune else hp_file_r


def main():
    import configargparse
    parser = configargparse.ArgParser(description='train and evaluate a molecular classifier according to several choices of fingerprint and learning methods.',
                                      formatter_class=configargparse.ArgumentDefaultsHelpFormatter, add_config_file_help=False)
    parser.add('--config', is_config_file=True, help='config file path')
    parser.add('--outDir', metavar='DIR', type=str, default='experiments',
               help='output directory for training results (model file, cross validation and optimization metrics, plots, variable importance)')
    parser.add('--logDir', metavar='DIR', type=str, default='logs', help='output directory for log files')
    parser.add('--cacheDir', metavar='DIR', type=str, default='cache', help='pipeline cache directory')
    parser.add('--data', metavar='FILE', type=str, required=True,
               help='training data, must contain a smiles and a label column')
    parser.add('--smilesCol', type=str, default='smiles', help='name of column containing the input smiles')
    parser.add('--targetCol', type=str, required=True, help='name of column containing the targets')
    parser.add('--fpType', type=csv_list, default='ecfp4',
               help='type of fingerprint %s; prop = rdkit molecular descriptors. Multiple elements mean their UNION is used, not sequentially.' % Fingerprinter.FP_TYPES)
    parser.add('--fpSize', metavar='N', type=int, default=2048, help='fingerprint size')
    parser.add('--methods', type=csv_list, default='rf,gbm,lr,svm_lin,svm_poly,svm_rbf,svm_sig,nb,qda,knn1,knn10,ada,gp,dummy,xgb',
               help='ml methods. Note: ada and gp crash on larger data sets. Add suffix _optim to do hyperparameter optimization; _optim2 with additional stage of local optimization (changing one parameter at a time).')
    parser.add('--maxFPR', type=float, default=0.2, help='false positive rate that is acceptable for threshold tuning')
    parser.add('--weightCat', type=str, default='',
               help='integer column for categories of sample weights (see catToSampleWeight). Used for cross validation stratification.')
    parser.add('--catToSampleWeight', type=csv_list, default='',
               help='mapping of categories to sample weights (cat1=w1,cat2=w2,..)')
    parser.add('--testFilterCat', type=int, default=-1, help='use only this category for test set evaluation')
    parser.add('--cv', metavar='N', type=int, default=5, help='number of cross validation folds')
    parser.add('--cvJobs', metavar='N', type=int, default=-1, help='number of cpus to use for cross validation')
    parser.add('--validFrac', metavar='F', type=float, default=0.2,
               help='Fraction of training data to hold out for threshold tuning')
    parser.add('--seed', metavar='N', type=int, default=42, help='random seed')
    parser.add('--resultsOverwrite', metavar='N', type=int,
               choices=[0, 1], default=0, help='Overwrite previous results')
    parser.add('--optimScoring', type=str, default='average_precision', choices=[
               'average_precision', 'roc_auc', 'f1', 'neg_log_loss', 'accuracy'], help='metrics scoring function for hyperparameter optimization')
    parser.add('--optimIter', metavar='N', type=int, default=100,
               help='number of iterations for hyperparameter optimization')
    parser.add('--optimCV', metavar='N', type=int, default=3, help='number of CV folds for hyperparameter optimization')
    parser.add('--optimOverwrite', metavar='N', type=int,
               choices=[0, 1], default=1, help='If zero and previous optimization results exists, use it')
    parser.add('--optimJobs', metavar='N', type=int, default=-1,
               help='number of cpus to use for hyperparameter optimization')
    parser.add('--logToFile', metavar='N', type=int, choices=[0, 1], default=1, help='log to file')
    parser.add('--cachePipeline', metavar='N', type=int, choices=[0, 1], default=1, help='use cached pipelines')
    parser.add('--refit', metavar='N', type=int,
               choices=[0, 1], default=1, help='refit final model on all data, save to .pkl file, and calculate variable importance')
    parser.add('--saveComment', type=str, default='', help='comment to store in refit model')
    parser.add('--svmCacheMemFrac', metavar='F', type=float, default=0.5,
               help='maximum percentage of total memory to use for svm training. Crucial for svm training speed.')
    parser.add('--verbose', metavar='N', type=int, choices=[
               0, 1, 2, 3], default=1, help='verbose logging [0 - no logging; 1 - info, no libraries; 2 - including libraries; 3 - debug')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # fpType is more used as a string identifier - join here
    args.fpType = '~'.join(sorted(args.fpType))
    # init logging
    log_file = None
    if args.logToFile > 0:
        mkdir_p('logs')
        log_file = 'logs/log_%s_%s.txt' % (core_basename(args.data), args.fpType)
        sys.stderr.write('logfile: %s\n' % log_file)

    # note: detect logging level rather than having a verbose flag!
    # verbose flag will lead underutilization of cached pipelines

    level = logging.ERROR
    if args.verbose == 0:
        level = logging.WARNING
    elif args.verbose <= 2:
        level = logging.INFO
    elif args.verbose == 3:
        level = logging.DEBUG

    logging.basicConfig(level=level,
                        format="[%(asctime)s\t%(process)d\t%(levelname)s]\t%(message)s",
                        datefmt="%Y%m%d %H:%M:%S",
                        filename=log_file)

    verbose_sklearn = max(args.verbose - 1, 0)
    verbose_mem = 11 if verbose_sklearn == 2 else verbose_sklearn
    verbose_par = 50 if args.verbose >= 2 else 0

    # compute maximum cache size for svm training

    any_svm_optim = False

    for m in args.methods:
        if m in ('svm_rbf_optim', 'svm_poly_optim', 'svm_sig_optim', 'svm_rbf_optim2', 'svm_poly_optim2', 'svm_sig_optim2'):
            any_svm_optim = True

    mem = psutil.virtual_memory().total
    procs = psutil.cpu_count()
    n_parallel = args.cv
    svm_cache_size = None
    if any_svm_optim:
        n_parallel = max(n_parallel, args.optimIter * args.optimCV)
        if args.optimJobs > 0:
            n_parallel = min(n_parallel, args.optimJobs)
    n_parallel = min(n_parallel, procs)
    svm_cache_size = int(args.svmCacheMemFrac * mem / n_parallel)

    logging.debug('total mem %s procs %s cache mem %s' % (mem, procs, svm_cache_size))

    # read the data

    df = pd.read_csv(args.data, delimiter='\t')
    X = df[args.smilesCol]

    y = df[args.targetCol]
    dist = np.bincount(y).astype(float)
    dist /= np.sum(dist)
    logging.debug('label distribution: %s', dist)

    # weight management
    if (args.weightCat is not None) and (len(args.weightCat) > 0):
        if args.weightCat not in df.columns:
            raise ValueError('No such column: %s' % args.weightCat)
        weight_cat = df[args.weightCat]
        us = np.unique(weight_cat)
        if len(args.catToSampleWeight) < len(us):
            raise ValueError('not enough values specified in catToSampleWeight, needed: %s' % us)
        d = {}
        for x in args.catToSampleWeight:
            k, v = x.split('=')
            d[int(k)] = float(v)
        args.catToSampleWeight = d
        for u in us:
            if u not in args.catToSampleWeight:
                raise ValueError('No weight specified for category %s' % u)
        sample_weight = pd.Series([args.catToSampleWeight[x] for x in weight_cat])
        sample_weight /= sample_weight.mean()
        le_strat = LabelStratificationEncoder()
        y_enc = le_strat.fit_transform(y, weight_cat)
        if args.testFilterCat < 0:
            test_filter = np.array([True] * len(df))
        else:
            test_filter = np.array(weight_cat == args.testFilterCat)
            logging.debug('filter test set: %.3f', np.mean(test_filter))
    else:
        weight_cat = None
        sample_weight = None
        if args.testFilterCat >= 0:
            logging.warning('testFilterCat specifified without weightCat, ignoring!')
        test_filter = np.array([True] * len(df))

    # set up feature processing pipeline

    if args.cachePipeline > 0:
        mkdir_p(args.cacheDir)
        mem = Memory(cachedir=args.cacheDir, compress=True, verbose=verbose_mem)
    else:
        mem = None

    le_strat = LabelStratificationEncoder()
    y_enc = le_strat.fit_transform(y, weight_cat)

    pipe, binary_features = create_fp_pipe(args.fpType, args.fpSize, mem)

    for method in args.methods:

        hyper_opt = False
        fine_tune = False
        hyper_param_file = None
        if method.endswith('_optim'):
            hyper_opt = True
            method = method[:(-len('_optim'))]
        if method.endswith('_optim2'):
            hyper_opt = True
            fine_tune = True
            method = method[:(-len('_optim2'))]

        experiment = '%s_%s_%s' % (core_basename(args.data), args.fpType, method)
        dir_out = os.path.join(args.outDir, experiment)
        mkdir_p(dir_out)
        sys.stderr.write('output directory: %s\n' % dir_out)

        # note: some classifiers' fit() function have no weights argument
        sample_weight_m = sample_weight if classifier_allows_weights(method) else None

        if hyper_opt:
            hyper_param_file = hyper_optim(pipe, method, X, y, sample_weight=sample_weight_m, binary_features=binary_features,
                                           scoring=args.optimScoring, fine_tune=fine_tune, n_iter=args.optimIter,
                                           cv=args.optimCV, seed=args.seed, n_jobs=args.optimJobs, dir_out=dir_out, experiment=experiment,
                                           overwrite=(args.optimOverwrite > 0), svm_cache_size=svm_cache_size, verbose=args.verbose)

        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

        classifier = get_classifier(method, binary_features=binary_features,
                                    hyper_param_file=hyper_param_file, random_state=args.seed, cache_size=svm_cache_size)
        tuner = ThresholdTuner(classifier, 0.5, args.validFrac, max_false_positive_rate=args.maxFPR,
                               random_state=args.seed, stratifier=le_strat)
        pipe.steps.append(('clf', tuner))

        logging.info('starting cross validation for %s' % experiment)
        logging.debug(pipe.steps)
        outs = Parallel(
            n_jobs=args.cvJobs, verbose=verbose_par,
        )(delayed(_fit_and_score)(clone(pipe), X, y, y_enc, sample_weight_m, test_filter, train_idx, test_idx)
          for (train_idx, test_idx) in cv.split(X, y_enc))

        # aggregate results from all cv folds

        (y_test, score, w_test, recs) = zip(*outs)
        rec = {}
        for k in recs[0].keys():
            rec[k] = [r[k] for r in recs]

        overall_thresh = np.mean(rec['thresh'])

        preds = list(zip(y_test, score, w_test))
        for plot_type in ['pr', 'roc', 'violin', 'thresh']:
            plot_folds(preds, dir_out=dir_out, plot_type=plot_type, tag=experiment, chosen_thresh=overall_thresh)

        logging.info('results:')
        opt_level = 0
        if fine_tune:
            opt_level = 2
        elif hyper_opt:
            opt_level = 1
        results = [args.data, args.fpType, method, opt_level]

        rec['fpr_all'] = []
        rec['tpr_all'] = []
        # what would tpr and fpr be when using the overall average threshold?
        for (y_test, score, w_test, recs) in outs:
            f, t = fpr_tpr_at_thresh(y_test, score, overall_thresh, sample_weight=w_test)
            rec['fpr_all'].append(f)
            rec['tpr_all'].append(t)

        for name in ['pr', 'auc', 'pearson', 'fpr', 'tpr', 'fpr_all', 'tpr_all', 'thresh']:
            results.extend([np.mean(rec[name]), np.std(rec[name])])
            logging.info('%s\t%.3g' % (name, np.mean(rec[name])))

        results.append(classifier.get_params())
        results.append(args)

        columns = ['data', 'fp', 'classifier', 'opt', 'mean_ap', 'std_ap', 'mean_auc', 'std_auc', 'mean_pearson',
                   'std_pearson', 'mean_fpr_cv', 'std_fpr_cv', 'mean_tpr_cv',
                   'std_tpr_cv', 'mean_fpr_all', 'std_fpr_all', 'mean_tpr_all',
                   'std_tpr_all', 'mean_thresh', 'std_thresh', 'params', 'args']

        results = pd.DataFrame(OrderedDict((k, [v]) for k, v in zip(columns, results)))
        file_res = os.path.join(dir_out, 'eval_%s.tsv' % experiment)
        df_to_csv_append(results, file_res, float_format='%.4g', overwrite=args.resultsOverwrite)

        if args.refit > 0:
            logging.info('refitting %s on all data' % method)

            pipe.fit(X, y, clf__sample_weight=sample_weight_m, clf__test_filter=test_filter)

            pipe.named_steps['clf'].thresh = overall_thresh
            pipe_out = copy.deepcopy(pipe)  # clone() does not include fitted params
            prepare_pipeline_for_scoring(pipe_out)
            pipe_out.comment = args.saveComment
            now = datetime.datetime.now()
            pipe_out.created = now.strftime('%c')
            pipe_out.target_name = args.targetCol
            # note: files for rf models can be huge!
            joblib.dump(pipe_out, os.path.join(dir_out, 'model_%s.pkl.xz' % experiment), compress=('xz', 9))

        classifier = pipe.named_steps['clf'].classifier

        # write out feature importances, if available
        if args.refit > 0:
            importances = None
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif method == 'lr':
                # conveniently, we dropped the classifier from the pipe, so pure feature generation
                X_trans = np.array(pipe.transform(X))
                importances = np.std(X_trans, 0) * classifier.coef_
                importances = importances.flatten()
                importances /= np.absolute(importances).sum()

            if importances is not None:
                feature_names = get_feature_names(pipe, -2)
                file_imp = os.path.join(dir_out, 'importance_%s.tsv' % experiment)

                imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
                imp.sort_values('importance', ascending=False, inplace=True)
                imp.to_csv(file_imp, sep='\t', index=False)

        # drop classifier to make room for next method
        pipe.steps = pipe.steps[:-1]

    logging.info('all done with %s' % args.fpType)


if __name__ == "__main__":
    main()
