# molecule_classification

"[Auto-Sklearn](https://github.com/automl/auto-sklearn) for Chemistry".

## Overview

Train and run machine-learned classifiers for molecular classification tasks; for
example, predict properties like solubility, atomization energies, or biological
affinities in QSAR projects.

The general purpose is to "throw anything easily computable at the wall and see what 
sticks". Molecules are input in the form of [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) strings. From these, different types
of fingerprints (fixed-sized vectors) can be computed using [RDKit](http://www.rdkit.org/). Model training is based 
on classifiers based on [scikit-learn](http://scikit-learn.org/stable/) or [xgboost](https://github.com/dmlc/xgboost). More advanced representations (e.g., graph
convolutions) or ML models (e.g., deep neural nets) are *not* supported. However, the package
does automate hyper-parameter optimization for the supported model types.

The project supplies two scripts:
* *train_mol_class.py*: Train a scikit-learn or xgboost model
* *predict_mol_class.py*: Generate predictions for an input file with a previously trained model

Please see the `demo` directory for a usage example.

##  train_mol_class.py

For one run, a single fingerprinting method and multiple learning methods can be specified,
which are applied in turn. When multiple fingerprints are specified, e.g., 
`-fpType ecfp4,prop`, their *union* is fed to the classifier. Outputs, including cross 
validation results, optimization results, plots, and pickled model file, are 
written under directory experiments/<identifier>.

Hyper-parameter optimization by random search can be triggered by adding the suffix `_optim` 
to the classifier name. If the number of optimization iterations (`--optimIter`) exceeds 70%
of the parameter space, the script automatically switches to exhaustive grid search. Suffix 
`_optim2` additionally finds a local minimum by optimizing each hyper-parameter individually
and sequentially. Best parameters are stored in the output directory, and by default are 
re-used if they exist (by checking in the output directory *or* the current directory as a backup). 

In general, there are many possible ways to tune the decision threshold for a classifier. In this
project, it is done by limiting the expected false positive rate (`-maxFPR`). This threshold is
estimated by holding out a fraction of the training data (`--validFrac`).

There are a couple of command line parameters related to dealing with sub-categories of training
data; one use case is to combine a large public data set with a smaller but more important proprietary 
data set. To increase the weight of this subset, we add an indicator column `-weightCat` in the input 
data, and assign a higher weight using `-catToSampleWeight`. Additionally, if we are interested in the 
cross validation results *only* on the subset, we use the `-testFilterCat` option. Note that if a 
`--weightCat` is specified, cross validation is stratified with respect to it (in addition to the
target column, as usual).

`train_mol_class.py` uses [configargparse](https://github.com/bw2/ConfigArgParse), so all options can be supplied in a configuation file.


```
usage: train_mol_class.py [-h] [--config CONFIG] [--outDir DIR] [--logDir DIR]
                          [--cacheDir DIR] --data FILE [--smilesCol SMILESCOL]
                          --targetCol TARGETCOL [--fpType FPTYPE] [--fpSize N]
                          [--methods METHODS] [--maxFPR MAXFPR]
                          [--weightCat WEIGHTCAT]
                          [--catToSampleWeight CATTOSAMPLEWEIGHT]
                          [--testFilterCat TESTFILTERCAT] [--cv N]
                          [--cvJobs N] [--validFrac F] [--seed N]
                          [--resultsOverwrite N] [--optimScoring OPTIMSCORING]
                          [--optimIter N] [--optimCV N] [--optimOverwrite N]
                          [--optimJobs N] [--logToFile N] [--cachePipeline N]
                          [--refit N] [--saveComment SAVECOMMENT]
                          [--svmCacheMemFrac F] [--verbose N]

train and evaluate a molecular classifier according to several choices of
fingerprint and learning methods.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file path (default: None)
  --outDir DIR          output directory for training results (model file,
                        cross validation and optimization metrics, plots,
                        variable importance) (default: experiments)
  --logDir DIR          output directory for log files (default: logs)
  --cacheDir DIR        pipeline cache directory (default: cache)
  --data FILE           training data, must contain a smiles and a label
                        column (default: None)
  --smilesCol SMILESCOL
                        name of column containing the input smiles (default:
                        smiles)
  --targetCol TARGETCOL
                        name of column containing the targets (default: None)
  --fpType FPTYPE       type of fingerprint ['ecfp4', 'ecfp6', 'ecfp12',
                        'maccs', 'daylight', 'ap', 'torsion', 'prop']; prop =
                        rdkit molecular descriptors. Multiple elements mean
                        their UNION is used, not sequentially. (default:
                        ecfp4)
  --fpSize N            fingerprint size (default: 2048)
  --methods METHODS     ml methods. Note: ada and gp crash on larger data
                        sets. Add suffix _optim to do hyperparameter
                        optimization; _optim2 with additional stage of local
                        optimization (changing one parameter at a time).
                        (default: rf,gbm,lr,svm_lin,svm_poly,svm_rbf,svm_sig,n
                        b,qda,knn1,knn10,ada,gp,dummy,xgb)
  --maxFPR MAXFPR       false positive rate that is acceptable for threshold
                        tuning (default: 0.2)
  --weightCat WEIGHTCAT
                        integer column for categories of sample weights (see
                        catToSampleWeight). Used for cross validation
                        stratification. (default: )
  --catToSampleWeight CATTOSAMPLEWEIGHT
                        mapping of categories to sample weights
                        (cat1=w1,cat2=w2,..) (default: )
  --testFilterCat TESTFILTERCAT
                        use only this category for test set evaluation
                        (default: -1)
  --cv N                number of cross validation folds (default: 5)
  --cvJobs N            number of cpus to use for cross validation (default:
                        -1)
  --validFrac F         Fraction of training data to hold out for threshold
                        tuning (default: 0.2)
  --seed N              random seed (default: 42)
  --resultsOverwrite N  Overwrite previous results (default: 0)
  --optimScoring OPTIMSCORING
                        scoring function for hyperparameter optimization
                        (default: average_precision)
  --optimIter N         number of iterations for hyperparameter optimization
                        (default: 100)
  --optimCV N           number of CV folds for hyperparameter optimization
                        (default: 3)
  --optimOverwrite N    If zero and previous optimization results exists, use
                        it (default: 1)
  --optimJobs N         number of cpus to use for hyperparameter optimization
                        (default: -1)
  --logToFile N         log to file (default: 1)
  --cachePipeline N     use cached pipelines (default: 1)
  --refit N             refit final model on all data, save to .pkl file, and
                        calculate variable importance (default: 0)
  --saveComment SAVECOMMENT
                        comment to store in refit model (default: )
  --svmCacheMemFrac F   maximum percentage of total memory to use for svm
                        training. Crucial for svm training speed. (default:
                        0.5)
  --verbose N           verbose logging [0 - no logging; 1 - info, no
                        libraries; 2 - including libraries; 3 - debug
                        (default: 1)
```

## predict_mol_class.py

Apply a trained model to a smiles file. Input can be single-or-multi-column file
with or without header, or can be read from stdin in a unix pipe.

*Example call:*

```predict_mol_class.py -m models/model_dmso.pkl.xz data/example.smi```

```
usage: predict_mol_class.py [-h] -m MODELFILE [-H N] [-d DELIMITER] [-s N]
                            [-b N] [-D {0,1}]
                            [PATH [PATH ...]]

apply a trained model to a smiles file.

positional arguments:
  PATH                  input files to score, containing smiles

optional arguments:
  -h, --help            show this help message and exit
  -m MODELFILE, --modelFile MODELFILE
                        pkl file of trained model
  -H N, --header N      does the input file contain a header line?
  -d DELIMITER, --delimiter DELIMITER
                        delimiter in input file
  -s N, --smilesCol N   input column containing the smiles string
  -b N, --batchMode N   for speed, read whole input and score in batch. To
                        skip individual lines with invalid smiles, use 0.
  -D {0,1}, --describeModelAndExit {0,1}
                        only print model info without scoring
```
