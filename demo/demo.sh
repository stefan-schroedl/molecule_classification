#!/bin/bash
# Subset of data for DMSO solubility are from https://ochem.eu:
# See: Tetko, I. V., Novotarskyi, S., Sushko, I., Ivanov, V., Petrenko, A. E., Dieden, R., … Mathieu, B. (2013). Development of dimethyl sulfoxide solubility models using 163 000 molecules: Using a domain applicability metric to select more reliable predictions. Journal of Chemical Information and Modeling, 53(8), 1990–2000. https://doi.org/10.1021/ci400213d

# use both molecule descriptors and ecfp4 fingerprints; train a random forest model with 10 iterations of hyperparameter optimization
../train_mol_class.py --config dmso.cfg --outDir ./experiments --fpType prop,ecfp4 --methods rf_optim --data data_train_dmso_solubility.tsv.gz

# use optimized model for prediction on new dataset
../predict_mol_class.py --header 1 --smilesCol 2 -m experiments/data_train_dmso_solubility_ecfp4~prop_rf/model_data_train_dmso_solubility_ecfp4~prop_rf.pkl.xz data_test_dmso_solubility.tsv > data_test_dmso_solubility_with_pred.tsv
