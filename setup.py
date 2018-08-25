import os
from setuptools import setup

files = ["data/*", "dmso.cfg" ]
script_files = ['train_mol_class.py', 'predict_mol_class.py']

# HACK
# note: rdkit is an install dependency, but it doesn't work to add it to the install_dependencies below
# because it is found in the non-standard 'rdkit' channel which setup doesn't seem to check.

try:
    import rdkit
except:
    raise ValueError('missing install dependency: rdkit')

setup(name="mol_class",
      version="0.1",
      description="Molecule classification model",
      scripts=script_files,
      package_data={'mol_class': files},
      install_requires=['psutil', 'configargparse', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib'],
      zip_safe=False
      )
