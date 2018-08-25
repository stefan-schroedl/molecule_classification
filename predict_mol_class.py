#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse
import pprint
import fileinput

from sklearn.externals import joblib

from train_mol_class import Fingerprinter, ThresholdTuner, LabelStratificationEncoder

parser = argparse.ArgumentParser(description='apply a trained model to a smiles file.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--modelFile', type=str, required=True, help='pkl file of trained model')
parser.add_argument('-H', '--header', metavar='N', type=int, choices=[0, 1], default=0, help='does the input file contain a header line?')
parser.add_argument('-d', '--delimiter', type=str, default='\t', help='delimiter in input file')
parser.add_argument('-s', '--smilesCol', metavar='N', type=int, default=1, help='input column number containing the smiles string (1-based)')
parser.add_argument('-b', '--batchMode', metavar='N', type=int, default=1, choices=[0, 1], help='for speed, read whole input and score in batch. To skip individual lines with invalid smiles, use 0.')
parser.add_argument('smilesFile', metavar='PATH', default='-', nargs = '*', help='input files to score, containing smiles')
parser.add_argument('-D', '--describeModelAndExit', type=int, choices=[0, 1], default=0, help='only print model info without scoring')

args = parser.parse_args()

pipe = joblib.load(args.modelFile)


if args.describeModelAndExit:
    if hasattr(pipe, 'created'):
        print('MODEL PREDICTING %s, CREATED: %s' % (pipe.target_name, pipe.created), file=sys.stderr)
    if hasattr(pipe, 'comment'):
        print(pipe.comment, file=sys.stderr)
    print('steps:', file=sys.stderr)
    for step in pipe.steps:
        pprint.pprint(step, stream=sys.stderr, depth=1000)
    sys.exit(0)


lines = []
smiles = []
cnt = 0

for line in fileinput.input(args.smilesFile):
    line = line.rstrip()
    cnt += 1
    if cnt == 1:
        if args.header > 0:
            print(args.delimiter.join([line, pipe.target_name]))
            continue
    fields = line.split(args.delimiter)
    if args.smilesCol > len(fields):
        raise ValueError('specified input column %d, but file only contains %d columns' %(args.smilesCol, len(fields)))
    if args.batchMode:
        lines.append(line)
        smiles.append(fields[args.smilesCol-1])
    else:
        try:
            score = pipe.predict([fields[args.smilesCol-1]])
            print(args.delimiter.join([line, str(score[0])]))
        except KeyboardInterrupt:
            sys.exit(1)
        except:
            print('error in line %d: %s' % (cnt, line), file=sys.stderr)

if args.batchMode:
    scores = pipe.predict(smiles)
    if len(scores) != len(lines):
        raise ValueError('could not predict %d cases' % (len(lines) - len(scores)))
    lines = [args.delimiter.join([line, str(score)]) for line, score in zip(lines, scores)]
    print('\n'.join(lines))
