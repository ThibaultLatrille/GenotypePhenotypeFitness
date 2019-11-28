#!python3
import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    args = parser.parse_args()
    rows = [{k: v[0] for k, v in pd.read_csv(filepath + '.tsv', sep='\t').items()} for filepath in args.input if
            os.path.isfile(filepath + '.tsv')]
    pd.DataFrame(rows).to_csv(args.output, index=False, header=rows[0].keys(), sep="\t")
