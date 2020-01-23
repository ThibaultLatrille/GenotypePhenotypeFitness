#!python3
import os
import argparse
import pandas as pd
import matplotlib

matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    args = parser.parse_args()
    print(args.input, args.output)

    for param in ["<dN>/<dN0>", "<dN/dN0>"]:
        my_dpi = 128
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        max_y = 0.0
        for filepath in args.input:
            if not os.path.isfile(filepath + ".substitutions.tsv"):
                continue

            df = pd.read_csv(filepath + ".substitutions.tsv", sep='\t')
            max_y = max((max(df[param]), max_y))
            plt.plot(df["AbsoluteStartTime"], df[param], linewidth=3)

        plt.ylim((0, max_y))
        plt.xlabel(r'$t$')
        plt.ylabel(param)
        plt.tight_layout()
        plt.savefig("{0}/{1}.svg".format(args.output, param.replace("/", "")), format="svg")
        plt.clf()
        plt.close('all')
