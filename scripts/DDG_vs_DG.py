#!python3
import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
import statsmodels.api as sm
import matplotlib

label_size = 24
my_dpi = 128
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=False, default="Heatmap/Experiments/figure-3C", type=str,
                        dest="input")
    args = parser.parse_args()
    x_range, y_range = [], []
    for filepath in sorted(glob("../{0}/merge/*_seed.tsv".format(args.input))):
        df = pd.read_csv(filepath, sep='\t')
        x_range.append(df["sub-ΔG-mean"].values)
        y_range.append(df["mut-ΔΔG-mean"].values)
    x = np.array(x_range).flatten()
    y = np.array(y_range).flatten()
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.scatter(x, y, linewidth=2, alpha=0.25, color="#5D80B4")
    results = sm.OLS(y, sm.add_constant(x)).fit()
    b, a = results.params[0:2]
    idf = np.linspace(min(x), max(x), 30)
    linear = a * idf + b
    reg = 'ΔΔG={0:.2g}ΔG {1} {2:.2g} ($r^2$={3:.2g})'.format(a, "+" if b > 0 else "-", abs(b), results.rsquared)
    plt.plot(idf, linear, '-', linewidth=4, linestyle="--", label=reg, color="#E29D26")
    plt.xlabel("ΔG", fontsize=label_size)
    plt.ylabel("ΔΔG", fontsize=label_size)
    # plt.ylim((0.3, 0.4))
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig("../{0}/DG-DDG.pdf".format(args.input), format="pdf", dpi=my_dpi)
    plt.clf()
    plt.close('all')
