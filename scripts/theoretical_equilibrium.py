#!python3
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib

label_size = 24
my_dpi = 96
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-g', '--exon_size', required=True, type=int, dest="n")
    parser.add_argument('-p', '--population_size', required=True, type=float, dest="population_size")
    parser.add_argument('-a', '--alpha', required=True, type=float, dest="alpha")
    parser.add_argument('-k', '--kappa', required=True, type=float, dest="kappa")
    parser.add_argument('-b', '--beta', required=True, type=float, dest="beta")
    args, unknown = parser.parse_known_args()
    dict_df = dict()


    def delta_g(x, alpha):
        return alpha + args.kappa * x


    def sel_coeff(x, alpha):
        edg = np.exp(args.beta * delta_g(x, alpha))
        return args.kappa * args.beta * edg / ((1 + edg) * args.n)


    def scaled_sel_coeff(x, alpha):
        return 4 * args.population_size * sel_coeff(x, alpha)


    def mut_bias(x):
        if x == 0.:
            return float("inf")
        elif x == 1.0:
            return -float("inf")
        return np.log((1 - x) / x)


    def self_consistent_eq(x, alpha):
        return mut_bias(x) - scaled_sel_coeff(x, alpha)


    x_eq = brentq(lambda x: self_consistent_eq(x, args.alpha), 0.0, 1.0, full_output=True)[0]
    assert (x_eq <= 0.5)
    s = sel_coeff(x_eq, args.alpha)
    S = 4 * args.population_size * s
    assert ((S - mut_bias(x_eq)) < 1e-5)
    x_min, x_max = 0, 0.5
    y_min, y_max = 0, S * 2
    x_range = np.linspace(x_min, x_max, 200)
    label = "$\\alpha={0:.2f}, \\kappa={1:.2f}, n={2}, Ne={3:.2f}$"
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.plot(x_range, [mut_bias(i) for i in x_range], linewidth=3, label="$ln[(1-x)/x]$")
    line, = plt.plot(x_range, [scaled_sel_coeff(i, args.alpha) for i in x_range], linewidth=3,
                     label="S: " + label.format(args.alpha, args.kappa, args.n, args.population_size))
    plt.plot(x_range, [10 * scaled_sel_coeff(i, args.alpha) for i in x_range],
             linestyle="--", color=line.get_color(), linewidth=3,
             label="S: " + label.format(args.alpha, args.kappa, args.n, 10 * args.population_size))
    dict_df["x"] = [x_eq]
    dict_df["ΔG"] = [delta_g(x_eq, args.alpha)]
    dict_df["s"] = [s]
    dict_df["S"] = [S]
    dict_df["dNdS"] = [x_eq * S / (1 - np.exp(-S)) + (1 - x_eq) * -S / (1 - np.exp(S))]
    args.kappa *= 0.1
    args.alpha = brentq(lambda a: s - sel_coeff(x_eq, a), 10 * args.alpha, 0.1 * args.alpha, full_output=True)[0]
    line, = plt.plot(x_range, [scaled_sel_coeff(i, args.alpha) for i in x_range], linewidth=3,
                     label="S: " + label.format(args.alpha, args.kappa, args.n, args.population_size))
    plt.plot(x_range, [10 * scaled_sel_coeff(i, args.alpha) for i in x_range],
             linestyle="--", color=line.get_color(), linewidth=3,
             label="S: " + label.format(args.alpha, args.kappa, args.n, 10 * args.population_size))
    plt.legend(fontsize=label_size)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.tight_layout()
    plt.savefig("{0}.pdf".format(args.output), format="pdf", dpi=my_dpi)
    pd.DataFrame(dict_df).to_csv(args.output, index=False, sep="\t")
