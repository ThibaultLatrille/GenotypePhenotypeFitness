#!python3
import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from plot_module import *


def replace_grec(s):
    return s.replace("\\alpha", "\\Delta G_{\\mathrm{min}}").replace("\\gamma", "\\Delta \\Delta G")


nbr_points = 100
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--age', required=True, type=float, dest="age")
    parser.add_argument('-b', '--branches', required=True, type=int, dest="branches")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument("--y_param_key", required=False, action='append', type=lambda kv: kv.split(":"),
                        dest='y_param_dict')
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    args = parser.parse_args()
    args.y_param_dict = dict(args.y_param_dict) if (args.y_param_dict is not None) else dict()
    fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    t_range = np.linspace(0, args.branches * args.age, nbr_points * args.branches)
    y_param = "<dN/dN0>"
    for prefix in args.input:
        print(prefix)
        array = []
        for tsv_path in sorted(glob("{0}/*.substitutions.tsv".format(prefix))):
            df = pd.read_csv(tsv_path, sep='\t', usecols=["NodeName", "AbsoluteStartTime", "EndTime", y_param])
            t_sample = np.searchsorted(df["AbsoluteStartTime"].values, t_range, side="right") - 1
            array.append(df[y_param].values[t_sample])

        n = prefix.split("/")[-1]
        label = args.y_param_dict[n] if (n in args.y_param_dict) else "n={0}".format(n)
        plt.plot(t_range, np.mean(array, axis=0), linewidth=3, label=replace_grec(label))
        plt.fill_between(t_range, np.percentile(array, 5, axis=0), np.percentile(array, 95, axis=0), alpha=0.2)

    plt.xlabel(r'$t$', fontsize=label_size)
    plt.ylabel("$\\omega$", fontsize=label_size)
    for s in range(1, args.branches):
        plt.axvline(x=s * args.age, linewidth=3, color='black')
    plt.xlim((0, args.branches * args.age))
    plt.legend(fontsize=legend_size)
    plt.tight_layout()
    plt.savefig(args.output, format=args.output[-3:])
    plt.clf()
    plt.close('all')
