#!python3
import os
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib

label_size = 16
my_dpi = 96
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--fitting', required=False, default=False, type=bool, dest="fitting")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    args = parser.parse_args()
    dict_df = {filepath: {k: v[0] for k, v in pd.read_csv(filepath + '.tsv', sep='\t').items()} for filepath in
               args.input if os.path.isfile(filepath + '.tsv')}

    cols = ["<dN/dN0>"]
    fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for col in cols:
        max_y = 0.0
        for filepath in dict_df.keys():
            if not os.path.isfile(filepath + ".substitutions.tsv"):
                continue
            df = pd.read_csv(filepath + ".substitutions.tsv", sep='\t',
                             usecols=["NodeName", "AbsoluteStartTime", "EndTime"] + cols)
            max_y = max((max(df[col]), max_y))
            base_line, = plt.plot(df["AbsoluteStartTime"], df[col], linestyle='--', linewidth=1, alpha=0.3)

            last_final = 1.0
            for node_name in set(df["NodeName"]):
                df_filt = df[df['NodeName'] == node_name]
                t = df_filt["AbsoluteStartTime"].values
                plt.axvline(x=t[0], linewidth=3, color='black')
                if not args.fitting:
                    continue

                def exponential(x, r, a, b):
                    return a * np.exp(-r * (x - t[0])) + b

                y = df_filt[col].values

                p_0 = [np.log(2) * 2.0 / (t[-1] - t[0]), y[0] - y[-1], y[-1]]
                [rate, gap, final], pcov = curve_fit(exponential, t, df_filt[col], p0=p_0, maxfev=50000,
                                                     bounds=([0., -np.inf, 0.], [np.inf, np.inf, np.inf]))

                plt.plot(t, [exponential(x, rate, gap, final) for x in t], linewidth=3, color=base_line.get_color())

                dict_df[filepath]["Node{0}_HalfLife{1}".format(node_name, col)] = np.log(2) / rate
                dict_df[filepath]["Node{0}_Gap{1}".format(node_name, col)] = gap
                dict_df[filepath]["Node{0}_Final{1}".format(node_name, col)] = final
                dict_df[filepath]["Node{0}_FinalIncrement{1}".format(node_name, col)] = final / last_final
                last_final = final

        if max_y == 0.0 or np.isnan(max_y):
            plt.clf()
            continue
        plt.ylim((0, max_y))
        plt.xlabel(r'$t$', fontsize=label_size)
        plt.ylabel(col, fontsize=label_size)
        plt.tight_layout()
        plt.savefig("{0}.ExponentialDecay.{1}.pdf".format(args.output, col.replace("/", '-')), format="pdf")
        plt.clf()
    plt.close('all')

    pd.DataFrame(dict_df.values()).to_csv(args.output, index=False, sep="\t")
