#!python3
import os
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
import matplotlib

label_size = 16
my_dpi = 128
nbr_points = 1000
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exponential(_t, _r, _a, _b):
    return _a * np.exp(-_r * (_t - t_min)) + _b


def tex_float(x):
    s = "{0:.3g}".format(x)
    if "e" in s:
        mantissa, exp = s.split('e')
        return mantissa + '\\times 10^{' + exp + '}'
    else:
        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--fitting', required=False, default=False, type=bool, dest="fitting")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    args = parser.parse_args()
    dict_df = {filepath: {k: v[0] for k, v in pd.read_csv(filepath + '.tsv', sep='\t').items()} for filepath in
               args.input if os.path.isfile(filepath + '.tsv')}
    fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    col = "<dN/dN0>"
    dico_files, dico_node = dict(), dict()
    for filepath in dict_df.keys():
        prefix = filepath.split("/")[-2]
        if prefix not in dico_files:
            dico_files[prefix] = list()
        dico_files[prefix].append(filepath)
    for prefix, list_files in dico_files.items():
        for filepath in dict_df.keys():
            if not os.path.isfile(filepath + ".substitutions.tsv"):
                continue
            df = pd.read_csv(filepath + ".substitutions.tsv", sep='\t',
                             usecols=["NodeName", "AbsoluteStartTime", "EndTime", col])
            base_line, = plt.plot(df["AbsoluteStartTime"], df[col], linewidth=1, alpha=0.3)
            for node_name in set(df["NodeName"]):
                if int(node_name) == 1 or not args.fitting:
                    continue
                df_filt = df[df['NodeName'] == node_name]
                if node_name not in dico_node:
                    dico_node[node_name] = []
                t = df_filt["AbsoluteStartTime"].values
                x = df_filt[col].values
                dico_node[node_name].append((t, x))
                x_half = (x[0] + x[-1]) / 2
                if x_half < x[0]:
                    t_half = df_filt[x < x_half].iloc[0]
                elif x_half > x[0]:
                    t_half = df_filt[x > x_half].iloc[0]
                else:
                    t_half = df_filt.iloc[0]
                dict_df[filepath]["Node{0}_HalfLife{1}".format(node_name, col)] = t_half["EndTime"]
                dict_df[filepath]["Node{0}_HalfLife{1}".format(node_name, col)] = np.log(2) / t_half["EndTime"]

        if len(dico_node) == 0:
            plt.clf()
            continue
        param_dico = dict()
        for node_name, list_values in dico_node.items():
            t_min = min([v[0][0] for v in list_values])
            t_max = max([v[0][-1] for v in list_values])
            t_range = np.linspace(t_min, t_max, nbr_points)
            x_range = np.mean([v[1][np.searchsorted(v[0], t_range, side="right") - 1] for v in list_values], axis=0)
            dico_node[node_name] = (t_range, x_range)
            plt.axvline(x=t_min, linewidth=3, color='black')
            l, = plt.plot(t_range, x_range, linewidth=3)
            if not args.fitting:
                continue
            p_0 = [np.log(2) * 2.0 / (t_max - t_min), x_range[0] - x_range[-1], x_range[-1]]

            [rate, gap, final], pcov = curve_fit(exponential, t_range, x_range, p0=p_0, maxfev=50000,
                                                 bounds=([0., -np.inf, 0.], [np.inf, np.inf, np.inf]))
            param_dico[node_name] = [rate, gap, final]
            label = "{1:.2g}exp(-{0:.2g} * (x - {3:.2g})) + {2:.2g}".format(rate, gap, final, t_min)
            plt.plot(t_range, [exponential(x, rate, gap, final) for x in t_range], label=label,
                     linestyle='--', color=l.get_color(), linewidth=3)

        plt.xlabel(r'$t$', fontsize=label_size)
        plt.ylabel(col, fontsize=label_size)
        if args.fitting:
            plt.legend()
        plt.tight_layout()
        file_format = "png" if (len(dict_df) >= 50) else "pdf"
        plt.savefig("{0}.{1}.{2}".format(args.output, col.replace("/", '-'), file_format), format=file_format)
        plt.clf()

        for node_name, (t, x) in dico_node.items():
            plt.plot(t[:-1], (x[1:] - x[:-1]) / (t[1:] - t[:-1]), linewidth=3)
            plt.axvline(x=t[0], linewidth=3, color='black')
        plt.xlabel(r'$t$', fontsize=label_size)
        plt.ylabel('d(' + col + ')/dt', fontsize=label_size)
        plt.tight_layout()
        plt.savefig("{0}.{1}.Differentiate.pdf".format(args.output, col.replace("/", '-')), format="pdf")
        plt.clf()

        if not args.fitting:
            continue

        for node_name, (t, x) in dico_node.items():
            [rate, gap, final] = param_dico[node_name]
            if node_name == 1:
                continue
            plt.subplot(int((len(dico_node) - 1) / 2), 2, int(node_name) - 1)
            delta_x = x[:-1] - final
            dxdt = (x[1:] - x[:-1]) / (t[1:] - t[:-1])
            plt.scatter(delta_x, dxdt, linewidth=3)
            model = sm.OLS(dxdt, delta_x)
            results = model.fit()
            a = results.params[0]
            idf = np.linspace(min(delta_x), max(delta_x), 30)
            plt.plot(idf, a * idf, '-', linewidth=2, linestyle="--",
                     label=r"$y={0}x$ ($r^2={1})$".format(tex_float(float(a)), tex_float(results.rsquared)))
            plt.xlabel(col + "-" + col + '_final', fontsize=label_size)
            plt.ylabel('d(' + col + ')/dt', fontsize=label_size)
            plt.ylim((min(dxdt), max(dxdt)))
            plt.legend()
            plt.title(node_name)
        plt.tight_layout()
        plt.savefig("{0}.{1}.Regression.pdf".format(args.output, col.replace("/", '-')), format="pdf")
        plt.clf()
        plt.close('all')

    pd.DataFrame(dict_df.values()).to_csv(args.output, index=False, sep="\t")
