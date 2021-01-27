#!python3
import os
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from plot_module import *


def print_b(text):
    print('\033[34m' + '\033[1m' + text + '\033[0m')


def grec_letter(s):
    if s == "beta" or s == "chi" or s == "omega":
        return '\\' + s
    elif s == "gamma":
        return "\\Delta \\Delta G"
    elif s == "alpha":
        return "\\Delta G_{\\mathrm{min}}"
    elif s == "pop_size":
        return "N_{\\mathrm{e}}"
    elif s == "gamma_std":
        return "\\sigma ( \\gamma )"
    elif s == "gamma_distribution_shape":
        return "k"
    elif s == "exon_size":
        return "n"
    elif s == "expression_level":
        return "y"
    elif s == "sub-ΔG-mean":
        return "\\Delta G"
    else:
        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    parser.add_argument('-f', '--node', required=False, default=False, type=bool, dest="node")
    args = parser.parse_args()
    array_values = dict()
    li = []
    for filepath in args.input:
        if not os.path.isfile(filepath): continue
        x, y = float(filepath.split("_")[-3]), float(filepath.split("_")[-2])
        li.append(pd.read_csv(filepath.replace(".tsv", ".parameters.tsv"), sep='\t').assign(x=x, y=y))
        for param, vals in pd.read_csv(filepath, sep='\t').items():
            if (args.node and "Node" not in param) or (("dnd" not in param) and ("sub-ΔG-mean" not in param)): continue
            if param not in array_values: array_values[param] = dict()
            array_values[param][(x, y)] = vals

    df_p = pd.concat(li, axis=0, ignore_index=True)
    uniq = df_p.apply(pd.Series.nunique)
    df = df_p.drop(uniq[uniq == 1].index, axis=1)

    x_uniq = df[df["x"] == df["x"].values[0]].apply(pd.Series.nunique)
    df_x = df.drop(x_uniq[x_uniq != 1].index, axis=1).drop_duplicates()
    col_x = [c for c in df_x if c != "x"][0]
    x_axis = sorted(df_x["x"].values)
    x_range = np.array([df_x[df_x["x"] == x][col_x].values[0] for x in x_axis])

    csv_output = []
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for param, x_y_z in array_values.items():
        if len(x_axis) < 2: continue
        name = param.replace("/", "").replace("|", "")
        y_axis = sorted(set([k[1] for k in x_y_z.keys()]))
        for (j, y) in enumerate(y_axis):
            label_dict = dict()
            if "chi" in df_p and "dnd" in param:
                label_dict["chi"] = df_p[df_p["y"] == y]["chi"].values[0]
            if len(y_axis) > 1:
                df_y = df[df["y"] == y]
                y_uniq = df_y.apply(pd.Series.nunique)
                df_y = df_y.drop(y_uniq[y_uniq != 1].index, axis=1).drop_duplicates()
                for col in df_y:
                    if col == "y" or col == "chi": continue
                    label_dict[col] = df_y[col].values[0]
            mean_z = [np.mean(x_y_z[(x, y)]) for x in x_axis]
            label = ("$" + ", \\ ".join(
                ["{0}={1:.3g}".format(grec_letter(k), v) for k, v in label_dict.items()]) + "$") if len(
                label_dict) > 0 else None
            base_line, = plt.plot(x_range, mean_z, linewidth=2, label=label)
            plt.fill_between(x_range, [np.percentile(x_y_z[(x, y)], 5) for x in x_axis],
                             [np.percentile(x_y_z[(x, y)], 95) for x in x_axis], alpha=0.3)
            if ('SimuStab' not in args.output) and ('SimuFold' not in args.output): continue
            results = sm.OLS(mean_z, sm.add_constant(np.log(x_range))).fit()
            b, a = results.params[0:2]
            idf = np.logspace(np.log(min(x_range)), np.log(max(x_range)), 30, base=np.exp(1))
            linear = a * np.log(idf) + b
            reg = '$\\hat{\\chi}' + '={0:.4g}\\ (r^2={1:.3g})$'.format(float(a), results.rsquared)
            print(reg)
            csv_output.append({"name": name, "mean": True, "a": a, "b": b, "r2:": results.rsquared,
                               "label": label})
            plt.plot(idf, linear, '-', linewidth=4, color=base_line.get_color(), linestyle="--", label=reg)

            if len(y_axis) > 1:
                continue
            for i, x in enumerate(x_axis):
                plt.scatter([x_range[i]] * len(x_y_z[(x, y)]), x_y_z[(x, y)], color=base_line.get_color(), alpha=0.05)
        plt.xscale("log")
        plt.xlabel("$" + grec_letter(col_x) + "$", fontsize=label_size)
        if "dnd" in param: param = 'omega'
        plt.ylabel("$" + grec_letter(param) + "$", fontsize=label_size)
        # plt.ylim((0.3, 0.4))
        plt.legend(fontsize=legend_size)
        if len([c for c in df_x]) > 2:
            plt.title("Scaling also $" + ", ".join(
                [grec_letter(c) for c in df_x if (c != "x" and c != col_x)]) + "$ on the x-axis.")
        plt.tight_layout()
        plt.savefig("{0}/{1}.pdf".format(args.output, name), format="pdf", dpi=my_dpi)
        plt.savefig("{0}/{1}.png".format(args.output, name), format="png", dpi=my_dpi)
        plt.clf()
    plt.close('all')
    pd.DataFrame(csv_output).to_csv(args.output + "/results.tsv", index=False, sep="\t")
