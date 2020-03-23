#!python3
import os
import argparse
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib

label_size = 16
my_dpi = 128
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tex_float(x):
    s = "{0:.3g}".format(x)
    if "e" in s:
        mantissa, exp = s.split('e')
        return mantissa + '\\times 10^{' + exp + '}'
    else:
        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    parser.add_argument('-x', '--x_param', required=True, type=str, dest="x_param")
    parser.add_argument('-y', '--y_param', required=False, default="", type=str, dest="y_param")
    parser.add_argument("--y_param_key", action='append', type=lambda kv: kv.split(":"), dest='y_param_dict')
    parser.add_argument('-f', '--node', required=False, default=False, type=bool, dest="node")
    args = parser.parse_args()
    args.y_param_dict = dict(args.y_param_dict) if (args.y_param_dict is not None) else dict()
    array_values = dict()
    len_values = set()
    for filepath in args.input:
        if not os.path.isfile(filepath):
            continue
        for param, vals in pd.read_csv(filepath, sep='\t').items():
            if args.node and "Node" not in param:
                continue
            if param not in array_values:
                array_values[param] = dict()
            x, y = float(filepath.split("_")[-3]), float(filepath.split("_")[-2])
            array_values[param][(x, y)] = vals
            len_values.add(len(vals))

    assert (len(len_values) == 1)
    nbr_simu = len_values.pop()
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for param, x_y_z in array_values.items():
        name = param.replace("/", "").replace("|", "")
        x_axis = sorted(set([k[0] for k in x_y_z.keys()]))
        if len(x_axis) < 2:
            continue

        y_axis = sorted(set([k[1] for k in x_y_z.keys()]))
        y_array, x_array = np.meshgrid(y_axis, x_axis)
        z_mean_array = np.zeros((len(x_axis), len(y_axis), nbr_simu))
        for (i, x), (j, y) in itertools.product(enumerate(x_axis), enumerate(y_axis)):
            z_mean_array[i, j] = x_y_z[(x, y)]

        for (j, y) in enumerate(y_axis):
            label = "{0}={1:.3g}".format(args.y_param, y) if args.y_param != "" else None
            if str(y) in args.y_param_dict:
                label = args.y_param_dict[str(y)]
            base_line, = plt.plot(x_axis, np.mean(z_mean_array[:, j, :], axis=1), linewidth=3,  label=label)
            plt.fill_between(x_axis, np.percentile(z_mean_array[:, j, :], 5, axis=1),
                             np.percentile(z_mean_array[:, j, :], 95, axis=1), alpha=0.3)
            for k in range(nbr_simu):
                plt.scatter(x_axis, z_mean_array[:, j, k], color=base_line.get_color(), alpha=0.25)

            model = sm.OLS(np.mean(z_mean_array[:, j, :], axis=1), sm.add_constant(np.log(x_axis)))
            results = model.fit()
            if results.rsquared < 0.9:
                continue
            b, a = results.params[0:2]
            idf = np.logspace(np.log(min(x_axis)), np.log(max(x_axis)), 30, base=np.exp(1))
            plt.plot(idf, a * np.log(idf) + b, '-', linewidth=2, color=base_line.get_color(), linestyle="--",
                     label=r"$y={0}x {3} {1}$ ($r^2={2})$".format(tex_float(float(a)), tex_float(abs(float(b))),
                                                                  tex_float(results.rsquared),
                                                                  "+" if float(b) > 0 else "-"))
        plt.xlabel(args.x_param, fontsize=label_size)
        plt.xscale("log")
        plt.ylabel(param, fontsize=label_size)
        plt.legend(loc='upper right', fontsize=label_size)
        plt.tight_layout()
        plt.savefig("{0}/plot1d.{1}.pdf".format(args.output, name), format="pdf", dpi=my_dpi)
        plt.clf()

        if len(y_axis) < 5:
            continue

        c = plt.pcolor(x_array, y_array, np.mean(z_mean_array, axis=2))
        plt.title(param, fontsize=label_size * 2)
        plt.colorbar(c)
        plt.xlabel(args.x_param, fontsize=label_size)
        plt.xscale("log")
        plt.ylabel(args.y_param, fontsize=label_size)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("{0}/plot2d.{1}.svg".format(args.output, name), format="svg", dpi=my_dpi)
        plt.clf()

    plt.close('all')
