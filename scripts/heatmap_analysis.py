#!python3
import os
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib

label_size = 24
my_dpi = 128
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_b(text):
    print('\033[34m' + '\033[1m' + text + '\033[0m')


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
    parser.add_argument("--y_param_key", required=False, action='append', type=lambda kv: kv.split(":"),
                        dest='y_param_dict')
    parser.add_argument('-f', '--node', required=False, default=False, type=bool, dest="node")
    args = parser.parse_args()
    args.y_param_dict = dict([(float(k), v) for k, v in args.y_param_dict]) if (
                args.y_param_dict is not None) else dict()
    array_values = dict()
    for filepath in args.input:
        if not os.path.isfile(filepath): continue
        for param, vals in pd.read_csv(filepath, sep='\t').items():
            if args.node and "Node" not in param: continue
            if param not in array_values:
                array_values[param] = dict()
            x, y = float(filepath.split("_")[-3]), float(filepath.split("_")[-2])
            array_values[param][(x, y)] = vals

    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for param, x_y_z in array_values.items():
        if "dnd" not in param: continue
        name = param.replace("/", "").replace("|", "")
        x_axis = sorted(set([k[0] for k in x_y_z.keys()]))
        y_axis = sorted(set([k[1] for k in x_y_z.keys()]))
        if len(x_axis) < 2: continue

        for log in [False, True]:
            for (j, y) in enumerate(y_axis):
                label = "{0}={1:.3g}".format(args.y_param, y) if args.y_param != "" else None
                if float(y) in args.y_param_dict:
                    label = args.y_param_dict[float(y)]
                mean_z = [np.mean(x_y_z[(x, y)]) for x in x_axis]
                base_line, = plt.plot(x_axis, mean_z, linewidth=3, label=label)
                plt.fill_between(x_axis, [np.percentile(x_y_z[(x, y)], 5) for x in x_axis],
                                 [np.percentile(x_y_z[(x, y)], 95) for x in x_axis], alpha=0.3)
                for x in x_axis:
                    plt.scatter([x] * len(x_y_z[(x, y)]), x_y_z[(x, y)], color=base_line.get_color(), alpha=0.25)

                if log: mean_z = np.log(mean_z)
                model = sm.OLS(mean_z, sm.add_constant(np.log(x_axis)))
                results = model.fit()
                if ("SimuStab" not in args.output) and ("SimuFold" not in args.output): continue
                b, a = results.params[0:2]
                idf = np.logspace(np.log(min(x_axis)), np.log(max(x_axis)), 30, base=np.exp(1))
                linear = a * np.log(idf) + b
                if log: linear = np.exp(linear)
                label = r"$y={0}x {3} {1}$ ($r^2={2})$".format(tex_float(float(a)), tex_float(abs(float(b))),
                                                               tex_float(results.rsquared),
                                                               "+" if float(b) > 0 else "-")
                if "dnd" in name:
                    print_b("d({0}{1}{2})/d(ln(Ne)) : {3}".format("ln(" if log else "", name,
                                                                  ")" if log else "", label))
                    print_b("{0} = {1}".format(name, tex_float(mean_z[int(len(x_axis) / 2)])))
                plt.plot(idf, linear, '-', linewidth=2, color=base_line.get_color(), linestyle="--", label=label)
            if log: plt.yscale("log")
            plt.xlabel(args.x_param, fontsize=label_size)
            plt.xscale("log")
            if "dnd" in param: param = '$\\omega$'
            plt.ylabel(param, fontsize=label_size)
            plt.legend(loc='upper right', fontsize=15)
            plt.tight_layout()
            plt.savefig("{0}/{1}{2}.pdf".format(args.output, "Log_" if log else "", name), format="pdf", dpi=my_dpi)
            plt.clf()
            if not log and "dnd" not in name: break
    plt.close('all')
