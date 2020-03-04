#!python3
import os
import argparse
import numpy as np
import pandas as pd
import itertools
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
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    parser.add_argument('-x', '--x_param', required=True, type=str, dest="x_param")
    parser.add_argument('-y', '--y_param', required=True, type=str, dest="y_param")
    args = parser.parse_args()
    mean_values, std_values = dict(), dict()
    for filepath in args.input:
        if not os.path.isfile(filepath):
            continue
        for param, vals in pd.read_csv(filepath, sep='\t').items():
            if param not in mean_values:
                mean_values[param] = dict()
                std_values[param] = dict()
            x, y = float(filepath.split("_")[-3]), float(filepath.split("_")[-2])
            mean_values[param][(x, y)] = np.mean(vals)
            std_values[param][(x, y)] = np.std(vals)

    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    for param, x_y_z in mean_values.items():
        name = param.replace("/", "").replace("|", "")
        x_axis = sorted(set([k[0] for k in x_y_z.keys()]))
        if len(x_axis) < 2:
            continue

        y_axis = sorted(set([k[1] for k in x_y_z.keys()]))
        y_array, x_array = np.meshgrid(y_axis, x_axis)
        z_mean_array = np.zeros((len(x_axis), len(y_axis)))
        z_std_array = np.zeros((len(x_axis), len(y_axis)))
        for (i, x), (j, y) in itertools.product(enumerate(x_axis), enumerate(y_axis)):
            z_mean_array[i, j] = x_y_z[(x, y)]
            z_std_array[i, j] = 1.96 * std_values[param][(x, y)]

        for (j, y) in enumerate(y_axis):
            if args.y_param != "":
                plt.plot(x_axis, z_mean_array[:, j], label="{0}={1:.3g}".format(args.y_param, y), linewidth=3)
            else:
                plt.plot(x_axis, z_mean_array[:, j], linewidth=3)
            plt.fill_between(x_axis, z_mean_array[:, j] - z_std_array[:, j], z_mean_array[:, j] + z_std_array[:, j],
                             alpha=0.3)
        plt.xlabel(args.x_param, fontsize=label_size)
        plt.xscale("log")
        plt.ylabel(param, fontsize=label_size)
        plt.legend(loc='upper right', fontsize=label_size)
        plt.tight_layout()
        plt.savefig("{0}/plot1d.{1}.svg".format(args.output, name), format="svg", dpi=my_dpi)
        plt.clf()

        if len(y_axis) < 5:
            continue

        c = plt.pcolor(x_array, y_array, z_mean_array)
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
