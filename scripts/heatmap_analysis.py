#!python3
import os
import argparse
import numpy as np
import pandas as pd
import itertools
import matplotlib

matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    parser.add_argument('-x', '--x_param', required=True, type=str, dest="x_param")
    parser.add_argument('-y', '--y_param', required=True, type=str, dest="y_param")
    args = parser.parse_args()
    files = dict()
    for filepath in args.input:
        if not os.path.isfile(filepath):
            continue
        for param, vals in pd.read_csv(filepath, sep='\t').items():
            if param not in files:
                files[param] = dict()
            files[param][(float(filepath.split("_")[-3]), float(filepath.split("_")[-2]))] = np.mean(vals)
    
    for param, x_y_z in files.items():
        x_axis = sorted(set([k[0] for k in x_y_z.keys()]))
        y_axis = sorted(set([k[1] for k in x_y_z.keys()]))
        y_array, x_array = np.meshgrid(y_axis, x_axis)
        z_array = np.zeros((len(x_axis), len(y_axis)))
        for (i, x), (j, y) in itertools.product(enumerate(x_axis), enumerate(y_axis)):
            z_array[i, j] = x_y_z[(x, y)]
        my_dpi = 256
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        c = plt.pcolor(x_array, y_array, z_array)
        plt.title(param)
        plt.colorbar(c)
        plt.xlabel(args.x_param)
        plt.xscale("log")
        plt.ylabel(args.y_param)
        plt.yscale("log")
        plt.tight_layout()
        name = param.replace("/", "").replace("|", "")
        plt.savefig("{0}/{1}.svg".format(args.output, name), format="svg")
        plt.clf()
        plt.close('all')
