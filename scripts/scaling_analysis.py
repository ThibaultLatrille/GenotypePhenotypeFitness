#!python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', dest="input")
    args = parser.parse_args()
    print(args.input, args.output)
    files = dict()
    for filepath in args.input:
        if not os.path.isfile(filepath):
            continue
        for param, vals in pd.read_csv(filepath, sep='\t').items():
            if param not in files:
                files[param] = list()
            files[param].append({'x': float(filepath.split("_")[-2]), 'y': vals})

    for param, list_x_y in files.items():
        sorted_x_y = sorted(list_x_y, key=lambda i: i['x'])
        x = [i['x'] for i in sorted_x_y]
        y_array = [i['y'] for i in sorted_x_y]
        my_dpi = 256
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        plt.plot(x, [np.mean(a) for a in y_array], linewidth=3)
        plt.fill_between(x, [np.nanpercentile(a, 10) for a in y_array], [np.nanpercentile(a, 90) for a in y_array], alpha=0.3)
        plt.xlabel(r'$\mathrm{N_{e}}$')
        plt.xscale("log")
        plt.ylabel(param)
        minimum = min([min(a) for a in y_array])
        if "dnd" in param:
            plt.ylim((0, 1))
        if minimum > 0 and ("dnd" not in param):
            plt.yscale("log")
        plt.tight_layout()
        name = param.replace("<", "").replace(">", "").replace("|", "-")
        plt.savefig("{0}/{1}.svg".format(args.output, name), format="svg")
        plt.clf()
        plt.close('all')
