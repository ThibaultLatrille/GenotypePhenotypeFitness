#!python3
import os
import argparse
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
    dict_df = {filepath: {k: v[0] for k, v in pd.read_csv(filepath + '.tsv', sep='\t').items()} for filepath in
               args.input if os.path.isfile(filepath + '.tsv')}

    cols = ["<dN>/<dN0>", "<dN/dN0>"]
    for col in cols:
        my_dpi = 128
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        max_y = 0.0
        for filepath in dict_df.keys():
            assert (os.path.isfile(filepath + ".substitutions.tsv"))
            df = pd.read_csv(filepath + ".substitutions.tsv", sep='\t',
                             usecols=["NodeName", "AbsoluteStartTime", "EndTime"] + cols)
            max_y = max((max(df[col]), max_y))
            base_line, = plt.plot(df["AbsoluteStartTime"], df[col], linewidth=3)

            end_value = 1.0
            for node_name in set(df["NodeName"]):
                df_filt = df[df['NodeName'] == node_name]
                plt.axvline(x=df_filt.iloc[0]["AbsoluteStartTime"], linewidth=3, color='black')
                start_value = df_filt[col].values[0]
                if node_name != "0":
                    dict_df[filepath]["Node{0}_incr{1}".format(node_name, col)] = df_filt[col].values[-1] / end_value
                end_value = df_filt[col].values[-1]
                half_value = (start_value + end_value) / 2
                if half_value < start_value:
                    half_life_row = df_filt[df_filt[col] < half_value].iloc[0]
                elif half_value > start_value:
                    half_life_row = df_filt[df_filt[col] > half_value].iloc[0]
                else:
                    half_life_row = df_filt.iloc[0]
                dict_df[filepath]["Node{0}_HalfLife{1}".format(node_name, col)] = half_life_row["EndTime"]
                plt.axvline(x=half_life_row["AbsoluteStartTime"], linewidth=2, color=base_line.get_color())
        plt.ylim((0, max_y))
        plt.xlabel(r'$t$')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig("{0}.HalfLife{1}.svg".format(args.output, col.replace("/", '-')), format="svg")
        plt.clf()
        plt.close('all')

    pd.DataFrame(dict_df.values()).to_csv(args.output, index=False, sep="\t")
