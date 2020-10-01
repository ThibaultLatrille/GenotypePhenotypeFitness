#!python3
import matplotlib

label_size = 24
legend_size = 20
my_dpi = 128
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams["font.family"] = ["Latin Modern Mono"]
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler

plt.rc('axes', prop_cycle=cycler(color=["#5D80B4", "#E29D26", "#8FB03E", "#EB6231", "#857BA1"]))
