#!/usr/bin/env python

import pandas as pd

# Check if matplot GUI works, it will not works if you run script from a server
import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except ImportError:
    mpl.use('agg')
from matplotlib.ticker import AutoMinorLocator

# Import regular expression, calling greek letter
import re

import numpy as np
from matplotlib.ticker import MultipleLocator
from optparse import OptionParser

def kpath_name_parse(KPATH_STR):
    '''
    Parse the kpath string
    '''

    KPATH_STR = KPATH_STR.upper()
    # legal kpath separators: blank space, comma, hypen or semicolon
    KPATH_SEPARATORS = ' ,-;'
    # Greek Letters Dictionaries
    GREEK_KPTS = {
        'G':      r'$\mathrm{\mathsf{\Gamma}}$',
        'GAMMA':  r'$\mathrm{\mathsf{\Gamma}}$',
        'DELTA':  r'$\mathrm{\mathsf{\Delta}}$',
        'LAMBDA': r'$\mathrm{\mathsf{\Lambda}}$',
        'SIGMA':  r'$\mathrm{\mathsf{\Sigma}}$',
    }

    # If any of the kpath separators is in the kpath string
    if any([s in KPATH_STR for s in KPATH_SEPARATORS]):
        kname = [
            GREEK_KPTS[x] if x in GREEK_KPTS else 
	    r'$\mathrm{{\mathsf{{{}}}}}$'.format(x)
            for x in re.sub('['+KPATH_SEPARATORS+']', ' ', KPATH_STR).split()
        ]
    else:
        kname = [
            GREEK_KPTS[x] if x in GREEK_KPTS else 
	    r'$\mathrm{{\mathsf{{{}}}}}$'.format(x)
            for x in KPATH_STR
        ]

    return kname

def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version=__version__)

    par.add_option('-f', '--file',
                    action='store', type="string",
                    dest='filename', default='band.xy,bndaxy_a1.xy',
                    help='Location of the band files, separated by a comma if multiple, no space in between.')

    
    par.add_option('-s', '--size', nargs=2,
                   action='store', type="float", dest='figsize',
                   default=(3.0, 4.0),
                   help='figure size of the output plot, ex: (3.0, 4.0)')
    
    par.add_option('-y', nargs=2,
                   action='store', type="float", dest='ylim',
                   default=None,
                   help='energy range of the band plot, ex: (-3,3)')
    
    par.add_option('--dpi',
                   action='store', type="int", dest='dpi',
                   default=360,
                   help='resolution of the output image, ex: 360')
    
    par.add_option('--lw',
                   action='store', type="float", dest='linewidth',
                   default=1.0,
                   help='linewidth of the band plot, ex: 1.0')
    
    par.add_option('-k', '--kpoints',
                   action='store', type="string", dest='kpts',
                   default=None,
                   help='kpoint path, use ",", " ", ";" or "-" to separate the K-point if necessary.')
    
    par.add_option('-o', '--output',
                   action='store', type="string", dest='bandimage',
                   default='band.png',
                   help='output image name, "band.png" by default')
    
    # New option for limiting components up to a specified orbital
    par.add_option('--components',
                   action='store', type="string", dest='components',
                   default='dx2y2',
                   help='Specify the highest component to include in the plot. Default is "dx2y2"')

    return par.parse_args()


__version__ = 1.0
opts, args = command_line_arg()

file_list = opts.filename.split(',')

file_path_band = file_list[0]
file_path_bndatm = file_list[1]

# band.xyデータをDataFrameにロード
data_band = pd.read_csv(file_path_band, sep=r'\s+', skiprows=1, 
                        names=['kx', 'ky', 'kz', 'dist', 'eig', 'iband', 'line'])

# lineの最大値を取得し、整数に変換
max_line_value = int(data_band['line'].max())

# lineの値が1からmax_line_valueまでの各値に対して、最初と最後に出現するdistの値を抽出
x_positions_for_labels = []
for line_value in range(1, max_line_value + 1):
    first_occurrence = data_band[data_band['line'] == line_value]['dist'].iloc[0]
    x_positions_for_labels.append(first_occurrence)
    if line_value == max_line_value:
        last_occurrence = data_band[data_band['line'] == line_value]['dist'].iloc[-1]
        x_positions_for_labels.append(last_occurrence)

# custom_labels の定義
custom_labels = ['Γ', 'X', 'U|K', 'R', 'Z', 'A']


column_names = ['dist', 'eig', 'iband', 's', 'pz', 'px', 'py', 'dz2r2', 'dxz', 'dyz', 'dxy', 'dx2y2',
                'fm0', 'f+m1', 'f-m1', 'f+m2', 'f-m2', 'f+m3', 'f-m3']
data_bndatm = pd.read_csv(file_path_bndatm, sep=r'\s+', header=None, names=column_names, comment='#')

# # Extract the necessary columns
dist = data_bndatm['dist']
eig = data_bndatm['eig']

# Default component order
component_order = ['s', 'pz', 'px', 'py', 'dz2r2', 'dxz', 'dyz', 'dxy', 'dx2y2',
                   'fm0', 'f+m1', 'f-m1', 'f+m2', 'f-m2', 'f+m3', 'f-m3']

# Determine the cutoff component
cutoff_index = component_order.index(opts.components) + 1
selected_components = component_order[:cutoff_index]

import random
import colorsys
# Generate dynamic, distinguishable light colors for each component
def generate_light_color():
    base_colors = []
    hue_step = 1.0 / len(selected_components)  # Distribute colors evenly across the hue spectrum

    for i in range(len(selected_components)):
        hue = i * hue_step  # Set a different hue for each component
        lightness = random.uniform(0.7, 0.85)  # Keep lightness high for a pastel color
        saturation = random.uniform(0.4, 0.6)  # Moderate saturation for distinguishability
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        base_colors.append(rgb)
    
    return {component: (base_colors[i], component) for i, component in enumerate(selected_components)}

components = generate_light_color()

# Convert to array to get the min and max later
x_positions_for_labels = np.array(x_positions_for_labels)

# Initiazlize graph
width, height = opts.figsize
if opts.ylim:
    ymin, ymax = opts.ylim
else:
    _ = opts.ylim

dpi = opts.dpi

fig = plt.figure()
fig.set_size_inches(width, height)
ax = plt.subplot(111)

# Scatter plot for each component based on their weights, with no fill
for component, (color, label) in components.items():
    plt.scatter(dist, eig, 
                s=data_bndatm[component] * 1, 
                edgecolors=color,  # Set edge color
                facecolors='none',  # No fill color
                label=label, 
                alpha=0.6)


opts.kpts = "G, X, U|K, R, Z, A" # K-path for diamond

for pos in x_positions_for_labels:
    ax.axvline(x=pos, 
               color ='k', 
               linestyle='--', 
               linewidth=0.5,
               alpha=0.5)

ax.set_ylabel('Energy [eV]',
              labelpad=5)

ax.set_xticks(x_positions_for_labels)

if opts.kpts:
    ax.set_xticklabels(kpath_name_parse(opts.kpts))
else:
    ax.set_xticklabels([])

if opts.ylim:
    ax.set_ylim(ymin, ymax)

# Plot the min and max of X axis
ax.set_xlim(x_positions_for_labels.min(), x_positions_for_labels.max())

# Update legend to use edge-only markers
handles, labels = plt.gca().get_legend_handles_labels()
legend_markers = [plt.Line2D([0], [0], marker='o', color=h.get_edgecolor()[0],  # Use edge color only
                              markerfacecolor='none', markersize=10, 
                              markeredgewidth=1.5) for h in handles]  # Adjust edge width for clarity
plt.legend(legend_markers, labels, loc='upper right')

plt.grid(True)

ax.set_xticks(x_positions_for_labels)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tight_layout(pad=0.20)
plt.savefig(opts.bandimage, dpi=opts.dpi)
plt.show()
