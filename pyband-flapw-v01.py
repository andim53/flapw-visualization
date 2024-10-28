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
                   dest='filename', default='band.xy',
                   help='location of band.xy')
    
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
    return par.parse_args()


__version__ = 1.0
opts, args = command_line_arg()

# データファイルのパス
file_path_band = 'band.xy'
file_path_bndatm = 'bndaxy_a1.xy'

# band.xyデータをDataFrameにロード
data_band = pd.read_csv(file_path_band, delim_whitespace=True, skiprows=1, 
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


# Extract the necessary columns
dist = data_bndatm['dist']
eig = data_bndatm['eig']

# List of components to plot
components = {
    's': ('blue', 's'),
    'pz': ('green', 'pz'),
    'px': ('cyan', 'px'),
    'py': ('magenta', 'py'),
    'dz2r2': ('red', 'dz2r2'),
    'dxz': ('orange', 'dxz'),
    'dyz': ('purple', 'dyz'),
    'dxy': ('brown', 'dxy'),
    'dx2y2': ('olive', 'dx2y2'),
    #'fm0': ('lightblue', 'fm0'),
    #f+m1': ('pink', 'f+m1'),
    #'f-m1': ('lightgreen', 'f-m1'),
    #'f+m2': ('lightcoral', 'f+m2'),
    #'f-m2': ('yellow', 'f-m2'),
    #'f+m3': ('lightgray', 'f+m3'),
    #'f-m3': ('darkblue', 'f-m3'),
}

# Create the plot
plt.figure(figsize=(12, 8))

# Scatter plot for each component based on their weights
for component, (color, label) in components.items():
    plt.scatter(dist, eig, s=data_bndatm[component] * 1, c=color, label=label, alpha=0.5)

plt.ylabel('Energy [eV]', fontsize=14)
plt.ylim(-10, 10)

# x位置での垂直線を描画
for pos in x_positions_for_labels:
    plt.axvline(x=pos, color='gray', linestyle='--', linewidth=0.1)

# カスタムx軸の目盛りとラベルを設定
plt.xticks(x_positions_for_labels, custom_labels)

# x軸の範囲をデータが存在する範囲に設定
plt.xlim(data_bndatm['dist'].min(), data_bndatm['dist'].max())

# 凡例の作成
handles, labels = plt.gca().get_legend_handles_labels()
legend_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=h.get_facecolor()[0], markersize=10) for h in handles]
plt.legend(legend_markers, labels, loc='upper right')

plt.grid(True)
plt.show()