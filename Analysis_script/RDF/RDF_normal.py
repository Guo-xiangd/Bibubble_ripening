#!/usr/bin/env python3
# coding: utf-8

import pickle
import numpy as np
np.set_printoptions(linewidth=100)
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF, InterRDF_s

# Color palette configuration
colors = [(31 ,59 ,115), 
          (47 ,146,148), 
          (80 ,178,141),
#          (167,214,85 ),
          (255,224,62 ),
          (255,169,85 ),
          (189, 48,47 )]
colors = list(tuple(i/255 for i in color) for color in colors)
colors.reverse()

# Set bigger font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 17
plt.rc('font', size=SMALL_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)   # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-1)  # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)   # fontsize of the figure title

#################### User Configuration Below ####################
geo_path    = "/home/xiangdang/dpmd/bubble_ion/nacl/data" # Update this path
trj_path    = "../MD"                                      # Update this path
save_path   = "./"
data_geo    = "D30L55N2_20NaOH_atomic.data"
trj_name    = "bubble.lammpstrj"
trj_skip    = 1000
mda_step    = 1
type_map    = {'H': 1, 'O': 2, 'N': 3, 'Na': 4, 'Cl': 5}
# NOOdict     = {'N': 163, 'O1': 169, 'O2': 171} # Serial IDs in lammps .data file
#################### User Configuration Above ####################

data_geo = os.path.join(geo_path, data_geo)
trj_file = os.path.join(trj_path, trj_name)

# Initialize MDAnalysis Universe (dt is in ps)
u = mda.Universe(data_geo, trj_file, atom_style='id type x y z', format='LAMMPSDUMP', dt=1.0e-3)

def get_rdf(c1, c2, u, type_map, trj_skip, mda_step, save_path):
    """Calculate RDF between two atom types."""
    s1 = u.select_atoms('type %s' % type_map[c1]) # e.g., H
    s2 = u.select_atoms('type %s' % type_map[c2]) # e.g., O
    half_box = u.dimensions[0] / 2
    rdf = InterRDF(s1, s2, nbins=1000, range=(0.01, 10))
    rdf.run(start=trj_skip, step=mda_step) 
    y_rdf = rdf.rdf
    x_rdf = rdf.bins
    
    # Save RDF data to file
    filename = os.path.join(save_path, "{0}_{1}_nep022_v2-9k.txt".format(c1, c2))
    with open(filename, 'w+') as file:
        file.write("#RDF %4s-%4s\n" % (c1, c2))
        file.write('#%8s%8s\n' % ('bin', 'rdf'))
        for i in range(len(x_rdf)):
            file.write(' %10.4f%10.4f\n' % (x_rdf[i], y_rdf[i]))
    
    return x_rdf, y_rdf

def get_fig(c1, c2, x_rdf, y_rdf, colors, save_path):
    """Plot RDF."""
    fig = plt.figure(figsize=(8, 3), dpi=150, facecolor='white')
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(x_rdf, y_rdf, alpha=0.8, lw=3, color=colors[0], label='%2s –%2s' % (c1, c2))
    ax.legend(loc="upper right")
    ax.set_xlim(0, 7)
    ax.set_xlabel("r " + r"$\ \rm (\AA)$")
    ax.set_ylabel("g(r)")
    
    # Note: fig.show() might not work in non-interactive environments
    # fig.show() 
    fig.savefig(os.path.join(save_path, "{0}_{1}_nep022_v2-9k.png".format(c1, c2)), dpi=600)

# Execution examples (Uncomment as needed)
# x_rdf, y_rdf = get_rdf('Na', 'Cl', u, type_map, trj_skip, mda_step, save_path)
# get_fig('Na', 'Cl', x_rdf, y_rdf, colors, save_path)

x_rdf, y_rdf = get_rdf('Na', 'O', u, type_map, trj_skip, mda_step, save_path)
get_fig('Na', 'O', x_rdf, y_rdf, colors, save_path)

# x_rdf, y_rdf = get_rdf('Cl', 'O', u, type_map, trj_skip, mda_step, save_path)
# get_fig('Cl', 'O', x_rdf, y_rdf, colors, save_path)

x_rdf, y_rdf = get_rdf('Na', 'N', u, type_map, trj_skip, mda_step, save_path)
get_fig('Na', 'N', x_rdf, y_rdf, colors, save_path)

# x_rdf, y_rdf = get_rdf('Cl', 'N', u, type_map, trj_skip, mda_step, save_path)
# get_fig('Cl', 'N', x_rdf, y_rdf, colors, save_path)

x_rdf, y_rdf = get_rdf('Na', 'H', u, type_map, trj_skip, mda_step, save_path)
get_fig('Na', 'H', x_rdf, y_rdf, colors, save_path)

# x_rdf, y_rdf = get_rdf('Cl', 'H', u, type_map, trj_skip, mda_step, save_path)
# get_fig('Cl', 'H', x_rdf, y_rdf, colors, save_path)

x_rdf, y_rdf = get_rdf('H', 'H', u, type_map, trj_skip, mda_step, save_path)
get_fig('H', 'H', x_rdf, y_rdf, colors, save_path)

x_rdf, y_rdf = get_rdf('H', 'O', u, type_map, trj_skip, mda_step, save_path)
get_fig('H', 'O', x_rdf, y_rdf, colors, save_path)

x_rdf, y_rdf = get_rdf('O', 'O', u, type_map, trj_skip, mda_step, save_path)
get_fig('O', 'O', x_rdf, y_rdf, colors, save_path)