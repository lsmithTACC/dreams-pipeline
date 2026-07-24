import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.stats import binned_statistic_2d
#arrs go True, then ML
def make_ML_comp(arrs,box,scalar = "Density"):
    if type(scalar) != str:
        raise TypeError("Scalar Input must be a string!")
    #load data
    ml_snap = arrs[-1]
    if arrs!= False:
        ml_snap = arrs[0]
    ml_x = ml_snap[:,0] - ml_snap[:,0].mean()
    ml_y = ml_snap[:,1] - ml_snap[:,1].mean()
    ml_z = ml_snap[:,2] - ml_snap[:,2].mean()
    ml_density = (ml_snap[:,4])
    ml_velo = np.sqrt(ml_snap[:,-4]**2 + ml_snap[:,-3]**2 + ml_snap[:,-2]**2)
    ml_potential = ml_snap[:,3]
    if scalar == "Density":
        ml_carr = ml_density
        cmap_label = "Log_10(Density)"
    if scalar == "Velocity":
        ml_carr = ml_velo
        cmap_label  = scalar
    if scalar == "Potential":
        ml_carr = ml_potential
        cmap_label = scalar
    elif scalar not in ["Velocity","Density","Potential"]:
        raise KeyError(scalar)
    tru_snap = arrs[1]
    tru_x = tru_snap[:,0] - tru_snap[:,0].mean()
    tru_y = tru_snap[:,1] - tru_snap[:,1].mean()
    tru_z = tru_snap[:,2] - tru_snap[:,2].mean()
    tru_density = (tru_snap[:,4])
    tru_velo = np.sqrt(tru_snap[:,-4]**2 + tru_snap[:,-3]**2 + tru_snap[:,-2]**2)
    tru_potential = tru_snap[:,3]
    if scalar == "Density":
        tru_carr = tru_density
    if scalar == "Velocity":
        tru_carr = tru_velo
    if scalar == "Potential":
        tru_carr = tru_potential
        cmap_label = scalar

    fig = plt.figure(figsize=(26, 12))
    subfigs = fig.subfigures(nrows=2 , ncols=1, hspace=0.1)
    #Gas first:    
    tru_coord_label = {"x_tru":tru_x , "y_tru":tru_y , "z_tru":tru_z}
    tru_coords = list(tru_coord_label.keys())
    tru_perms = list(combinations(tru_coords,r=2))
    top_ax = subfigs[0].subplots(nrows = 1,ncols = 3)
    for a,p in enumerate(tru_perms):
        c1 = p[0]
        c2 = p[1]
        stat, xedges, yedges, binnumber = binned_statistic_2d(
        tru_coord_label[c1], tru_coord_label[c2], tru_carr,
        statistic='mean',
        bins= 500,
        range=None,
        expand_binnumbers=False)
        tru_mesh = top_ax[a].pcolormesh(xedges, yedges, stat.T,cmap = "viridis")
        top_ax[a].set_xlabel(f'{c1.split("_")[0]}_coordinate [kpc]')
        top_ax[a].set_ylabel(f'{c2.split("_")[0]}_coordinate [kpc]')
        top_ax[a].set_title(f'{c1.split("_")[0]}{c2.split("_")[0]} {scalar} Projection')
    subfigs[0].colorbar(tru_mesh,ax = top_ax, label = cmap_label , shrink = 0.8)
    pos_left = top_ax[0].get_position()
    pos_right = top_ax[-1].get_position()
    center_x = (pos_left.x0 + pos_right.x1) / 2
    subfigs[0].suptitle(f"True {scalar} DM Distributions", fontsize=20, fontweight='bold',y=1.02,x=center_x)

    #Dark Matter Now
    print(subfigs.shape)
    ml_coord_label = {"x_ml":ml_x , "y_ml":ml_y , "z_ml":ml_z}
    ml_coords = ["x_ml","y_ml","z_ml"]
    ml_perms = list(combinations(ml_coords,r=2))
    bottom_ax = subfigs[1].subplots(nrows = 1,ncols = 3)
    for a,p in enumerate(ml_perms):
        c1 = p[0]
        c2 = p[1]
        stat, xedges, yedges, binnumber = binned_statistic_2d(
        ml_coord_label[c1], ml_coord_label[c2], ml_carr,
        statistic='mean',
        bins= 500,
        range=None,
        expand_binnumbers=False)
        mesh = bottom_ax[a].pcolormesh(xedges, yedges, stat.T,cmap="viridis")
        bottom_ax[a].set_xlabel(f'{c1.split("_")[0]}_coordinate [kpc]')
        bottom_ax[a].set_ylabel(f'{c2.split("_")[0]}_coordinate [kpc]')
        bottom_ax[a].set_title(f'{c1.split("_")[0]}{c2.split("_")[0]} Density Projection')
    subfigs[1].colorbar(mesh,ax = bottom_ax, label = cmap_label , shrink = 0.8)
    pos_left = bottom_ax[0].get_position()
    pos_right = bottom_ax[-1].get_position()
    center_x = (pos_left.x0 + pos_right.x1) / 2
    subfigs[1].suptitle(f"ML Evaluated Dark Matter {scalar} Distributions", fontsize=20, fontweight='bold',y=1.02,x=center_x)
    plt.savefig(f"box_{box}_comparison.png",bbox_inches='tight')
