"""
Calculate the peak mass function (PMF) for subhalos and derive their hierarchical origin.

This script demonstrates how to compute the PMF for subhalos at different hierarchical
levels and analyze their initial merger ratio and accretion redshift distributions.
It uses a double-Schechter function for the level-1 PMF and a convolutional kernel
for higher levels.

Dependencies:
- numpy
- matplotlib
- colossus (for cosmology)
- fourier_c (custom module, assumed to provide `db`, `level_o`, `get_correct_parameter`)
- HiSub (custom module, assumed to provide hierarchical subhalo calculations)
- MAH_Zhao (custom module, assumed to provide mass assembly history)

License: MIT License (see LICENSE file for details)
Author: [Wenkang Jiang]
"""

import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology


try:
    from fourier_c import db, level_o, get_correct_parameter
    from HiSub import HiSub
    from MAH import MAH_Zhao
except ImportError:
    raise ImportError(
        "Custom modules 'fourier_c', 'HiSub', and 'MAH' are required. "
        "Please provide these modules or check documentation for alternatives."
    )

# Set global plot styling
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'legend.fontsize': 14,
    'figure.figsize': (8, 8),
    'axes.grid': False,
})

# Constants
MIN_MASS_RATIO = 1e-4
MAX_MASS_RATIO = 1
NUM_MASS_POINTS = 100
MIN_MERGER_RATIO = 1e-3
MAX_MERGER_RATIO = 1.585  # 10^0.2
NUM_MERGER_POINTS = 40
MIN_REDSHIFT = 0
MAX_REDSHIFT = 3
NUM_REDSHIFT_POINTS = 50
HOST_HALO_MASS = 1e3  # in 10^10 Msun h^-1
SUBHALO_MASS_RATIO = 1e-3  # mu_0
MASS_UNIT = "10^10 Msun h^-1"

# Configuration
PMF_PARAMS = {
    'level_1_params': (0.029, 0.273, 1-0.94, 1-0.54, 12.89, 2.26),  # Double-Schechter params fitting for the level 1 PMF in the form of μdN/dlnμ
    'beta': 0.726,  # Convolutional kernel parameter
    'colors': ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'cyan', 'yellow', 'pink'],
}


def plot_peak_mass_function(mass_ratios, max_depth=4):
    """
    Plot the peak mass function for subhalos at different hierarchical levels.

    Input
    ----------
    mass_ratios : ndarray
        Array of mass ratios (mu) for the x-axis.
    max_depth : int, optional
        Maximum hierarchical level to compute (default: 4).

    Output
    -------
    None
    """
    if not np.all(mass_ratios > 0):
        raise ValueError("Mass ratios must be positive for logarithmic scale.")

    fig, ax = plt.subplots()
    level_1_pmf = lambda x: db(x, *PMF_PARAMS['level_1_params'])
    kernel_pmf = lambda x: db(x, *get_correct_parameter(PMF_PARAMS['level_1_params'], PMF_PARAMS['beta']))

    # Plot PMF for each level
    for depth in range(1, max_depth + 1):
        pmf_values = level_o(mass_ratios, depth, level_1_pmf, kernel_pmf)
        ax.plot(
            mass_ratios, pmf_values, color=PMF_PARAMS['colors'][depth - 1],
            linestyle='--', linewidth=3, alpha=0.6, label=f'Level {depth}'
        )

    # Plot PMF for all levels
    all_pmf_values = level_o(mass_ratios, 0, level_1_pmf, kernel_pmf)
    ax.plot(
        mass_ratios, all_pmf_values, color='black',
        linestyle='-', linewidth=3, alpha=0.4, label='All Levels'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(MIN_MASS_RATIO, MAX_MASS_RATIO)
    ax.set_ylim(1e-3, 7e-1)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\mu \, \mathrm{d}N / \mathrm{d}\ln\mu$')
    ax.legend(loc='upper left', frameon=False)
    plt.show()
    plt.savefig('./output/Subhalo_PMFs.jpg',dpi=720,bbox_inches='tight')

def plot_initial_merger_ratio(merger_ratios, subhalo_mass_ratio, min_depth=2, max_depth=4):
    """
    Plot the initial merger ratio distribution for subhalos at different levels.

    Input
    ----------
    merger_ratios : ndarray
        Array of merger ratios (xi) for the x-axis.
    subhalo_mass_ratio : float
        Final peak mass ratio of the subhalo (mu_0).
    min_depth : int, optional
        Minimum hierarchical level to compute (default: 2).
    max_depth : int, optional
        Maximum hierarchical level to compute (default: 4).

    Output
    -------
    None
    """
    if not np.all(merger_ratios > 0):
        raise ValueError("Merger ratios must be positive for logarithmic scale.")

    fig, ax = plt.subplots()
    hier = HiSub(PMF_PARAMS['level_1_params'], PMF_PARAMS['beta'])

    for depth in range(min_depth, max_depth + 1):
        distribution = hier.initial_merger_ratio(merger_ratios, mu_0=subhalo_mass_ratio, depth=depth)
        ax.loglog(
            merger_ratios, merger_ratios * distribution,
            color=PMF_PARAMS['colors'][depth - 1], linewidth=3, label=f'Level {depth}'
        )

    ax.set_xlim(1e-4, 10)
    ax.set_ylim(1e-3, 1e1)
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$p_{1|\ell}(\ln\xi|\mu)$')
    ax.text(0.7, 0.9, f'$\\mu_0 = {subhalo_mass_ratio:.0e}$', transform=ax.transAxes)
    ax.legend(loc='upper left', frameon=False)
    plt.show()
    plt.savefig('./output/Initial_merger_ratio.jpg',dpi=720,bbox_inches='tight')

def plot_accretion_redshift(redshifts, subhalo_mass_ratio, host_halo_mass, max_depth=4):
    """
    Plot the accretion redshift distribution for subhalos at different levels.

    Input
    ----------
    redshifts : ndarray
        Array of redshifts (z) for the x-axis.
    subhalo_mass_ratio : float
        Peak mass ratio of the subhalo (mu_0).
    host_halo_mass : float
        Mass of the host halo in 10^10 Msun h^-1.
    max_depth : int, optional
        Maximum hierarchical level to compute (default: 4).

    Output
    -------
    None
    """
    fig, ax = plt.subplots()
    hier = HiSub(PMF_PARAMS['level_1_params'], PMF_PARAMS['beta'])
    mah_halo = MAH_Zhao()

    for depth in range(1, max_depth + 1):
        distribution = hier.accretion_redshift(
            mu_0=subhalo_mass_ratio, z_array=redshifts, depth=depth,
            MAH=mah_halo, M_halo=host_halo_mass
        )
        ax.plot(
            redshifts, distribution, color=PMF_PARAMS['colors'][depth - 1],
            linewidth=3, alpha=0.5, linestyle='-', label=f'Level {depth}'
        )

    ax.set_xlim(MIN_REDSHIFT, MAX_REDSHIFT)
    ax.set_ylim(0, 2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel('z')
    ax.set_ylabel(r'$\mathrm{d}P / \mathrm{d}z$')
    ax.text(0.4, 0.9, f'$M_h = 10^{{{int(np.log10(host_halo_mass*1e10))}}}\\, \\mathrm{{M_\\odot}}\,h^{{-1}}$', transform=ax.transAxes)
    ax.text(0.4, 0.8, f'$\\mu_0 = {subhalo_mass_ratio:.0e}$', transform=ax.transAxes)
    ax.legend(loc='upper left', frameon=False)
    plt.show()
    plt.savefig('./output/Accretion_redshifts_distribution.jpg',dpi=720,bbox_inches='tight')
    
def main():
    """Main function to run the subhalo analysis and generate plots."""
    # Generate input arrays
    mass_ratios = np.logspace(np.log10(MIN_MASS_RATIO), np.log10(MAX_MASS_RATIO), NUM_MASS_POINTS)
    merger_ratios = np.logspace(np.log10(MIN_MERGER_RATIO), np.log10(MAX_MERGER_RATIO), NUM_MERGER_POINTS)
    redshifts = np.linspace(MIN_REDSHIFT, MAX_REDSHIFT, NUM_REDSHIFT_POINTS)

    # Generate plots
    plot_peak_mass_function(mass_ratios)
    plot_initial_merger_ratio(merger_ratios, SUBHALO_MASS_RATIO)
    plot_accretion_redshift(redshifts, SUBHALO_MASS_RATIO, HOST_HALO_MASS)


if __name__ == '__main__':
    main()