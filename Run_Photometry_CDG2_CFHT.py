#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'viridis'

import astropy.units as u
from astropy.io import fits

from toolbox import parse_arguments, make_apertures, make_plot_CFHT

from Run_Photometry_CDG2 import (coords_cdg2, coords_gc,
                                 run_photometry_analysis,
                                 calculate_cdg2_luminosity)

# Constants
pixel_scale = 0.178 # arcsec/pix
zp = 30 # zero-point
distance_Perseus = 75 * u.Mpc
conversion_mag = -0.24 # This is the magnitude correction from the used band to V band. Here V=g-0.24 is used for CFHT g.

fix_gc_centers = True # If True, fix GC centers in PSF fitting
iso_fit_kws = {'min_sma':0.5, 'max_sma':22, 'step':0.5} # Parameters for isophote fitting
                            
n_iter = 20 # number of iterations
cutoff_SB = 4.2 # cutoff radius in arcsec

def main(file_name, file_name_psf, output_dir=None):
    """
    Perform photometric analysis on the CDG-2 galaxy image.
    
    Parameters
    ----------
    file_name : str
        Path to the input FITS image file.
    file_name_psf : str
        Path to the input FITS PSF image file.
    output_dir : str, optional
        Directory to save output files. If None, uses current directory.

    """

    # Prepare output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Setup apertures
    aper_cdg2, aper_gc = make_apertures(coords_cdg2, coords_gc, file_name, pixel_scale, aperture_cdg2=4, aperture_gc=0.5)
    
    # Run photometry analysis
    table, data, models, isolist = run_photometry_analysis(
        file_name, file_name_psf, aper_gc, aper_cdg2,
        r_inner=4,
        r_outer=5,
        aperture_radius=3,
        min_separation=10,
        fix_gc_centers=fix_gc_centers,
        smooth_stddev=2,
        n_iter=n_iter,
        nan_pad=False,
        cutoff_SB=cutoff_SB,
        isophote_fitting_kws=iso_fit_kws,
        output_dir=output_dir
    )
    
    make_plot_CFHT(data, models,
                  pixel_scale, zp,
                  aper_cdg2, aper_gc,
                  cutoff_SB=cutoff_SB,
                  output_dir=output_dir)
    
    # Averaged total flux of GC and total flux of diffuse emission
    flux_gc_med = np.median(table['flux_gc'][1:])
    flux_cdg2_med = np.median(table['flux_cdg2'][1:])
    
    # Calculate luminosity
    calculate_cdg2_luminosity(flux_cdg2_med, flux_gc_med, distance_Perseus, zp, factor=0.24)
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args.file_name, args.file_name_psf, args.output_dir)
