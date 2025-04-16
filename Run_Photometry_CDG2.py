#!/usr/bin/env python3
import os
from tqdm import tqdm
import numpy as np
import warnings
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.visualization import ImageNormalize, LogStretch, AsinhStretch

from photutils.segmentation import detect_sources, deblend_sources
from photutils.background import LocalBackground, MMMBackground, Background2D
from photutils.psf import EPSFModel, SourceGrouper, PSFPhotometry
from photutils.isophote import EllipseGeometry, Ellipse, IsophoteList, build_ellipse_model

from toolbox import *

# CDG-2 coordinates in hourangle, deg
coords_cdg2 = ("3h17m12.61s", "41d20m51.5s") 

# GC coordinates in hourangle, deg
coords_gc = [("3h17m12.50s", "41d20m51s"),
             ("3h17m12.66s", "41d20m50.58s"),
             ("3h17m12.62s", "41d20m52.79s"),
             ("3h17m12.63s", "41d20m53.68s")]

# Constants
pixel_scale = 0.3 # arcsec/pix
zp = 30.132 # zero-point
distance_Perseus = 75 * u.Mpc

fix_gc_centers = False # If True, fix GC centers
iso_fit_kws = {'min_sma':0.5, 'max_sma':17, 'step':0.25} # Parameters for isophote fitting

n_iter = 10 # number of iterations

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
    aper_cdg2, aper_gc = make_apertures(coords_cdg2, coords_gc, file_name, pixel_scale, aperture_cdg2=5, aperture_gc=0.6)
    
    # Run photometry analysis
    table, data, models, isolist = run_photometry_analysis(
        file_name, file_name_psf, aper_gc, aper_cdg2,
        fix_gc_centers=fix_gc_centers,
        isophote_fitting_kws=iso_fit_kws,
        n_iter=n_iter, output_dir=output_dir
    )
    
    # Plot the model and residual
    make_plot_Euclid(data, models, aper_cdg2, aper_gc, output_dir=output_dir)
    
    # Averaged total flux of GC and total flux of diffuse emission
    flux_gc_med = np.median(table['flux_gc'][1:])
    flux_cdg2_med = np.median(table['flux_cdg2'][1:])
    
    # Calculate luminosity
    calculate_cdg2_luminosity(flux_cdg2_med, flux_gc_med, distance_Perseus, zp, factor=0.5)
    
    return models


def do_PSF_photometry(data, psf_model,
                      positions_guess, fit_shape, 
                      r_inner=4, r_outer=6, 
                      min_separation=5,
                      aperture_radius=3,
                      error=None, verbose=True):

    """
    Perform PSF photometry to measured flux and positions of sources.
    It needs an input of a PSF model and initial guess of positions.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D image data to perform photometry on.
    psf_model : photutils.psf.EPSFModel
        PSF model for the image.
    positions_guess : numpy.ndarray
        Initial guess for source positions in pixel coordinates.
    fit_shape : tuple
        Shape of the fitting region.
    r_inner : float, optional
        Inner radius for local background estimation in pixel. Default is 4.
    r_outer : float, optional
        Outer radius for local background estimation in pixel. Default is 6.
    min_separation : float, optional
        Minimum separation between sources. Default is 5.
    aperture_radius : float, optional
        Radius for aperture photometry in pixel. Default is 3.
    error : numpy.ndarray, optional
        Error array for the image. If None, uses image standard deviation.
    verbose : bool, optional
        Whether to print photometry results. Default is True.
    
    Returns
    -------
    phot : astropy.table.QTable
        Table of photometric measurements.
    gc_model : numpy.ndarray
        Model of the globular clusters.
    residual : numpy.ndarray
        Residual image after PSF model subtraction.
    """
    
    # Setup for PSF photometry
    localbkg_estimator = LocalBackground(r_inner, r_outer, bkg_estimator=MMMBackground())
    grouper = SourceGrouper(min_separation=min_separation)

    # Set class for PSF photometry
    psfphot = PSFPhotometry(psf_model, 
                            grouper=grouper,
                            fit_shape=fit_shape,
                            localbkg_estimator=localbkg_estimator,
                            aperture_radius=aperture_radius)
    
    # Initial parameters
    init_params = QTable()
    init_params['x'] = positions_guess[:,0]
    init_params['y'] = positions_guess[:,1]
    
    if error is None:
        # Using stddev as estimate of uncertainty
        error = np.std(data) * np.ones_like(data)
    
    # run photometry
    phot = psfphot(data, error=error, init_params=init_params)
    if verbose:
        print("PSF photometry output: ")
        print(phot[('id', 'x_fit', 'y_fit', 'flux_fit', 'flux_err', 'group_id', 'flags')])  
    
    # Compute residual
    residual = psfphot.make_residual_image(data, psf_shape=fit_shape)
    gc_model = data - residual

    return phot, gc_model, residual


def do_Isophote_Fitting(data, position, 
                        min_sma=0.5, max_sma=17., step=0.25, 
                        eps_init=0.01, pa_init=np.pi/2, 
                        nclip=5, fflag=0.25):
    
    """
    Perform elliptical isophote fitting on a galaxy image to model the light distribution.
    Parameters follow those in the photutils.isophote module.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D image data.
    position : tuple
        Central position (x, y) for isophote fitting.
        
    min_sma : float, optional
        Minimum semi-major axis length. Default is 0.5.
    max_sma : float, optional
        Maximum semi-major axis length. Default is 17.0.
    step : float, optional
        Step size for semi-major axis iteration. Default is 0.25.
    eps_init : float, optional
        Initial ellipticity. Default is 0.01.
    pa_init : float, optional
        Initial position angle. Default is π/2.
    nclip : int, optional
        Number of sigma clipping iterations. Default is 5.
    fflag : float, optional
        Fraction of pixels to ignore during fitting. Default is 0.25.
    
    Returns
    -------
    isolist : photutils.isophote.IsophoteList
        List of fitted isophotes.
    iso_model : numpy.ndarray
        Model of the galaxy's light distribution.
    residual_iso : numpy.ndarray
        Residual image after isophote model subtraction.
    """
    
    # Initial ellipse geometry
    geometry = EllipseGeometry(x0=position[0], y0=position[1], sma=(min_sma+max_sma)/2, eps=eps_init, pa=pa_init)
    
    # Setup ellipse
    ellipse = Ellipse(data, geometry)
    
    # Do isophote fitting over a range of semi-major axis length 
    isophote_list = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for sma in np.arange(min_sma, max_sma+step, step):
            iso = ellipse.fit_isophote(sma=sma, maxrit=max_sma, nclip=nclip, fflag=fflag, isophote_list=isophote_list)

            # If the centroid of ellipse moves too far away, or is too ellipitical,
            # run the isophote fitting with previous geometry (with noniterate=True)
            dist_centeroid = np.sqrt((iso.x0-position[0])**2+(iso.y0-position[1])**2)
            if (dist_centeroid > sma) or (iso.eps>0.5):
                isophote_list.pop(-1)
                iso = ellipse.fit_isophote(sma=sma, nclip=nclip, fflag=fflag, noniterate=True, isophote_list=isophote_list)
       
    # Build isophote model
    isolist = IsophoteList(isophote_list)
    iso_model = build_ellipse_model(data.shape, isolist)
    
    # Compute residual
    residual_iso = data - iso_model
    
    return isolist, iso_model, residual_iso
    
def run_photometry_analysis(file_name,
                            file_name_psf,
                            aper_gc,
                            aper_cdg2,
                            r_inner=4,
                            r_outer=6,
                            aperture_radius=3,
                            min_separation=5,
                            fix_gc_centers=False,
                            back_size=32,
                            sn_thre=2,
                            smooth_stddev=1,
                            n_iter=3,
                            nan_pad=True,
                            cutoff_SB=5,
                            isophote_fitting_kws=iso_fit_kws,
                            output_dir=None):
    """
    Perform iterative PSF photometry + isophote fitting on a galaxy image.
    
    Parameters
    ----------
    file_name : str
        Path to input galaxy FITS image.
    file_name_psf : str
        Path to PSF model FITS file.
    aper_gc : list of CircularAperture
        Apertures for globular clusters.
    aper_cdg2 : CircularAperture
        Aperture for the main galaxy CDG-2.
    r_inner : float, optional
        Inner radius for local background estimation in pixel. Default is 4.
    r_outer : float, optional
        Outer radius for local background estimation in pixel. Default is 6.
    aperture_radius : float, optional
        Radius for aperture photometry in pixel. Default is 3.
    min_separation : float, optional
        Minimum separation between sources in pixel. Default is 5.
    fix_gc_centers : bool, optional
        If True, fix GC centers.
    back_size : int, optional
        Size of background estimation boxes. Default is 32.
    sn_thre : float, optional
        Signal-to-noise threshold for masking nearby sources. Default is 2.
    smooth_stddev : float, optional
        Standard deviation for Gaussian smoothing. Default is 1.
    n_iter : int, optional
        Number of iterative refinement. Default is 3.
    isophote_fitting_kws : dict, optional
        Keywords for isophote fitting.
    output_dir : str, optional
        Directory to save output files.
    
    Returns
    -------
    table : astropy.table.Table
        Table of measurement results.
    gc_model : numpy.ndarray
        Model of globular clusters.
    iso_model : numpy.ndarray
        Isophote model of CDG-2.
    isolist : photutils.isophote.IsophoteList
        List of fitted isophotes.
    """
    
    # Read data
    data, header = fits.getdata(file_name, header=True)
    wcs = WCS(header)
    
    # Fill nan and subtract mean background
    data[np.isnan(data)] = np.nanmedian(data)
    bkg = np.median(data)
    data = data - bkg
    
    # Read PSF
    psf = fits.getdata(file_name_psf)
    if nan_pad:
        psf = psf[:-1,:-1]   # Remove the input PSF nan padding, as shape needs to be in odd
    psf = psf/np.nansum(psf)

    # Make PSF model
    psf_model = EPSFModel(psf, x_0=psf.shape[1]/2+0.5, y_0=psf.shape[0]/2+0.5, flux=1)
    yy, xx = np.mgrid[:psf.shape[0], :psf.shape[1]]
    psf_data = psf_model(xx, yy)
    
    # Fix the position of models if fix_gc_centers is True
    if fix_gc_centers:
        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True
        
    # Constrain the flux to be greater than 0
    psf_model.flux.bounds = (0, None)
    
    # Make mask for GCs and CDG-2
    mask_gc = np.logical_or.reduce([aper.to_mask().to_image(data.shape)>0.5 for aper in aper_gc])
    mask_cdg2 = aper_cdg2.to_mask().to_image(data.shape).astype(bool)
    
    # Compute a 2D background after masking CDG-2 
    back, back_rms = background_extraction(data, mask=mask_cdg2, box_size=back_size)
    
    ##### Run PSF photometry #####
    psf_photometry_kws = dict(r_inner=r_inner, 
                              r_outer=r_outer,
                              aperture_radius=aperture_radius,
                              min_separation=min_separation,
                              error=back_rms,
                              fit_shape=psf_data.shape)
    positions_gc = np.array([aper.positions for aper in aper_gc])

    phot, gc_model, residual = do_PSF_photometry(data, psf_model, positions_gc, verbose=False, **psf_photometry_kws)

    # Detect and mask nearby bright sources
    threshold = back + (sn_thre * back_rms)
    segm = detect_sources(data, threshold, mask=mask_cdg2, npixels=5)
    mask_src = segm.data > 0

    # Smooth the data for ellipse isophote fitting
    data_bgsub = residual - back
    data_conv = convolve_fft(data_bgsub, Gaussian2DKernel(smooth_stddev), mask=mask_gc | mask_src)

    ##### Run ellipse isophote fitting on the diffuse light  #####
    isolist, iso_model, _ = do_Isophote_Fitting(data_conv, 
                                                aper_cdg2.positions, 
                                                **isophote_fitting_kws)

    # Compute residual of isophote model
    residual_iso = data - iso_model

    # Re-do PSF photometry on the diffuse light subtracted image
    phot2, gc_model, _ = do_PSF_photometry(residual_iso, psf_model, positions_gc, **psf_photometry_kws) 
   
    # Run iteration
    columns = ['flux_gc', 'flux_gc_err', 'flux_cdg2', 'flux_cdg2_err', 'f_gc', 'f_gc_err']
    result = {column: np.zeros(n_iter) for column in columns}
    
    for k in tqdm(range(n_iter)):
        flux_gc = phot2['flux_fit'].sum()
        flux_cdg2 = iso_model.sum()
        f_gc = flux_gc/flux_cdg2
        
        flux_gc_err = np.sqrt(np.sum(phot2['flux_err']**2))
        flux_cdg2_err = np.sqrt(np.mean((isolist.int_err/isolist.intens)**2)) * flux_cdg2
        f_gc_err = f_gc * np.sqrt((flux_gc_err/flux_gc)**2+(flux_cdg2_err/flux_cdg2)**2)
        
        values = flux_gc, flux_gc_err, flux_cdg2, flux_cdg2_err, f_gc, f_gc_err, 
        for column, val in zip(columns, values):
            result[column][k] = val
        
        # print(f"Iteration {k+1:d}: Flux GC = {flux_gc:.3f}+/-{flux_gc_err:.3f}, Flux CDG-2 = {flux_cdg2:.3f}+/-{flux_cdg2_err:.3f}, F_GC/F_CDG-2 = {f_gc:.4f}+/-{f_gc_err:.4f}")
        
        # Subtract new GC model
        data_bgsub = data - gc_model - back
        data_conv = convolve_fft(data_bgsub, Gaussian2DKernel(smooth_stddev), mask=mask_gc | mask_src)

        # Re-do isophote fitting
        isolist, iso_model, _ = do_Isophote_Fitting(data_conv, aper_cdg2.positions, **isophote_fitting_kws)
        residual_iso = data - iso_model
        
        # print(iso_model.sum())
        
        # Re-do PSF photometry
        verbose = True if k>=n_iter-1 else False
        phot2, gc_model, residual = do_PSF_photometry(residual_iso, psf_model, positions_gc, verbose=verbose, **psf_photometry_kws) 
        
        # Compute the 2D background after subtracting CDG-2 and GCs
        back, back_rms = background_extraction(residual, mask=mask_src, box_size=back_size)
    
    flux_gc = phot2['flux_fit'].sum()
    flux_cdg2 = iso_model.sum()
    f_gc = flux_gc/flux_cdg2
    
    flux_gc_err = np.sqrt(np.sum(phot2['flux_err']**2))
    flux_cdg2_err = np.sqrt(np.mean((isolist.int_err/isolist.intens)**2)) * flux_cdg2
    f_gc_err = f_gc * np.sqrt((flux_gc_err/flux_gc)**2+(flux_cdg2_err/flux_cdg2)**2)
    
    # Final iteration
    values = flux_gc, flux_gc_err, flux_cdg2, flux_cdg2_err, f_gc, f_gc_err
    for column, val in zip(columns, values):
        result[column][-1] = val
    
    print(f"Flux GC = {flux_gc:.3f}+/-{flux_gc_err:.3f}, Flux CDG-2 = {flux_cdg2:.3f}+/-{flux_cdg2_err:.3f}, F_GC/F_CDG-2 = {f_gc:.4f}+/-{f_gc_err:.4f}\n")
    
    data = np.ma.array(data - back, mask=mask_src)
    models = {'gc':gc_model, 'iso':iso_model}
    
    # Write results to a table
    table = Table(result)
    
    table.write(os.path.join(output_dir, 'CDG-2_fitting.txt'), format='ascii', overwrite=True)
    
    # Calculate mean SB
    calculate_cdg2_mean_SB(isolist, data, pixel_scale, zp, cutoff=cutoff_SB)
    
    return table, data, models, isolist

    
if __name__ == "__main__":
    args = parse_arguments()
    main(args.file_name, args.file_name_psf, args.output_dir)
