#!/usr/bin/env python3
import os
import math
import argparse
from tqdm import tqdm
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import rcParams
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'viridis'
random_seed = 4676

import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, SigmaClip
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.visualization import ImageNormalize, LogStretch, AsinhStretch

from photutils.aperture import CircularAperture, EllipticalAperture, EllipticalAnnulus
from photutils.background import Background2D, SExtractorBackground
from photutils.isophote import build_ellipse_model
from photutils.datasets import make_noise_image
from photutils.centroids import centroid_com


def make_apertures(coords_cdg2, coords_gc, file_name, pixel_scale, aperture_cdg2=5, aperture_gc=0.6):
    """
    Create circular apertures for the CDG-2 galaxy and its globular clusters.
    
    Parameters
    ----------
    coords_cdg2 : tuple
        Celestial coordinates of the CDG-2 galaxy center (RA, Dec).
    coords_gc : list of tuples
        Celestial coordinates of globular clusters.
    file_name: str
        Path to the input FITS image file.
    pixel_scale: float
        Pixel scale in arcsec/pix.
    aperture_cdg2 : float, optional
        Aperture radius for CDG-2 in arcseconds. Default is 5.
    aperture_gc : float, optional
        Aperture radius for globular clusters in arcseconds. Default is 0.6.
    
    Returns
    -------
    aper_cdg2 : CircularAperture
        Aperture for the CDG-2 galaxy.
    aper_gc_mask : list of CircularAperture
        Apertures for globular clusters.
    """
    
    data, header = fits.getdata(file_name, header=True)
    wcs = WCS(header)

    radius_cdg2 = int(aperture_cdg2/pixel_scale) # in pixel
    radius_gc = int(aperture_gc/pixel_scale) # in pixel

    # CDG2 aperture
    coords_cdg2 = SkyCoord(ra=coords_cdg2[0], dec=coords_cdg2[1], unit=(u.hourangle, u.deg))
    pos_cdg2 = wcs.all_world2pix(coords_cdg2.ra, coords_cdg2.dec, 0)
    aper_cdg2 = CircularAperture(pos_cdg2, radius_cdg2)
    
    # GC apertures
    coords_gc = [SkyCoord(coords[0], coords[1], unit=(u.hourangle, u.deg)) for coords in coords_gc]
    positions_gc = np.array([wcs.all_world2pix(coords.ra, coords.dec, 0) for coords in coords_gc])
    aper_gc_mask = [CircularAperture(pos, radius_gc) for pos in positions_gc]
    
    return aper_cdg2, aper_gc_mask
    
    

def background_extraction(data, mask, box_size=32, filter_size=3):
    """
    Extract 2D background from the image.
    
    Uses photutils.background.Background2D to estimate background and background RMS.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D image data.
    mask : numpy.ndarray
        Mask (=1) to exclude certain regions from background estimation.
    box_size : int, optional
        Size of background estimation boxes. Default is 32.
    filter_size : int, optional
        Size of background smoothing filter. Default is 3.
    
    Returns
    -------
    bkg : numpy.ndarray
        Estimated background.
    bkg_rms : numpy.ndarray
        Estimated background RMS.
    """
    Bkg = Background2D(data, mask=mask,
                       bkg_estimator=SExtractorBackground(),
                       box_size=box_size, filter_size=filter_size,
                       sigma_clip=SigmaClip(sigma=3., maxiters=5))
    bkg = Bkg.background
    bkg_rms = Bkg.background_rms
    
    return bkg, bkg_rms

def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments with file_name, file_name_psf, and output_dir.
    """
    parser = argparse.ArgumentParser(description='Perform photometric analysis on a CDG-2 galaxy.')
    parser.add_argument('file_name', help='Path to the input FITS image file')
    parser.add_argument('file_name_psf', help='Path to the input PSF FITS file')
    parser.add_argument('-o', '--output_dir',
                        help='Directory to save output files (optional)',
                        default=None)
    
    return parser.parse_args()
    
def calculate_cdg2_mean_SB(isolist, data, pixel_scale, zp, cutoff=5):
    """
    Calculate mean surface brightness of the galaxy from the isophote model.
    
    Parameters
    ----------
    isolist : IsophoteList
        List of fitted isophotes.
    data : numpy.ndarray
        2D image data to perform photometry on.
    cutoff : float, optional
        Cutoff radius in arcseconds. Default is 5.
    """
    
    # Only use isophote models within a cutoff (checked with radial profile)
    index_cutoff = np.argmin(isolist.sma*pixel_scale < cutoff)
    iso_model_final = build_ellipse_model(data.shape, isolist[:index_cutoff])
    
    # Compute mean SB
    flux_iso = iso_model_final.sum()
    area = np.sum(iso_model_final>0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        SB_mean = -2.5*np.log10(flux_iso/area) + zp + 2.5 * np.log10(pixel_scale**2)
    
    print(f"Mean surface brightness CDG-2 (isophote model): {SB_mean:.2f} mag/arcsec^2\n")
    
    
def calculate_cdg2_luminosity(flux_cdg2, flux_gc, distance, zp, factor=0.5):
    """
    Calculate luminosity of the galaxy and GCs.
    
    Parameters
    ----------
    flux_cdg2 : float
        Flux of the galaxy (diffuse component).
    flux_gc : float
        Flux of globular clusters.
    zp: float
        Zero-point.
    """
    
    distance_modulus = 5*np.log10(distance.to(u.pc).value)-5

    flux_tot = flux_gc + flux_cdg2
    
    m_cdg2 = -2.5*np.log10(flux_cdg2) + zp
    M_cdg2 = m_cdg2 - distance_modulus
    M_cdg2 = M_cdg2+factor # conversion factor from I_E to V is 0.5
    L_cdg2 = 10**((M_cdg2-4.8)/(-2.5))
    
    print(f"Absolute V-mag CDG-2 (diffuse): {M_cdg2:.3f}")
    print(f"Luminosity CDG-2 (diffuse) in Solar Units: {L_cdg2:.3e}\n")
    
    m_gc = -2.5*np.log10(flux_gc) + zp
    M_gc = m_gc - distance_modulus
    M_gc = M_gc+factor # conversion factor from I_E to V is 0.5
    L_gc = 10**((M_gc-4.8)/(-2.5))
    
    print(f"Absolute V-mag GC: {M_gc:.3f}")
    print(f"Luminosity GC in Solar Units: {L_gc:.3e}\n")

##### Plotting function #####

def prepre_plot(data, models):
    
    # Using 1.5 * mad_std as estimate of uncertainty
    std = 1.5 * mad_std(data[~data.mask], ignore_nan=True)
    
    noise = make_noise_image(data.shape, distribution='gaussian', mean=0.0, stddev=0.5*std, seed=random_seed)
    
    gc_model, iso_model = models['gc'], models['iso']
    model_tot = gc_model + iso_model
    
    return model_tot, noise, std


def make_plot_Euclid(data, models, aper_cdg2, aper_gc, output_dir='./'):
    """ Plot the results """
    
    model_tot, noise, std = prepre_plot(data, models)
    data = data.data
    
    norm = ImageNormalize(data, stretch=AsinhStretch(0.25), vmin=-std/2, vmax=std*10)
    norm2 = ImageNormalize(data, stretch=AsinhStretch(0.25), vmin=-std/5, vmax=std*10)
        
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5), dpi=80, constrained_layout=True)
    ax1.imshow(data, norm=norm)
    ax2.imshow(model_tot + noise, norm=norm2)
    ax3.imshow(data - model_tot, norm=norm)

    for ax in (ax1, ax2,ax3):
        ax.set_xlim(data.shape[1]//2-75, data.shape[1]//2+75)
        ax.set_ylim(data.shape[0]//2-75, data.shape[0]//2+75)
        ax.axis('off')

        aper_cdg2.plot(color='r', ls='--', lw=3, ax=ax)
        for aper in aper_gc:
            aper.plot(color='w', lw=2, ax=ax)

    ax1.set_title('Data', fontsize=20)
    ax2.set_title('GC+Isophote model', fontsize=20)
    ax3.set_title('Residual', fontsize=20)

    plt.savefig(os.path.join(output_dir, 'CDG-2_Euclid_modeling_zoom.png'))
    plt.close()

def make_plot_CFHT(data, models,
                  pixel_scale, zp,
                  aper_cdg2, aper_gc,
                  cutoff_SB=5,
                  figsize=(16,5), output_dir='./'):
    import matplotlib.gridspec as gridspec
    
    model_tot, noise, std = prepre_plot(data, models)
    
    mask = data.mask
    data = data.data
    
    norm = ImageNormalize(data, stretch=AsinhStretch(0.25), vmin=-std/2, vmax=std*10)
    norm2 = ImageNormalize(data, stretch=AsinhStretch(0.25), vmin=-std/5, vmax=std*10)
    
    # centroid of isophote model
    gc_model, iso_model = models['gc'], models['iso']
    cen_iso = centroid_com(iso_model)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1.2], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(data, norm=norm, cmap='viridis')
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(model_tot+noise, norm=norm2, cmap='viridis')
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(data-model_tot, norm=norm, cmap='viridis')
    
    aper_cdg2_iso = CircularAperture(cen_iso, aper_cdg2.r)
    
    for ax in (ax1, ax2,ax3):
        ax.set_xlim(data.shape[1]//2-120, data.shape[1]//2+120)
        ax.set_ylim(data.shape[0]//2-120, data.shape[0]//2+120)
        ax.set_xticks([])
        ax.set_yticks([])

        aper_cdg2_iso.plot(color='r', ls='--', lw=1, ax=ax)
        ax.scatter(cen_iso[0], cen_iso[1], s=100, marker='+', color='r')
        
        for aper in aper_gc:
            aper.plot(color='w', lw=1, ax=ax)
            
    ax1.set_title('Data')
    ax2.set_title('Isophote + GC model')
    ax3.set_title('Residual')
        
    ax = fig.add_subplot(gs[3])
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        r_rbin, I_rbin, _ = cal_profile_1d(data, cen_iso, mask=mask, dr=0.8, seeing=1, sky_std=std,
                                           pixel_scale=pixel_scale, ZP=zp, xunit='arcsec', yunit="SB", color='orange', markersize=6, use_annulus=True, theta=np.pi/2, b_a=0.85, label='Data')

        r_rbin, I_rbin, _ = cal_profile_1d(model_tot, cen_iso, dr=0.8, seeing=1, sky_std=std,
                                           pixel_scale=pixel_scale, ZP=zp, xunit='arcsec', yunit="SB", markersize=6, use_annulus=True, theta=np.pi/2, b_a=0.85, label='GC+Isophote Model')

        r_rbin, I_rbin, _ = cal_profile_1d(data-gc_model, cen_iso, mask=mask, dr=0.8, seeing=1, sky_std=std,
                                           pixel_scale=pixel_scale, ZP=zp, xunit='arcsec', yunit="SB", color='g', markersize=6, use_annulus=True, theta=np.pi/2, b_a=0.85, label='Data $-$ GC Model')

    plt.ylim(32.5,25.5)
    plt.xlim(0.1,35)
    plt.axvline(cutoff_SB, ls='--', color='r')
    plt.legend(frameon=False)
    
    plt.savefig(os.path.join(output_dir, 'CDG-2_CFHT_modeling.png'))
    plt.close()

def cal_profile_1d(img, cen=None, mask=None, back=None, bins=None,
                       color="steelblue", xunit="pix", yunit="Intensity",
                       seeing=2.5, pixel_scale=2.5, ZP=27.1,
                       sky_mean=0, sky_std=3, dr=1,
                       lw=2, alpha=0.7, markersize=5, I_shift=0,
                       core_undersample=False, figsize=None,
                       label=None, plot_line=False, mock=False,
                       plot=True, errorbar=False,
                       scatter=False, fill=False,
                       use_annulus=False, theta=0, b_a=1):
    """
    Calculate 1d radial profile of a given star postage.
    """
    if mask is None:
        mask =  np.zeros_like(img, dtype=bool)
    if back is None:
        back = np.ones_like(img) * sky_mean
    bkg_val = np.median(back)
    if cen is None:
        cen = (img.shape[1]-1)/2., (img.shape[0]-1)/2.
    
    if use_annulus:
        img[mask] = np.nan
    
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
    r = rr[~mask].ravel()  # radius in pix
    z = img[~mask].ravel()  # pixel intensity
    r_core = np.int32(2 * seeing) # core radius in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * bkg_val
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.00005 * z_diff[-1]))
    r_max = np.min([img.shape[0]//2, np.sqrt(n_pix_max/np.pi)])
    
    if xunit == "arcsec":
        r *= pixel_scale   # radius in arcsec
        r_core *= pixel_scale
        r_max *= pixel_scale
        d_r = dr * pixel_scale
    else:
        d_r = dr
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clip = lambda z: sigma_clip((z), sigma=5, maxiters=5)
        
    if bins is None:
        # Radial bins: discrete/linear within r_core + log beyond it
        if core_undersample:
            # for undersampled core, bin at int pixels
            bins_inner = np.unique(r[r<r_core]) - 1e-3
        else:
            n_bin_inner = int(min((r_core/d_r*2), 6))
            bins_inner = np.linspace(0, r_core-d_r, n_bin_inner) - 1e-3

        n_bin_outer = np.max([6, np.min([np.int32(r_max/d_r/10), 50])])
        if r_max > (r_core+d_r):
            bins_outer = np.logspace(np.log10(r_core+d_r),
                                     np.log10(r_max+2*d_r), n_bin_outer)
        else:
            bins_outer = []
        bins = np.concatenate([bins_inner, bins_outer])
        _, bins = np.histogram(r, bins=bins)
    
    # Calculate binned 1d profile
    r_rbin = np.array([])
    z_rbin = np.array([])
    zerr_rbin = np.array([])
    
    for k, b in enumerate(bins[:-1]):
        r_in, r_out = bins[k], bins[k+1]
        in_bin = (r>=r_in) & (r<=r_out)
        if use_annulus:
            # Fractional ovelap w/ annulus
            annl = EllipticalAnnulus(cen, abs(r_in)/pixel_scale, r_out/pixel_scale, b_a*r_out/pixel_scale, theta=theta)
            annl_ma = annl.to_mask()
            # Intensity by fractional mask
            z_ = annl_ma.multiply(img)
            area = annl.area_overlap(img, mask=mask)
            # area = annl.area
            z_clip = z_[~np.isnan(z_)]
            zb = np.nansum(z_clip) / area
            zstd_b = np.std(z_clip) if len(z_clip) > 3 else 0
            zerr_b = np.sqrt((zstd_b**2 + sky_std**2) / area)
            rb = np.nanmean(r[in_bin])
            
        else:
            z_clip = clip(z[~np.isnan(z) & in_bin])
            if np.ma.is_masked(z_clip):
                z_clip = z_clip.compressed()
            if len(z_clip)==0:
                continue

            zb = np.mean(z_clip)
            zstd_b = np.std(z_clip) if len(z_clip) > 10 else 0
            zerr_b = np.sqrt((zstd_b**2 + sky_std**2) / len(z_clip))
            rb = np.mean(r[in_bin])
           
        z_rbin = np.append(z_rbin, zb)
        zerr_rbin = np.append(zerr_rbin, zerr_b)
        r_rbin = np.append(r_rbin, rb)

    logzerr_rbin = 0.434 * abs( zerr_rbin / (z_rbin-sky_mean))
    
    if yunit == "SB":
        I_rbin = Intensity2SB(z_rbin, BKG=bkg_val,
                              ZP=ZP, pixel_scale=pixel_scale) + I_shift
    
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        if yunit == "Intensity":
            # plot radius in Intensity
            good = np.isfinite(np.log10(z_rbin))
            plt.plot(r_rbin[good], np.log10(z_rbin)[good], "-o", color=color,
                     mec="k", lw=lw, ms=markersize, alpha=alpha, zorder=3, label=label)
            if scatter:
                I = np.log10(z)
                
            if fill:
                plt.fill_between(r_rbin, np.log10(z_rbin)-logzerr_rbin, np.log10(z_rbin)+logzerr_rbin,
                                 color=color, alpha=0.2, zorder=1)
            plt.ylabel("log Intensity")

        elif yunit == "SB":
            # plot radius in Surface Brightness
            if mock is False:
                I_sky = -2.5*np.log10(sky_std) + ZP + 2.5 * math.log10(pixel_scale**2)
            
            good = np.isfinite(I_rbin)
            p = plt.plot(r_rbin[good], I_rbin[good], "-o", mec="k",
                         lw=lw, ms=markersize, color=color,
                         alpha=alpha, zorder=3, label=label)
            if scatter:
                I = Intensity2SB(z, BKG=bkg_val,
                                 ZP=ZP, pixel_scale=pixel_scale) + I_shift
                
            if errorbar:
                Ierr_rbin_up = I_rbin - Intensity2SB(z_rbin+zerr_rbin, BKG=bkg_val,
                                 ZP=ZP, pixel_scale=pixel_scale) - I_shift
                Ierr_rbin_lo = Intensity2SB(z_rbin-zerr_rbin, BKG=bkg_val,
                                ZP=ZP, pixel_scale=pixel_scale) - I_rbin + I_shift
                lolims = np.isnan(Ierr_rbin_lo)
                uplims = np.isnan(Ierr_rbin_up)
                Ierr_rbin_lo[lolims] = 99
                Ierr_rbin_up[uplims] = np.nan
                plt.errorbar(r_rbin, I_rbin, yerr=[Ierr_rbin_up, Ierr_rbin_lo],
                             fmt='', ecolor=p[0].get_color(), capsize=2, alpha=0.5)
                
            plt.ylabel("Surface Brightness [mag/arcsec$^2$]", fontsize=20)
            plt.gca().invert_yaxis()
            plt.ylim(30,17)

        plt.xscale("log")
        plt.xlim(max(r_rbin[np.isfinite(r_rbin)][0]*0.8, pixel_scale*0.5),
                 r_rbin[np.isfinite(r_rbin)][-1]*1.2)
        if xunit == "arcsec":
            plt.xlabel("Radius [arcsec]", fontsize=20)
        else:
            plt.xlabel("radius [pix]", fontsize=20)
        
        if scatter:
            plt.scatter(r[r<3*r_core], I[r<3*r_core], color=color,
                        s=markersize/2, alpha=alpha/2, zorder=1)
            plt.scatter(r[r>=3*r_core], I[r>=3*r_core], color=color,
                        s=markersize/5, alpha=alpha/10, zorder=1)
            

        # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dz_rbin = np.diff(np.log10(z_rbin))
        dz_cum = np.cumsum(dz_rbin)

        if plot_line:
            r_satr = r_rbin[np.argmax(dz_cum<-0.5)] + 1e-3
            plt.axvline(r_satr,color="k",ls="--",alpha=0.9)
            plt.axvline(r_core,color="k",ls=":",alpha=0.9)
            if yunit == "SB":
                plt.axhline(I_sky,color="gray",ls="-.",alpha=0.7)
        
    if yunit == "Intensity":
        return r_rbin, z_rbin, logzerr_rbin
    elif yunit == "SB":
        return r_rbin, I_rbin, None

def Intensity2SB(Intensity, BKG, ZP, pixel_scale):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """
    I = np.atleast_1d(np.copy(Intensity))
    I[np.isnan(I)] = BKG
    if np.any(I<=BKG):
        I[I<=BKG] = np.nan
    I_SB = -2.5*np.log10(I - BKG) + ZP + 2.5 * math.log10(pixel_scale**2)
    return I_SB
