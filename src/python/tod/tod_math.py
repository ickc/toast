# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import scipy.constants as constants

import scipy.fftpack as sft
import scipy.interpolate as si
import scipy.sparse as sp

from .. import rng as rng
from .. import qarray as qa


def calibrate( toitimes, toi, gaintimes, gains, order=0, inplace=False ):
    """
    Interpolate the gains to TOI samples and apply them.

    Args:
        toitimes (float):  Increasing TOI sample times in same units as gaintimes
        toi (float):  TOI samples to calibrate
        gaintimes (float):  Increasing timestamps of the gain values in same units
                            as toitimes
        gains (float):  Multiplicative gains
        order (int):  Gain interpolation order. 0 means steps at the gain times,
                      all other are polynomial interpolations.
        inplace (bool):  Overwrite input TOI.

    Returns:
        calibrated timestream.
    """

    if len(gaintimes) == 1:
        g = gains
    else:    
        if order == 0:
            ind = np.searchsorted( gaintimes, toitimes )
            ind[ind > 0] -= 1
            g = gains[ind]
        else:
            if len(gaintimes) <= order:
                order = len(gaintimes) - 1
            p = np.polyfit( gaintimes, gains, order )
            g = np.polyval( p, toitimes )

    if inplace:
        toi_out = toi
    else:
        toi_out = np.zeros_like(toi)

    toi_out[:] = toi * g

    return toi_out


def sim_noise_timestream(realization, telescope, component, obsindx, detindx,
                         rate, firstsamp, samples, oversample, freq, psd):
    """
    Generate a noise timestream, given a starting RNG state.

    Use the RNG parameters to generate unit-variance Gaussian samples
    and then modify the Fourier domain amplitudes to match the desired
    PSD.

    The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
    which each consist of two unsigned 64bit integers.  These four
    numbers together uniquely identify a single sample.  We construct
    those four numbers in the following way:

    key1 = realization * 2^32 + telescope * 2^16 + component
    key2 = obsindx * 2^32 + detindx 
    counter1 = currently unused (0)
    counter2 = sample in stream

    counter2 is incremented internally by the RNG function as it calls
    the underlying Random123 library for each sample.

    Args:
        realization (int): the Monte Carlo realization.
        telescope (int): a unique index assigned to a telescope.
        component (int): a number representing the type of timestream
            we are generating (detector noise, common mode noise,
            atmosphere, etc).
        obsindx (int): the global index of this observation.
        detindx (int): the global index of this detector.
        rate (float): the sample rate.
        firstsamp (int): the start sample in the stream.
        samples (int): the number of samples to generate.
        oversample (int): the factor by which to expand the FFT length
            beyond the number of samples.
        freq (array): the frequency points of the PSD.
        psd (array): the PSD values.

    Returns (tuple):
        the timestream array, the interpolated PSD frequencies, and
            the interpolated PSD values.
    """
    
    fftlen = 2
    while fftlen <= (oversample * samples):
        fftlen *= 2
    npsd = fftlen // 2 + 1
    norm = rate * float(npsd - 1)

    interp_freq = np.fft.rfftfreq(fftlen, 1/rate)
    if interp_freq.size != npsd:
        raise RuntimeError("interpolated PSD frequencies do not have expected "
                           "length")

    # Ensure that the input frequency range includes all the frequencies
    # we need.  Otherwise the extrapolation is not well defined.

    if np.amin(freq) < 0.0:
        raise RuntimeError("input PSD frequencies should be >= zero")

    if np.amin(psd) < 0.0:
        raise RuntimeError("input PSD values should be >= zero")

    increment = rate / fftlen

    if freq[0] > increment:
        raise RuntimeError("input PSD does not go to low enough frequency to "
                           "allow for interpolation")

    nyquist = rate / 2
    if np.abs((freq[-1]-nyquist)/nyquist) > .01:
        raise RuntimeError(
            "last frequency element does not match Nyquist "
            "frequency for given sample rate: {} != {}".format(
                freq[-1], nyquist))

    # Perform a logarithmic interpolation.  In order to avoid zero values, we 
    # shift the PSD by a fixed amount in frequency and amplitude.

    psdshift = 0.01 * np.amin(psd[(psd > 0.0)])
    freqshift = increment

    loginterp_freq = np.log10(interp_freq + freqshift)
    logfreq = np.log10(freq + freqshift)
    logpsd = np.log10(psd + psdshift)

    interp = si.interp1d(logfreq, logpsd, kind='linear',
                         fill_value='extrapolate')
    
    loginterp_psd = interp(loginterp_freq)
    interp_psd = np.power(10.0, loginterp_psd) - psdshift

    # Zero out DC value

    interp_psd[0] = 0.0

    # gaussian Re/Im randoms, packed into a complex valued array

    key1 = realization * 4294967296 + telescope * 65536 + component
    key2 = obsindx * 4294967296 + detindx 
    counter1 = 0
    counter2 = firstsamp * oversample

    rngdata = rng.random(2*npsd, sampler="gaussian", key=(key1, key2), 
        counter=(counter1, counter2))
    fdata = rngdata[:npsd] + 1j * rngdata[npsd:]

    # set the Nyquist frequency imaginary part to zero

    fdata[-1] = fdata[-1].real + 0.0j

    # scale by PSD

    scale = np.sqrt(interp_psd * norm)
    fdata *= scale

    # inverse FFT

    tdata = np.fft.irfft(fdata)

    # subtract the DC level- for just the samples that we are returning

    offset = (fftlen - samples) // 2

    DC = np.mean(tdata[offset:offset+samples])
    tdata[offset:offset+samples] -= DC

    # return the timestream and interpolated PSD for debugging.
    
    return (tdata[offset:offset+samples], interp_freq, interp_psd)


def dipole(pntg, vel=None, solar=None, cmb=2.72548, freq=0):
    """
    Compute a dipole timestream.

    This uses detector pointing, telescope velocity and the solar system
    motion to compute the observed dipole.  It is assumed that the detector
    pointing, the telescope velocity vectors, and the solar system velocity 
    vector are all in the same coordinate system.

    Args:
        pntg (array): the 2D array of quaternions of detector pointing.
        vel (array): 2D array of velocity vectors relative to the solar 
            system barycenter.  if None, return only the solar system dipole.
            Units are km/s
        solar (array): a 3 element vector containing the solar system velocity
            vector relative to the CMB rest frame.  Units are km/s.
        cmb (float): CMB monopole in Kelvin.  Default value from Fixsen 
            2009 (see arXiv:0911.1955)
        freq (float): optional observing frequency in Hz (NOT GHz).

    Returns:
        (array):  detector dipole timestream.
    """

    zaxis = np.array([0,0,1], dtype=np.float64)
    nsamp = pntg.shape[0]

    inv_light = 1.0e3 / constants.speed_of_light

    # accumulate velocity components
    v = np.zeros((nsamp, 3), dtype=np.float64)

    if (vel is not None) and (solar is not None):
        # relativistic addition of velocities

        vsol = np.tile(solar, nsamp).reshape((-1,3))

        solar_speed = np.sqrt(np.sum(solar * solar, axis=0))

        vpar = ( np.sum(vel * vsol, axis=1).reshape((-1,1)) / solar_speed**2 ) * vsol
        vperp = vel - vpar

        vdot = 1.0 / ( 1.0 + solar * vel * inv_light**2 )
        invgamma = np.sqrt( 1.0 - (solar_speed * inv_light)**2 )

        vpar = vdot * (vpar * solar)
        vperp = vdot * vperp * invgamma

        v += vpar + vperp
    else:
        if solar is not None:
            v += np.tile(solar, nsamp).reshape((-1,3))
        if vel is not None:
            v += vel

    speed = np.sqrt(np.sum(v * v, axis=1)).reshape((-1,1))
    v[:] /= speed[:]

    beta = inv_light * speed.flatten()

    dir = qa.rotate(pntg, np.tile(zaxis, nsamp).reshape((-1,3)))

    dipoletod = None
    if freq == 0:
        inv_gamma = np.sqrt(1.0 - beta**2)
        num = 1.0 - beta * np.sum(v * dir, axis=1)
        dipoletod = cmb * ( inv_gamma / num - 1.0 )
    else:
        # Use frequency for quadrupole correction
        fx = constants.h * freq / (constants.k * cmb)
        fcor = (fx / 2) * (np.exp(fx) + 1) / (np.exp(fx) - 1)
        bt = beta * np.sum(v * dir, axis=1)
        dipoletod = cmb * (bt + fcor * bt**2)

    return dipoletod
