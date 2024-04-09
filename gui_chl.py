# -*- coding: utf-8 -*-

"""Main module."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pandas import Timedelta
from pyproj import Geod
from scipy.optimize import least_squares
import xarray as xr

try:
    import oceansdb

    OCEANSDB_AVAILABLE = True
except:
    OCEANSDB_AVAILABLE = False

module_logger = logging.getLogger(__name__)

def fit_fchl_exponential_decay(df):
    """
    Returns
    -------
    sigma : float
        Standard deviation of the residue
    """
    df = df.dropna(how="any")
    df = df.sort_values(by="profile")

    t = np.array(df.running_time) / 86400.0
    fchl = np.array(df.fchl_deep_ref)

    def fun(x, t, y):
        return x[0] + x[1] * np.exp(-t / x[2]) - y

    # fchl0, fchl_scale, t_scale
    x0 = [np.percentile(fchl, 3), 0.1, 14.0]
    res_robust = least_squares(fun, x0, loss="soft_l1", f_scale=0.001, args=(t, fchl))
    assert res_robust.success

    output = {
        "exp_fchl0": res_robust.x[0],
        "exp_fchl_scale": res_robust.x[1],
        "exp_t_scale": res_robust.x[2] * 86400.0,
        "sigma": fun(res_robust.x, t, fchl).std(),
    }
    return output


def fchl_dark_offset_castelao2018(da, threshold: float = 0.01, decay: bool = False):
    """Dark count correction as Castelao and Rudnick 2020

    Assumes that deep chlorophyll fluorescence measurements should tend to zero
    in the deep layers. Here it is imposed measurements at least up to 200m.

    Returns
    -------
    fchl_unbiased
        Unbiased chlorophyll fluorescence, i.e. wihtout offset from dark count
    fchl_deep_ref
        The fchl deep reference, from deep layers of only offshore profiles
        after smoothed to remove spikes.
    fchl_dark
        Dark count offset reference. Estimated value obtained in absence of
        chlorophyll.
    exp_fchl0
    exp_fchl_scale
        The scale for the fchl exponential decay, i.e. the correction offset at
        the initial time 0.
    exp_t_scale
        The time decay scale for the fchl exponential decay.
    sigma
        The standard deviation of fchl_deep_ref - the exponential decay fit.
        This gives an idea of how good was the fit.

    Notes
    -----
    - Chlorophyll fluorescence is a relatively noisy measurement, so smooth it
      before defining the dark count. The unbias output is based on the actual
      input, thus the output is not smoothed.
    """
    if "trajectory" in da.dims:
        return da.groupby("trajectory").map(
            fchl_dark_offset_castelao2018, threshold=threshold, decay=decay
        )
    da = da.dropna(dim="profile", how="all").dropna(dim="obs", how="all")

    # Simplistic criterion: Local depth >=450m & profile at least 200m
    # Local bathymetry
    bathymetry = get_bathymetry_OceansDB(da)
    # Maximum measured depth
    profile_max_depth = da.depth.max(dim="obs")
    profile_max_depth.attrs["description"] = "Deepest measurement per profile"
    # Simplex deep offshore flag
    deep_offshore_flag = (bathymetry >= 450) & (profile_max_depth >= 200)
    deep_offshore_flag.attrs[
        "description"
    ] = "Offshore simplex criteria: bathymetry >= 450 & profile at least up to 200m depth"

    # Remove high frequency variability (mostly spikes)
    fchl_deep = da.groupby("profile").map(
        lambda x: x.dropna(dim="obs")
        .rolling(obs=5, center=True)
        .median()
        .dropna(dim="obs")
        .rolling(obs=3, center=True)
        .mean()
        .dropna(dim="obs")
    )
    # zlimit = ds.depth.quantile(0.99)
    # idx = (ds.depth >= 120) & (ds.depth <= zlimit)
    fchl_deep = fchl_deep.where(deep_offshore_flag, np.nan).where(
        fchl_deep.depth >= 80, np.nan
    )
    fchl_deep = fchl_deep.dropna(dim="profile", how="all").dropna(dim="obs", how="all")
    # Maybe introduce alternative for minimum instead of a percentile
    fchl_deep_ref = fchl_deep.quantile(0.025, dim="obs")

    description = "Using robust minimum fl in profile as deep reference."

    if decay is True:
        tmp = xr.Dataset()
        tmp["running_time"] = trajectory_chronometer(da.profile_time)
        tmp["fchl_deep_ref"] = fchl_deep_ref
        df = tmp.reset_coords()[
            ["profile", "fchl_deep_ref", "running_time"]
        ].to_dataframe()
        coef = fit_fchl_exponential_decay(df)
        # Time dependent component only
        fchl_decay = coef["exp_fchl_scale"] * np.exp(
            -tmp.running_time / coef["exp_t_scale"]
        ).rename("exp_fchl_decay")
        fchl_decay = fchl_decay.reset_coords(drop=True)
        # Remove the time dependent component from the deep layer data
        fchl_deep = fchl_deep - fchl_decay
        # Dark count: a percentile from time independent deeplayer.
        fchl_dark = fchl_deep.quantile(threshold).rename(
            {"quantile": "fchl_dark_quantile"}
        )
        coef["fchl_dark"] = float(fchl_dark)
        # A dark count time dependent, thus a 1D array.
        fchl_offset = fchl_decay + fchl_dark

        description = "Using robust minimum fl in profile as deep reference."

    else:
        # A dark count constant, thus a scalar.
        fchl_dark = fchl_deep.quantile(threshold)
        coef = {"fchl_dark": float(fchl_dark)}
        fchl_offset = fchl_dark

    fchl_unbiased = da - fchl_offset
    # expand_dims('f0_method').assign_coords(f0_method=['min'])
    fchl_unbiased = fchl_unbiased.where(
        (fchl_unbiased >= 0) | (fchl_unbiased.isnull()), 0
    )
    fchl_unbiased = fchl_unbiased.rename("fchl_unbiased")
    fchl_unbiased.attrs["long_name"] = "Chlorophyll Fluorescence (unbiased)"
    fchl_unbiased.attrs["description"] = "The fchl after removing dark count"

    output = fchl_unbiased.to_dataset()
    output["fchl_deep_ref"] = fchl_deep_ref.reset_coords(drop=True)
    for k in coef:
        output[k] = coef[k]

    return output

def chl_vis_morel2001(da, smooth: bool = True):
    """Estimates Euphotic depth from a chl profile and related products.

    Based on Morel 2001, integrate the total Chl from the surface towards the
    bottom and estimate the depth of the euphotic zone (z_eu, 1% of the
    surface). From that, obtain the first penetration depth (z_pd, 1/e from
    the surface). The visible chlorophyll is obtained from an weighted average
    considering the layer thickness that a measurement is representative and
    the light assuming an exponential decay scaled by z_pd.

    Parameters
    ----------
    da : xr.DataArray
        Chl profile, i.e., chl measurement in respect to depth. Depth should
        be a coordinate of this da.
    smooth: boolean
        If True smooth the chl with median 5pts than mean 3 pts before
        estimating the euphotic depth.

    Returns
    -------
    xr.Dataset
      - z_eu: Euhpotic depth
      - z_pd: First penetration depth
      - chl_pd: Average chl on the layer above z_pd
      - chl_vis: Sea surface footprint of chl, i.e., visible chl from the sky

    Notes
    -----
    - Remove measurements when oscilating vertically is not a job for here but
      for a QC procedure that should be done before this stage. Here we should
      assume that all data are valid and do the best with it.
    - The measurements are ordered by depth, thus if there are oscillations it
      will consider the combined effect.
    - The top layer has a higher weight due to less light decay, and also tend
      to be thicker since it's hard to get a valid measurement on the surface.
      Thus, if not smoothed, this might bias the final result.
    - Alternative, but it didn't work for me.
       chl_log = np.log10(chl_tot)
       10**(2.1236 + 0.932468*chl_log -1.4264*chl_log**2 +
            0.52776 * chl_log**3 -0.07617 * chl_log**4) - z
    - Confirm that for Morel 2001, surface chl was the average chlorophyll
      on the first penetration depth
    - Should it impose some minimum sampling near the surface to prevent
      profiles with only subsurface data to quietly pass?
    """
    assert isinstance(da, xr.DataArray)

    dims = [d for d in da.dims if d not in ("profile", "obs")]
    if len(dims) > 0:
        module_logger.debug("Groupping by {}".format(dims[0]))
        return da.groupby(dims[0]).map(chl_vis_morel2001, restore_coord_dims=True)
    elif "profile" in da.dims:
        return (
            da.groupby("profile", restore_coord_dims=True)
            .map(chl_vis_morel2001, smooth=smooth)
            .expand_dims("ze_method")
            .assign_coords(ze_method=["Morel2001"])
        )

    empty_answer = xr.Dataset(
        data_vars=dict(z_eu=np.nan, z_pd=np.nan, chl_pd=np.nan, chl_vis=np.nan)
    )

    assert da.dims == ("obs",), "For now it is expected single dimension obs"
    assert "depth" in da.coords, "Missing depth coordinate"

    if smooth:
        da = da.isel(obs=~np.isnan(da.depth))
        da = da.rolling(obs=5, min_periods=3, center=True).median(keep_attrs=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            da = da.rolling(obs=3, center=True, min_periods=2).mean(keep_attrs=True)

    # Remove nan.
    # Data flagged bad should be NaN at this point, but a valid depth should
    # indicate that there was a measurement at that level
    da = da.dropna(dim="obs")
    # Must be sorted to correctly estimate the cummulative effect.
    if da.depth.median() < 0:
        module_logger.warning("Depth was expected to be positive.")
    da = da.sortby(da.depth, ascending=True)

    z = da.depth.data
    chl = da.data

    # Define layers
    dz = np.diff(z)
    bottom = np.append(z[:-1] + dz / 2.0, z[-1])
    top = np.append([0], z[1:] - dz / 2.0)
    thickness = bottom - top

    # [mg m^2] accumulated from surface up to that depth
    chl_cumsum = np.cumsum(chl * thickness)
    # Morel (2001)
    max_penetration = 912.5 * chl_cumsum ** (-0.839)

    idx = np.nonzero((z - max_penetration) > 0)[0]
    if not idx.size > 0:
        module_logger.info(
            "Profile too shallow. Penetration: {}, max depth:{})".format(
                min(max_penetration), max(z)
            )
        )
        return empty_answer

    idx = np.min(idx) + 2
    # Remember: np.interp requires increasing xp (2nd arg)
    z_eu = np.interp(0, z[:idx] - max_penetration[:idx], z[:idx])

    if z_eu > 102:
        max_penetration = 426.3 * chl_cumsum ** (-0.547)

        idx = np.nonzero((z - max_penetration) > 0)[0]
        if not idx.size > 0:
            module_logger.info(
                "Profile too shallow. Penetration: {}, max depth:{})".format(
                    min(max_penetration), max(z)
                )
            )
            return empty_answer

        idx = np.min(idx) + 2
        # Remember: np.interp requires increasing xp (2nd arg)
        z_eu = np.interp(0, z[:idx] - max_penetration[:idx], z[:idx])

    if z_eu > 180:
        module_logger.warning("Morel (2001) valid up to 180m but z_eu={}m".format(z_eu))
        return empty_answer

    # First penetration depth
    z_pd = z_eu / -np.log(0.01)
    # Weight, based on the exp decay for each layer
    w = z_pd * (np.exp(-2 * top / z_pd) - np.exp(-2 * bottom / z_pd))
    chl_vis = (chl * w).sum() / w.sum()
    # weight = np.exp(-(2*z)/z_pd) * thickness
    # chl_vis = (chl * weight).sum() / weight.sum()
    # chl_vis = np.sum(chl * np.exp(-(2*z)/z_pd) * thickness)
    # z_pd = - zeu_from_morel2001(z, chl) / np.log(0.01)
    # chl * thickness => Gives absolute chl per layer
    # chl_cumsum => Gives the total chl up to the bottom of the considered layer
    # chl_pd => Mean Chl in the first penetration depth (layer)
    # Remember: np.interp requires increasing xp (2nd arg)
    chl_pd = np.interp(z_pd, bottom, chl_cumsum) / z_pd

    return xr.Dataset(
        data_vars=dict(z_eu=z_eu, z_pd=z_pd, chl_pd=chl_pd, chl_vis=chl_vis)
    )