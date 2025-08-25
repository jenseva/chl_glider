# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional

import numpy as np
from numpy import ma
#import oceansdb
import pandas as pd
from pandas import Timedelta
from pyproj import Geod
from scipy.optimize import least_squares
import xarray as xr
import warnings


def mld_xing2012_profile(p, threshold: Optional[float] = 0.003, z_ref: Optional[float] = 15):
    """Mixed layer depth according to Xing et al. 2012

    Which is based on de Boyer Montégut et al. 2004.

    Parameters
    ----------
    p :
    threshold : float, optional
    z_ref : float, optional

    Notes
    -----
    - Assume sigma ( rho - 1000) is available.
    - Expects a single profile.
    - Improve to accept a full dive, i.e. descending and ascending together.
    - Expects monotonic vertical movement
    - Should I take the reference come from the closest depth or interpolate?
    """
    assert 'depth' in p, "Missing depth"
    assert 'sigma' in p, "Requires sigma to estiamte MLD"

    p = p.dropna(dim='obs', how='any', subset=('depth', 'sigma'))
    ref = p.isel(obs=(np.abs(p.depth - z_ref).argmin()))
    mld = np.interp(ref.sigma + 0.03, p.sigma[::-1], p.depth[::-1])

    assert mld >= 15, "Xing2012 implies at least 15m. Something went wrong."

    mld = min(180, mld)
    mld = xr.DataArray(mld, name='MLD')
    mld.attrs["method"] = "Xing et al. 2012"
    mld.attrs["z_ref"] = z_ref
    mld.attrs["units"] = "m"
    mld.attrs["comments"] = "Possible range between 15 and 180 m"

    return mld


def mixed_layer_depth(ds, threshold: Optional[float] = 0.003):
    """Mixed layer depth for a Dataset

    Estimate the mixed layer depth for a collection of profiles as a Dataset.

    Notes
    -----
    - It is expected more methods in the future. Currently only Xing2012
    - Need a solution to handle higher dimension ds, such as multiple
      trajectories.
    """
    assert 'depth' in ds, "Missing depth"
    assert 'sigma' in ds, "Requires sigma to estiamte MLD"

    valid = ds[['depth', 'sigma']].dropna(dim='profile', how='all')
    mld = valid.groupby('profile').apply(mld_xing2012_profile)

    return mld


def daylight_simplex(ds, time_name="time", lon_name="lon"):
    """A minimalist approach to guess if it was daylight

    An approximate solution to guess if a measurement was obtained on daylight.
    It expects time to be UTC, not local.

    Estimate local time with resolution of minutes from longitude

       To determine if a profile was on daylight or night time we need the
         local time. The official local time would do 1hr jumps, so I
         estimate within minutes from the longitude for a continuous timing.

    Parameters
    ----------
    ds :
    """
    assert time_name in ds
    assert lon_name in ds

    local_dt = ds.lon.to_series().apply(
                   lambda x: Timedelta(x/360., 'D')).to_xarray()
    local_time = (ds.time + local_dt)
    daylight = (local_time.dt.hour >= 6) & (local_time.dt.hour < 18)
    return daylight.rename("daylight")


def non_photochemical_quenching_xing2012(ds):
    """Non Photochemical Quenching (NPQ) correction (Xing et al. 2012)

    In summary, for daytime, measurements above the max fl observed inside the
    mixed layer (ML) are replaced by the max fl.

    Parameters
    ----------
    ds :
    """
    assert 'lon' in ds, "Missing lon"
    assert 'fl' in ds, "Missing fl"
    assert 'depth' in ds, "Missing depth"

    tmp = ds[['fl', 'depth']].set_coords('depth')
    mld = mixed_layer_depth(ds)
    tmp = tmp.merge(mld).set_coords('MLD')
    daylight = daylight_simplex(ds[['lon', 'profile_time']].rename({'profile_time': 'time'})).reset_coords(drop=True)
    tmp = tmp.merge(daylight).set_coords('daylight')

    tmp['fl_smooth'] = tmp.groupby('profile').map(lambda p: xr.DataArray(p.to_dataframe().fl.rolling(3, center=True).median(), name='fl_smooth'))
    tmp['fl_ml'] = tmp.fl_smooth.where(tmp.depth < tmp.MLD, np.nan)

    def single_profile(p):
        """Inside the mixed layer, the maximum fl and its deepest occurence

        This is a temporary solution. It would probably be better if could be
        done with with xarray, all profiles at once.
        """
        # p.dropna(dim='obs').sortby(['fl_ml', 'depth'], ascending=False)
        tmp = p.to_dataframe()
        tmp = tmp.sort_values(['fl_ml', 'depth'], ascending=False).iloc[0]
        tmp = tmp[['depth', 'fl_ml']]
        tmp = xr.Dataset(tmp).rename({'depth': 'fl_max_depth', 'fl_ml': 'fl_max'})
        return tmp

    tmp = tmp.merge(tmp.groupby('profile').map(single_profile))
    fl = tmp.fl.where(~tmp.daylight | (tmp.depth > tmp.fl_max_depth) | (tmp.fl > tmp.fl_max), tmp.fl_max)

    fl.attrs["method"] = "Xing et al. 2012"
    fl.attrs["comments"] = "Chlorophyll fluorescence corrected for non photochemical quenching"

    return fl


def convert_missionObs2profileObs(ds):
    """ Add new dimension groupping observations per profile

        For example, temp(mission, obs) is reorganized into
          temp(mission, profile, obs)

        It should use profile_obs if there is one

    Build tests to convert:
    <xarray.Dataset>
Dimensions:       (mission: 1, obs: 283363, profile: 394)
Coordinates:
    profile_id    (mission, obs) uint16 ...
    profile_time  (mission, profile) datetime64[ns] ...
    profile_lat   (mission, profile) float64 ...
    profile_lon   (mission, profile) float64 ...
    depth         (mission, obs) float64 ...
    fchl_qc       (mission, obs) uint8 7 7 7 7 7 7 7 7 7 7 ... 0 0 0 0 0 0 0 0 0
  * mission       (mission) object '18505901'
  * profile       (profile) uint64 1 2 3 4 5 6 7 ... 388 389 390 391 392 393 394
  * obs           (obs) int64 0 1 2 3 4 5 ... 283358 283359 283360 283361 283362
Data variables:
    fchl          (mission, obs) float64 nan nan nan nan ... 0.507 0.504 0.504
    dive_obs      (mission, obs) int64 ...
    time          (mission, obs) datetime64[ns] ...
    press         (mission, obs) float64 ...
    theta         (mission, obs) float64 ...
    sigma_theta   (mission, obs) float64 ...
Attributes:
    keywords:             AUVS > Autonomous Underwater Vehicles, Oceans > Oce...
    keywords_vocabulary:  GCMD Science Keywords
    history:              readsat - 2018-09-27T11:41:13Z, fixgps3 - 2018-09-2...
    date_created:         2020-06-22T04:57:07
    date_modified:        2021-03-12T20:21:49.837156
    platform:             sp059
    mission:              18505901
    src_filename:         spray.nc
    src_checksum:         6ef7125dc432ddd9b2d87b866395d0b5
    description:          MD059 18/05:01 Med      Calypso Pilot SOCIB boat ER...

    into:
    <xarray.Dataset>
Dimensions:       (mission: 1, obs: 1152, profile: 394)
Coordinates:
    profile_time  (mission, profile) datetime64[ns] ...
    profile_lat   (mission, profile) float64 ...
    profile_lon   (mission, profile) float64 ...
    depth         (mission, profile, obs) float64 103.8 103.4 102.7 ... nan nan
    fchl_qc       (mission, profile, obs) float32 7.0 7.0 7.0 ... nan nan nan
  * mission       (mission) object '18505901'
  * profile       (profile) uint64 1 2 3 4 5 6 7 ... 388 389 390 391 392 393 394
  * obs           (obs) int64 0 1 2 3 4 5 6 ... 1146 1147 1148 1149 1150 1151
Data variables:
    fchl          (mission, profile, obs) float64 nan nan nan ... nan nan nan
    dive_obs      (mission, profile, obs) float64 237.0 238.0 239.0 ... nan nan
    time          (mission, profile, obs) datetime64[ns] 2018-05-24T15:53:35 ...
    press         (mission, profile, obs) float64 104.6 104.2 103.5 ... nan nan
    theta         (mission, profile, obs) float64 nan nan nan ... nan nan nan
    sigma_theta   (mission, profile, obs) float64 nan nan nan ... nan nan nan
    mission_obs   (mission, profile, obs) float64 0.0 1.0 2.0 ... nan nan nan
Attributes:
    keywords:             AUVS > Autonomous Underwater Vehicles, Oceans > Oce...
    keywords_vocabulary:  GCMD Science Keywords
    history:              readsat - 2018-09-27T11:41:13Z, fixgps3 - 2018-09-2...
    date_created:         2020-06-22T04:57:07
    date_modified:        2021-03-12T20:21:49.837156
    platform:             sp059
    mission:              18505901
    src_filename:         spray.nc
    src_checksum:         6ef7125dc432ddd9b2d87b866395d0b5
    description:          MD059 18/05:01 Med      Calypso Pilot SOCIB boat ER...
    """
    assert 'profile_id' in ds
    varnames = [v for v in ds if sorted(ds[v].dims) == ['mission', 'obs']]

    tmp = ds[varnames].to_dataframe()

    if 'mission_obs' in tmp:
        tmp = tmp.reset_index('obs', drop=True)
    else:
        tmp = tmp.reset_index('obs').rename(index=str,
                                            columns={'obs': 'mission_obs'})
    #tmp = tmp.rename(index=str, columns={'profile_id': 'profile'}).set_index('profile', append=True)
    tmp['profile'] = tmp.profile_id
    tmp = tmp.set_index('profile', append=True)
    tmp['obs'] = tmp.groupby('profile')['mission_obs'].apply(lambda x: x-x.min())
    tmp = tmp.set_index('obs', append=True)
    tmp = tmp.to_xarray()

    ds = ds.rename({'obs': 'tmp'})
    ds['obs'] = tmp['obs']
    for c in tmp.coords:
        if c not in ds:
            ds[c] = tmp[c]
    for v in tmp:
        ds[v] = tmp[v]

    ds = ds.drop(['profile_id', 'tmp'])

    return ds


def convert_missionObs2diveObs(ds):
    """ Add new dimension groupping observations per dive

        For example, temp(mission, obs) is reorganized into
          temp(mission, dive, obs)
    """
    assert 'dive_id' in ds

    varnames = [v for v in ds if sorted(ds[v].dims) == ['mission', 'obs']]
    tmp = ds[varnames].to_dataframe()

    if 'mission_obs' in tmp:
        tmp = tmp.reset_index('obs', drop=True)
    else:
        tmp = tmp.reset_index('obs').rename(index=str,
                                            columns={'obs': 'mission_obs'})

    tmp['dive'] = tmp.dive_id
    tmp = tmp.set_index('dive', append=True)
    if 'dive_obs' in tmp:
        tmp['obs'] = tmp.dive_obs
    else:
        tmp['obs'] = tmp.groupby('dive')['mission_obs'].apply(lambda x: x-x.min())
    tmp = tmp.set_index('obs', append=True)
    tmp = tmp.to_xarray()

    ds = ds.rename({'obs': 'tmp'})
    ds['obs'] = tmp['obs']
    for c in tmp.coords:
        if c not in ds:
            ds[c] = tmp[c]
    for v in tmp:
        ds[v] = tmp[v]

    ds = ds.drop(['dive_id', 'tmp'])

    return ds


def convert_profileObs2missionObs(ds):
    varnames = [v for v in ds.variables if 'mission' in ds[v].dims]
    varnames = [v for v in varnames if 'profile' in ds[v].dims]
    varnames = [v for v in varnames if 'obs' in ds[v].dims]

    tmp = ds.reset_coords()[varnames].to_dataframe().dropna(how='all')

    if 'profile_id' in tmp:
        tmp = tmp.reset_index('profile', drop=True)
    else:
        tmp = tmp.reset_index('profile').rename(columns={'profile': 'profile_id'})
    if 'profile_obs' in tmp:
        tmp = tmp.reset_index('obs', drop=True)
    else:
        tmp = tmp.reset_index('obs').rename(columns={'obs': 'profile_obs'})
    if 'mission_obs' in tmp:
        tmp['obs'] = tmp.mission_obs.astype(int)
        tmp = tmp.sort_values('obs').set_index('obs', append=True)
        tmp = tmp.drop('mission_obs', axis='columns')
        ds = ds.drop('mission_obs')
    else:
        assert False

    tmp.sort_index(inplace=True)
    tmp = tmp.to_xarray()
    ds = ds.rename({'obs': 'tmp'})

    for c in tmp.coords:
        if c not in ds:
            ds[c] = tmp[c]
    for v in tmp:
        ds[v] = tmp[v]

    assert [v for v in ds.variables if 'tmp' in ds[v].dims] == ['tmp']
    ds = ds.drop('tmp')

    return ds


def alongtrack_distance(ds, units: Optional[str] = 'km'):
    """From Lat/Lon estimate horizontal distance along the path

    Parameters
    ----------
    ds: Dataset
    units: string, optional
           Units to use for distance (km or m)
    """
    assert units in ('km', 'm'), "Valid units are: km, m."

    assert ('lat' in ds), "Missing lat in dataset"
    assert ('lon' in ds), "Missing lon in dataset"
    assert ds.lat.dims == ds.lon.dims, "Mismatching dimensions: lat/lon"
    assert len(ds.lat.dims) == 1

    dims = ds.lat.dims

    ds = ds.copy(deep=True)

    g = Geod(ellps='WGS84')
    L = g.inv(ds.lon.values[:-1], ds.lat.values[:-1],
              ds.lon.values[1:], ds.lat.values[1:])[2]
    L = np.append([0], L.cumsum())
    if units == 'km':
        L *= 1e-3
    ds['distance'] = xr.DataArray(L, dims=dims)

    ds.distance.attrs['name'] = 'distance'
    ds.distance.attrs['long_name'] = 'Distance along path'
    ds.distance.attrs['units'] = units

    ds = ds.set_coords('distance')

    return ds


def get_bathymetry_OceansDB(ds):
    """Define bathymetry for each profile lat x lon from etopo
    """
    if 'trajectory' in ds.dims:
        return ds.groupby('trajectory').map(get_bathymetry_OceansDB)

    ETOPO = oceansdb.ETOPO(resolution='1min')
    bathymetry = -ETOPO['topography'].track(lat=ds.profile_lat, lon=ds.profile_lon)['height']

    bathymetry = xr.DataArray(bathymetry, dims=['profile'])

    bathymetry.attrs["description"] = "Sea floor depth."
    bathymetry.attrs["source"] = "ETOPO 1min via OceansDB."

    return bathymetry


def trajectory_chronometer(da):
    """Running time of a deployment, i.e. seconds since the start

    Notes
    -----
    - For now I'll assume that profile_time is the best reference.
    """
    if 'trajectory' in da.dims:
        return da.groupby('trajectory').map(trajectory_chronometer)

    t = (da.profile_time - da.profile_time.min()) / (np.timedelta64(1, 's'))
    t.attrs['units'] = 'seconds'
    t.attrs['description'] = 'Seconds since the start of the trajectory'

    return t.reset_coords(drop=True)


def fit_fchl_exponential_decay(df):
    """
    Returns
    -------
    sigma : float
        Standard deviation of the residue
    """
    df = df.dropna(how='any')
    df = df.sort_values(by='profile')

    t = np.array(df.running_time) / 86400.
    fchl = np.array(df.fchl_deep_ref)

    def fun(x, t, y):
        return x[0] + x[1] * np.exp( - t / x[2]) - y

    # fchl0, fchl_scale, t_scale
    x0 = [np.percentile(fchl, 3), 0.1, 14.]
    res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.001, args=(t, fchl))
    assert res_robust.success

    output = {
            "fchl0": res_robust.x[0],
            "fchl_scale": res_robust.x[1],
            "t_scale": res_robust.x[2] * 86400.,
            "sigma": fun(res_robust.x, t, fchl).std()
            }
    return output


def fchl_dark_offset_castelao2020(da, threshold=0.01, decay=False):
    """Dark count correction as Castelao and Rudnick 2020

    Assumes that deep chlorophyll fluorescence measurements should tend to zero
    in the deep layers. Here it is imposed measurements at least up to 200m.

    Returns
    -------
    fchl_unbiased
    coef
    fchl_deep_ref

    Notes
    -----
    - Chlorophyll fluorescence is a relatively noisy measurement, so smooth it
      before defining the dark count. The unbias output is based on the actual
      input, thus the output is not smoothed.
    """
    if 'trajectory' in da.dims:
        return da.groupby('trajectory').map(fchl_dark_offset_castelao2020, threshold=threshold, decay=decay)
    da = da.dropna(dim='profile', how='all').dropna(dim='obs', how='all')

    bathymetry = get_bathymetry_OceansDB(da)
    profile_max_depth = da.depth.max(dim='obs')
    deep_offshore_flag = (bathymetry >= 450) & (profile_max_depth >= 200)

    # Remove spikes
    fchl_deep = da.groupby('profile').map(lambda x: x.dropna(dim='obs').rolling(obs=5, center=True).median().dropna(dim='obs').rolling(obs=3, center=True).mean().dropna(dim='obs'))
    # zlimit = ds.depth.quantile(0.99)
    # idx = (ds.depth >= 120) & (ds.depth <= zlimit)
    fchl_deep = fchl_deep.where(deep_offshore_flag, np.nan).where(fchl_deep.depth >= 80, np.nan)
    fchl_deep = fchl_deep.dropna(dim='profile', how='all').dropna(dim='obs', how='all')
    fchl_deep_ref = fchl_deep.quantile(0.05, dim='obs')

    description = "Using robust minimum fl in profile as deep reference."

    if decay is True:
        tmp = xr.Dataset()
        tmp['running_time'] = trajectory_chronometer(da.profile_time)
        tmp['fchl_deep_ref'] = fchl_deep_ref
        df = tmp.reset_coords()[['profile', 'fchl_deep_ref', 'running_time']].to_dataframe()
        coef = fit_fchl_exponential_decay(df)
        # Time dependent component only
        fchl_decay = coef['fchl_scale'] * np.exp(- tmp.running_time / coef['t_scale']).rename('fchl_decay')
        fchl_decay = fchl_decay.reset_coords(drop=True)
        # Remove the time dependent component from the deep layer data
        fchl_deep = fchl_deep - fchl_decay
        # Dark count: a percentile from time independent deeplayer.
        fchl_dark = fchl_deep.quantile(threshold).rename({'quantile': 'fchl_dark_quantile'})
        coef["fchl_dark"] = float(fchl_dark)
        # A dark count time dependent, thus a 1D array.
        fchl_offset = fchl_decay + fchl_dark

        description = "Using robust minimum fl in profile as deep reference."

    else:
        # A dark count constant, thus a scalar.
        fchl_dark = fchl_deep.quantile(threshold)
        coef = {"fchl_dark": float(fchl_dark)}
        fchl_offset = fchl_dark

    fchl_unbiased = (da - fchl_offset)
    fchl_unbiased = fchl_unbiased.where((fchl_unbiased >= 0) | (fchl_unbiased.isnull()), 0)
    fchl_unbiased.attrs["long_name"] = "Chlorophyll Fluorescence (unbiased)"
    output = fchl_unbiased.rename('fchl').to_dataset()

    output["fchl_deep_ref"] = fchl_deep_ref.reset_coords(drop=True)

    for k in coef:
        output[k] = coef[k]

    return output


def zeu_from_chl_morel2001_profile(da, smooth=True):
    """Estimates Z_eu, Z_pd, and weigthed averaged Chl

       From Morel 2001, integrate the total Chl from the surface towards the
       bottom and estimate the depth of the euphotic zone. From that, defines
       the first penetration depth (z_pd).
       The visible chlorophyll is obtained from an weighted average considering
       the layer thickness that a measurement is representative and the light
       assuming an exponential decay scaled by z_pd.

    Notes
    -----
    - Consider to order by z than expect one single direction. In the case
      of isotrack, it will show some cycles, and in that case it is probably
      better to order by z and use thinner layers.
    """
    """For one profile estimates z_pd and mean chl in that layer

       From Morel 2001, integrate the total Chl from the surface towards
         the bottom and estimate the depth of the euphotic zone. From that,
         estimates the first penetration depth (z_pd), and the average chl
         from the surface up to z_pd.

       Consider to order by z than expect one single direction. In the case
       of isotrack, it will show some cycles, and in that case it is probably
       better to order by z and use thinner layers.
    """
    """Estimate the Euphotic Depth from the Chl profile

       Using Morel 2001, integrates Chl from the surface until the z_eu

       Alternative, but it didn't work for me.
       chl_log = np.log10(chl_tot)
       10**(2.1236 + 0.932468*chl_log -1.4264*chl_log**2 +0.52776 * chl_log**3 -0.07617 * chl_log**4)-z
    """
    assert da.dims == ('obs',), "For now it is expected single dimension obs"
    assert 'depth' in da.coords, "Missing depth coordinate"
    if smooth:
        da = da.isel(obs=~np.isnan(da.depth))
        da = da.rolling(obs=5, min_periods=3, center=True).median(keep_attrs=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            da = da.rolling(obs=3, center=True, min_periods=2).mean(keep_attrs=True)
    # Remove nan.
    # Data flagged bad should be NaN at this point, but a valid depth should
    # indicate that there was a measurement at that level
    da = da.dropna(dim='obs')
    da = da.sortby(da.depth)

    z = da.depth.data
    chl = da.data

    # ====
    # Some profile oscillate near the surface. If that happens, remove any
    #   measurement that didn't move up or all sucessive that went down.
    #assert np.median(np.diff(z)) < 0, "Profile top-down"
    #if (np.diff(z) >= 0).any():
    #    zmax = z[0]
    #    idx = [0]
    #    for i, level in enumerate(z):
    #        if level < zmax:
    #            zmax = level
    #            idx.append(i)
    #    z = z[idx]
    #    chl = chl[idx]
    # ====

    # Expected to be ordered surface to bottom, otherwise, flip it.
    #if np.median(np.diff(z)) < 0:
    #    z = z[::-1]
    #    chl = chl[::-1]

    # The top layer is usually thicker since we don't measure all the way up
    # to the surface. Let's create some robustness by replacing the first value
    # by the median of the 3 top ones.
    # idx = np.argsort(z)[:3]
    # chl[idx[0]] = np.median(chl[idx])

    # tmp = np.ones((chl.size,3), dtype=chl.dtype) * np.nan
    # tmp[:-1, 0] = chl[1:]
    # tmp[:, 1] = chl[:]
    # tmp[1:,2] = chl[:-1]
    # Don't replace the top measurement defined by the median previously
    # chl[1:] = np.nanmedian(tmp, axis=1)[1:]

    # Define layers
    dz = np.diff(z)
    bottom = np.append(z[:-1] + dz/2., z[-1])
    top = np.append([0], z[1:] - dz/2.)
    thickness = bottom - top

    chl_cumsum = np.cumsum(chl * thickness)
    max_penetration = 912.5 * chl_cumsum ** (-.839)

    idx = np.nonzero((max_penetration - z) < 0)[0]
    assert idx.size > 0, "Profile too shallow. penetration: {}, z:{})".format(
            min(max_penetration), max(z))

    idx = np.min(idx) + 2
    z_eu = np.interp(0, z[:idx] - max_penetration[:idx], z[:idx])

    if z_eu > 102:
        max_penetration = 426.3 * chl_cumsum ** (-.547)
        idx = np.nonzero((max_penetration - z) < 0)[0]
        assert idx.size > 0, "Profile stops before ze"
        idx = np.min(idx) + 2
        z_eu = np.interp(0, z[:idx] - max_penetration[:idx], z[:idx])

    assert z_eu < 180, "Estimated z_eu deeper than 180m"

    z_pd = z_eu / -np.log(0.01)
    w = z_pd * (np.exp(-2 * top / z_pd) - np.exp(-2 * bottom / z_pd))
    chl_vis = (chl * w).sum() / w.sum()
    # weight = np.exp(-(2*z)/z_pd) * thickness
    # chl_vis = (chl * weight).sum() / weight.sum()
    # chl_vis = np.sum(chl * np.exp(-(2*z)/z_pd) * thickness)
    # z_pd = - zeu_from_morel2001(z, chl) / np.log(0.01)
    # chl * thickness => Gives absolute chl per layer
    # chl_cumsum => Gives the total chl up to the bottom of the considered layer
    # chl_pd => Mean Chl in the first penetration layer
    chl_pd = np.interp(z_pd, bottom, chl_cumsum) / z_pd

    return {"z_eu": z_eu, "z_pd": z_pd, "chl_pd": chl_pd, "chl_vis": chl_vis}

    output = xr.Dataset()
    output['z_eu'] = z_eu
    output['z_pd'] = z_pd
    output['chl_pd'] = chl_pd
    output['chl_vis'] = chl_vis

    return output




    assert da.dims == ('obs',)
    assert hasattr(da, 'depth')

    da = da.dropna(dim='obs')
    da = da.sortby('depth', ascending=True)
    dz = np.diff(da.depth)
    thickness = np.ones_like(da.depth) * np.nan
    thickness[1:-1] = (z[2:] - z[:-2])/2.
    thickness[0] = (z[:2]).mean()

    bottom = np.append(z[:-1] + dz/2., z[-1])
    top = np.append([0], z[1:] - dz/2.)
    thickness = bottom - top


def chl_vis_morel2001(da):
    """Chl average up to z_pd according to Morel 2001

       Average of Chlorophyll in the first penetration depth layer according
         to Morel 2001.

    Note
    ----
    - Would be here the place to impose data near the surface? The motivation
      was to prevent some profiles flagged bad near the surface to be used.
    """
    assert isinstance(da, xr.DataArray)

    if 'trajectory' in da.dims:
        return da.groupby('trajectory').map(chl_vis_morel2001)

    output = pd.DataFrame()
    for pn, p in da.groupby('profile', restore_coord_dims=True):
        p = p.dropna(dim='obs', how='any')
        try:
            # tmp = chl_vis_morel2001_profile(z=p.depth.data, chl=p.data)
            tmp = zeu_from_chl_morel2001_profile(p)
            # chl_10 = chl_10m(z=p.depth.data, chl=p.data)
            output = output.append({'profile': pn,
                                    'z_eu': tmp['z_eu'],
                                    'z_pd': tmp['z_pd'],
                                    'chl_pd': tmp['chl_pd'],
                                    # 'chl_10': chl_10,
                                    'chl_vis': tmp['chl_vis']},
                                   ignore_index=True)
        except Exception as e:
            print("Failed on profile {}: {}".format(pn, e))

    output['profile'] = output.profile.astype('i')
    output = output.set_index('profile').\
                    to_xarray().\
                    expand_dims('ze_method').\
                    assign_coords(ze_method=['morel2001'])

    return output


def tukey53H(x):
    """Spike test Tukey 53H from Goring & Nikora 2002
    """
    N = len(x)

    u1 = ma.masked_all(N)
    for n in range(N - 4):
        if x[n : n + 5].any():
            u1[n + 2] = ma.median(x[n : n + 5])

    u2 = ma.masked_all(N)
    for n in range(N - 2):
        if u1[n : n + 3].any():
            u2[n + 1] = ma.median(u1[n : n + 3])

    u3 = ma.masked_all(N)
    u3[1:-1] = 0.25 * (u2[:-2] + 2 * u2[1:-1] + u2[2:])

    u3[u3.mask] = np.nan
    return u3.data


def convert_missionObs2TrajectoryProfile(ds):
    """ Add new dimension groupping observations per profile

        For example, temp(mission, obs) is reorganized into
          temp(mission, profile, obs)
    """
    assert 'profile_id' in ds
    varnames = [v for v in ds if sorted(ds[v].dims) == ['mission', 'obs']]

    tmp = ds[varnames].to_dataframe()

    if 'mission_obs' in tmp:
        tmp = tmp.reset_index('obs', drop=True)
    else:
        tmp = tmp.reset_index('obs').rename(index=str,
                                            columns={'obs': 'mission_obs'})
    tmp = tmp.rename(index=str, columns={'profile_id': 'profile'}).\
              set_index('profile', append=True)
    #tmp = tmp.rename(index=str, columns={'profile_id': 'profile'}).set_index('profile', append=True)
    # tmp['profile'] = tmp.profile_id
    # tmp = tmp.set_index('profile', append=True)
    # tmp['obs'] = tmp.groupby('profile')['mission_obs'].apply(lambda x: x-x.min())
    tmp['obs'] = tmp.groupby(['mission', 'profile'])['mission_obs'].\
                     rank().astype('i')
    tmp = tmp.set_index('obs', append=True)
    tmp = tmp.to_xarray()

    ds = ds.rename({'obs': 'tmp'})
    ds['obs'] = tmp['obs']
    for c in tmp.coords:
        if c not in ds:
            ds[c] = tmp[c]
    for v in tmp:
        ds[v] = tmp[v]

    ds = ds.set_coords('mission_obs')
    ds = ds.drop(['profile_id', 'tmp'])

    return ds

