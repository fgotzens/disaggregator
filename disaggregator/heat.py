# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:39:43 2022

@author: Paul Verwiebe
"""
# %% Imports
from .spatial import disagg_applications
from .config import (data_in, data_out, get_config, dict_region_code,
                     hist_weather_year, bl_dict)
from .data import (database_shapes, ambient_T, CTS_power_slp_generator,
                   households_per_size, t_allo)
from .temporal import (disagg_temporal_gas_CTS_water, disagg_temporal_gas_CTS, 
                       disagg_temporal_gas_households,
                       disagg_temporal_power_CTS,
                       disagg_temporal_gas_CTS_by_state,
                       disagg_temporal_gas_CTS_water_by_state)

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset, num2date
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)

# %% Functions


def create_hp_load(detailed=False, p_ground=0.36, p_air=0.58,
                   p_water=0.06, use_nuts3code=False, state=None, **kwargs):
    """
    Creates normalized electrical load profiles for heat pumps per NUTS-3

    Parameters
    -------
    detailed : bool
        Throughput
    p_ground, p_air, p_water : float, default 0.36, 0.58, 0.06
        percentage of ground/air/water heat pumps sum must be 1
    use_nuts3code : bool
        Throughput for spatial disaggregation functions
        If True use NUTS-3 codes as region identifiers.
    state : str, default None
        Specifies state. Only needed if detailed=True. Must by one of the
        entries of bl_dict().values(),
        ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']

    Returns
    -------
    None.

    """
    if p_ground + p_air + p_water != 1:
        raise ValueError("sum of percentage of ground/air/water heat pumps"
                         " must be 1")
    # get base year
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    # creat path for reading existing heat load profiles
    if not detailed:
        path = data_out('CTS_heat_norm_' + str(year) + '.csv')
    else:
        path = data_out('CTS_heat_norm_' + str(state) + '_'
                        + str(year) + '.csv')

    # get normalised heat load profiles
    if os.path.exists(path) and not detailed:
        logger.info('Reading existing heat norm timeseries for detailed: ' +
                    str(detailed))
        heat_norm = pd.read_csv(path, index_col=0, header=[0])
        heat_norm.index = pd.to_datetime(heat_norm.index)
        heat_norm.columns = heat_norm.columns.astype(int)
    if os.path.exists(path) and detailed:
        logger.info('Reading existing heat norm timeseries for detailed: ' +
                    str(detailed))
        heat_norm = pd.read_csv(path, index_col=0, header=[0, 1])
        heat_norm.index = pd.to_datetime(heat_norm.index)
    else:
        logger.info('Creating heat norm timeseries for detailed: ' +
                    str(detailed))
        heat_norm, gas_total, gas_tempinde = (create_heat_norm(
            detailed=detailed, year=year, state=state))

    # Creating COP timeseries
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(year=year)
    # columns have to be cast to str, for matching with multicolumns in
    # heat_norm DataFrame
    if detailed:
        air_floor_cop.columns = air_floor_cop.columns.astype(str)
        ground_floor_cop.columns = ground_floor_cop.columns.astype(str)
        water_floor_cop.columns = water_floor_cop.columns.astype(str)

    # compute electricity consumption timeseries for heat pumps with COP
    # respecting shares of HP technologies. el = heat/cop
    ec_heat = (p_ground * heat_norm.div(ground_floor_cop, level=0)
               + p_air * heat_norm.div(air_floor_cop, level=0)
               + p_water * heat_norm.div(water_floor_cop, level=0))
    # normalize resulting electricity consuption
    ec_heat = ec_heat.divide(ec_heat.sum(axis=0), axis=1)
    # rename columns if nuts-3 codes should be used
    if use_nuts3code:
        ec_heat = ec_heat.rename(columns=dict_region_code(
            level='lk', keys='ags_lk', values='natcode_nuts3'),
            level=(0 if detailed else None))
    return ec_heat


def create_heat_norm(detailed=False, state=None, **kwargs):
    """
    Creates normalised heat demand timeseries for CTS per NUTS-3, and branch
    if detailed is set to True.

    Parameters
    ----------
    detailed : bool, default False
        If True heat demand per branch and disctrict is calculated.
        Otherwise just the heat demand per district.
    state : str, default None
        Specifies state. Only needed if detailed=True. Must by one of the
        entries of bl_dict().values(),
        ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']

    Returns
    -------
    heat_norm : pd.DataFrame
        normalised heat demand
        index = datetimeindex
        columns = Districts / (Branches)
    gas_total : pd.DataFrame
        total gas consumption
        index = datetimeindex
        columns = Districts / (Branches)
    gas_tempinde : pd.DataFrame
        gas consumption for temoeratureindependent applications
        (hot water, process heat, mechanical energy for CTS)
        index = datetimeindex
        columns = Districts / (Branches)
    """
    # get base year
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    # create timeseries total gas consumption
    path = data_out('CTS_gas_total_' + str(year) + '.csv')
    if os.path.exists(path) and not detailed:
        logger.info('Reading existing total gas temporal disaggregated'
                    ' timeseries.')
        gas_total = pd.read_csv(path, index_col=0)
        gas_total.index = pd.to_datetime(gas_total.index)
        gas_total.columns = gas_total.columns.astype(int)
    elif not detailed:
        logger.info('Disaggregating total gas consumption')
        gas_total = disagg_temporal_gas_CTS(detailed=detailed,
                                            use_nuts3code=False)
        gas_total.to_csv(path)  # TBD Moved to function disagg_temporal gas
        gas_total.columns = gas_total.columns.astype(int)
    else:
        logger.info('Disaggregating detailed total gas consumption for state: '
                    + str(state))
        gas_total = disagg_temporal_gas_CTS_by_state(detailed=detailed,
                                                     use_nuts3code=False,
                                                     state=state, year=year)
    # create timeseries for temperature independent gas consumption
    path = data_out('CTS_gas_tempinde_' + str(year) + '.csv')
    if os.path.exists(path) and not detailed:
        logger.info('Reading existing temperature independent gas temporal '
                    'disaggregated timeseries.')
        gas_tempinde = pd.read_csv(path, index_col=0)
        gas_tempinde.index = pd.to_datetime(gas_tempinde.index)
        gas_tempinde.columns = gas_tempinde.columns.astype(int)
    elif not detailed:
        logger.info('Disaggrating temperature independent gas consumption')
        gas_tempinde = disagg_temporal_gas_CTS_water(detailed=detailed,
                                                     use_nuts3code=False)
        gas_tempinde.to_csv(path)  # TBD Moved to function disagg_temporal gas
        gas_tempinde.columns = gas_tempinde.columns.astype(int)
    else:
        logger.info('Disaggregating detailed temperature independent gas'
                    ' consumption for state: ' + str(state))
        gas_tempinde = (disagg_temporal_gas_CTS_water_by_state(
                                    detailed=detailed, use_nuts3code=False,
                                    state=state, year=year))

    # create space heating timeseries: difference between total heat demand
    # and water heating demand
    heat_norm = (gas_total - gas_tempinde).clip(lower=0)

    temp_allo = resample_t_allo()
    # clip heat demand above heating threshold
    if detailed:
        df = (heat_norm
              .droplevel([1], axis=1)[temp_allo[temp_allo > 13].isnull()]
              .fillna(0))
        df.columns = heat_norm.columns
        heat_norm = df.copy()
    else:
        # set heat demand to 0 if temp_allo is higher then 13°C
        heat_norm = heat_norm[temp_allo[temp_allo > 13].isnull()].fillna(0)

    # normalise
    heat_norm = heat_norm.divide(heat_norm.sum(axis=0), axis=1)

    if not detailed:
        # safe as csv
        heat_norm.to_csv(data_out('CTS_heat_norm_' + str(year) + '.csv'))
    else:
        heat_norm.to_csv(data_out('CTS_heat_norm_' + str(state) + '_' +
                                  str(year) + '.csv'))

    return heat_norm, gas_total, gas_tempinde


def cop_ts(**kwargs):
    """
    Creates COP timeseries for ground/air/water heat pumps with floor heating.

    Returns
    -------
    air_floor_cop : pd.DataFrame
    index = datetimeindex
        columns = Districts
    ground_floor_cop : pd.DataFrame
        index = datetimeindex
        columns = Districts
    water_floor_cop : pd.DataFrame
        index = datetimeindex
        columns = Districts
    """
    # get base year (if a future year, assing historical weather year)
    cfg = kwargs.get('cfg', get_config())
    base_year = kwargs.get('year', cfg['base_year'])
    year = hist_weather_year().get(base_year)
    while year > 2018:
        year = hist_weather_year().get(year)

    # leap years
    if ((year % 4 == 0)
            & (year % 100 != 0)
            | (year % 4 == 0)
            & (year % 100 == 0)
            & (year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040

    date = (pd.date_range((str(base_year) + '-01-01'),
                          periods=periods / 4,
                          freq='H'))

    # get source temperatures
    soil_t = soil_temp(year)
    soil_t.index = date

    # heat loss between soil and brine
    ground_t = soil_t.sub(5)

    # get air temperature
    air_t = ambient_T(year=year)
    air_t = change_nuts3_to_ags(air_t)
    air_t.index = date

    # create water temperatur DataFrame (constant temperature 0f 10°C,
    # heat loss between water and brine)
    water_t = pd.DataFrame(index=ground_t.index,
                           columns=ground_t.columns,
                           data=10 - 5)

    # sink temperature of 40°C (floor heatin)
    floor_t = pd.DataFrame(index=ground_t.index,
                           columns=ground_t.columns,
                           data=40)

    # create cop timeseries based on temperature difference between source
    # and sink
    air_floor_cop = cop_curve((floor_t - air_t), 'air')
    ground_floor_cop = cop_curve((floor_t - ground_t), 'ground')
    water_floor_cop = cop_curve((floor_t - water_t), 'water')

    return air_floor_cop, ground_floor_cop, water_floor_cop


# %% Utility functions


def resample_t_allo(**kwargs):
    """
    Resamples the allocation temperature to match other dataframes in plot
    """
    # get base year
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])

    # change index to datetime
    if ((year % 4 == 0)
            & (year % 100 != 0)
            | (year % 4 == 0)
            & (year % 100 == 0)
            & (year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040
    date = pd.date_range((str(year) + '-01-01'), periods=periods / 4, freq='H')
    df = t_allo()
    df.index = pd.to_datetime(df.index)
    last_hour = df.copy()[-1:]
    last_hour.index = last_hour.index + timedelta(1)
    df = df.append(last_hour)

    # resample
    df = df.resample('H').pad()
    df = df[:-1]
    df.index = date
    df.columns = df.columns.astype(int)

    return df


def change_nuts3_to_ags(df):
    """
    changes colums of given df from nuts3 region code to ags
    """
    dict_nuts3_name = dict_region_code(keys='natcode_nuts3', values='ags_lk')

    for nuts3 in dict_nuts3_name.keys():

        # DE915 und DE919 not in ambient_t
        if nuts3 in df.columns:
            df.rename(columns={nuts3: dict_nuts3_name[nuts3]}, inplace=True)

    df = df.reindex(sorted(df.columns), axis=1)
    df.columns.name = None

    return df


# Following functions for heat demand and COP are completly or partly taken
# from https://github.com/oruhnau/when2heat (Ruhnau et al.)
def soil_temp(year):
    """
    Reads and processes soil temperature timeseries

    Parameters
    ----------
    year : int

    Returns
    -------
    pd.DataFrame
        index = datetimeindex
        columns = Districts
    """
    input_path = data_in('temporal', 'ERA_temperature_' + str(year) + '.nc')

    # -----------------------------------------------
    # from read.weather (When2Heat)

    # read weather nc
    nc = Dataset(input_path, only_use_cftime_datetimes=False,
                 only_use_python_datetimes=True)
    time = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    latitude = nc.variables['latitude'][:]
    longitude = nc.variables['longitude'][:]
    variable = nc.variables['stl4'][:]

    # Transform to pd.DataFrame
    df = pd.DataFrame(data=variable.reshape(len(time),
                                            len(latitude) * len(longitude)),
                      index=pd.Index(num2date(time, time_units), name='time'),
                      columns=pd.MultiIndex.from_product([latitude, longitude],
                                                         names=('latitude',
                                                                'longitude')))
    df.index = pd.to_datetime(df.index.astype(str))
    # ------------------------------------------------
    # upsample from 6h resolution
    df = upsample_df(df, '60min')

    # ------------------------------------------------
    # get LK and representative coords
    DF = database_shapes()
    # Derive lat/lon tuple as representative point for each shape
    DF['coords'] = DF.geometry.apply(
        lambda x: x.representative_point().coords[:][0])
    DF['coords_WGS84'] = DF.to_crs({'init': 'epsg:4326'}).geometry.apply(
        lambda x: x.representative_point().coords[:][0])

    # ------------------------------------------------

    DF['ags_lk'] = DF.id_ags.floordiv(1000)

    # round to Era 0.75 x 0.75° grid
    def round_grid(x):
        return round(x * (4 / 3)) / (4 / 3)

    # create dataframe with soil temperature timeseries per LK
    soil_t = pd.DataFrame(index=df.index)
    soil_t.index.name = None

    # find lk representative coords in era5 soil temp
    for coords in DF['coords_WGS84']:
        lat = round_grid(coords[1])
        lon = round_grid(coords[0])
        ags_lk = DF.loc[DF['coords_WGS84'] == coords, 'ags_lk'].values[0]
        soil_t[ags_lk] = df.loc[:, lat].loc[:, lon]

    # from Kelvin to Celsius
    soil_t = soil_t.sub(275.15)

    # sort columns
    soil_t = soil_t.reindex(sorted(soil_t.columns), axis=1)

    return soil_t


def cop_curve(delta_t, source_type):
    """
    Creates cop timeseries based on temperature difference between
    source and sink

    Parameters
    ----------
    delta_t : pd.DataFrame
        index = datetimeindex
        columns = District
    source_type : str
        must be in ['ground', 'air', 'water']

    Returns
    -------
    pd.DataFrame
        index = datetimeindex
        columns = Districts
    """

    cop_params = (pd.read_csv(data_in('dimensionless', 'cop_parameters.csv'),
                              sep=';', decimal=',', header=0, index_col=0)
                  .apply(pd.to_numeric, downcast='float'))
    delta_t.clip(lower=13, inplace=True)
    return sum(cop_params.loc[i, source_type] * delta_t ** i for i in range(3))


def upsample_df(df, resolution):  # from misc.upsample_df (When2Heat)
    """
    Resamples DataFrame to given resolution

    Parameters
    ----------
    df : pd.DataFrame
    resolution : str

    Returns
    -------
    pd.DataFrame
    """
    # The low-resolution values are applied to all high-resolution values up
    # to the next low-resolution value
    # In particular, the last low-resolution value is extended up to
    # where the next low-resolution value would be

    df = df.copy()

    # Determine the original frequency
    freq = df.index[-1] - df.index[-2]

    # Temporally append the DataFrame by one low-resolution value
    df.loc[df.index[-1] + freq, :] = df.iloc[-1, :]

    # Up-sample
    df = df.resample(resolution).pad()

    # Drop the temporal low-resolution value
    df.drop(df.index[-1], inplace=True)

    return df
