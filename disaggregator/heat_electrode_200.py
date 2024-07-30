# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:39:43 2022

@author: Paul Verwiebe
"""
# %% Imports
from .spatial import (disagg_applications, disagg_applications_eff)
from .config import (data_in, data_out, get_config, dict_region_code,
                     hist_weather_year, bl_dict, shift_profile_industry,
                     get_efficiency_level)
from .data import (database_shapes, ambient_T, t_allo,
                   shift_load_profile_generator,
                   gas_slp_weekday_params,
                   h_value, h_value_water)
from .temporal import (disagg_temporal_gas_CTS_water, disagg_temporal_gas_CTS,
                       disagg_temporal_power_CTS,
                       disagg_temporal_gas_CTS_by_state,
                       disagg_temporal_gas_CTS_water_by_state,
                       disagg_temporal_industry_blp, disagg_temporal_industry,
                       disagg_temporal_power_CTS_blp,
                       disagg_temporal_industry_blp_by_state,
                       disagg_temporal_industry_by_state,
                       disagg_temporal_power_CTS_blp_by_state,
                       disagg_temporal_power_CTS_by_state,
                       create_heat_norm_industry,
                       resample_t_allo,
                       disagg_temporal_applications)

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset, num2date
from datetime import timedelta, date
import logging
logger = logging.getLogger(__name__)

# %% Functions


def sector_fuel_switch_fom_gas(sector, switch_to, **kwargs):
    """
    Determines yearly gas demand per branch and NUTS-3 for heat applications
    that will be replaced by a different fuel in the future.
    Fuel is specified by parameter 'switch_to'.

    Parameters
    ----------
    sector : str
        must be one of ['CTS', 'industry']
    switch_to: str
        must be one of ['power', 'hydrogen']

    Returns
    -------
    pd.DataFrame

    """
    assert switch_to in ['power', 'hydrogen'], ("'switch_to' needs to be in "
                                                "['power', 'hydrogen'].")
    # get base year
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get('year', cfg['base_year'])
    # read input data, which contains reduction levels of gas usage for heat
    # applications
    fuel_switch = read_fuel_switch_share(sector, switch_to)
    # linear projection to the given year
    fuel_switch_projected = projection_fuel_switch_share(fuel_switch,
                                                         target_year=year)
    # get yearly gas demand per application
    df_app = disagg_applications_eff(source='gas', sector=sector,
                                     disagg_ph=True, no_self_gen=False,
                                     year=year)

    # create new DF for gas demand which will be replaced by different fuel
    df_gas_switch = pd.DataFrame(index=df_app.index,
                                 columns=df_app.columns,
                                 data=0)
    # for each branch and each application multiply gas demand by share of
    # fuel switch
    for branch in df_app.columns.unique(level=0):
        for app in fuel_switch_projected.columns.unique():
            df_gas_switch[branch, app] = (df_app[branch, app]
                                          * (fuel_switch_projected
                                             .loc[branch][app]))
    # drop all columns with only zeros
    df_gas_switch = df_gas_switch.loc[:, (df_gas_switch != 0).any(axis=0)]

    return df_gas_switch


def hydrogen_after_switch(df_gas_switch):
    """
    Determines hydrogen consumption to replace gas consumption.

    Returns
    -------
    pd.DataFrame() with regional hydrogen consumption per consumer group and
        application.

    """
    # define slice for easier DataFrame selection
    col = pd.IndexSlice

    # for non-energetic use of hydrogen:
    # conversion from natural gas to hydrogen in steam reforming has an
    # efficiency of about 70%
    df_hydro = df_gas_switch.copy()
    df_hydro.loc[:, col[:, 'Nichtenergetische Nutzung']] = (
        df_hydro.loc[:, col[:, 'Nichtenergetische Nutzung']]
        * (get_efficiency_level('Nichtenergetische Nutzung')))

    # for energetic use of hydrogen:
    # process heat applications are assumed to thave the same energy conversion
    # efficiency for natural gas and hydrogen

    return df_hydro


def disagg_temporal_fuel_switch_simple_cts(switch_to, detailed=False,
                                           state=None, use_nuts3code=False,
                                           **kwargs):
    """
    Creates gas demand time series for a given future year per application
	to be switched to power or hydrogen.

    Returns
    -------
    DataFrame with gas demand time series per branch and application if 
	detailed==False or per branch, application and region if detailed==True.

    """
    # check if a year was specified
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get("year", cfg["base_year"])
    assert switch_to in ['power', 'hydrogen'], ("'switch_to' needs to be in "
                                                "['power', 'hydrogen'].")
    # define slice for easier DataFrame selection
    col = pd.IndexSlice

    # read input data, which contains reduction levels of gas usage for heat
    # applications
    fuel_switch = read_fuel_switch_share(sector='CTS', switch_to=switch_to)
    # linear projection to the given year
    fuel_switch_projected = projection_fuel_switch_share(fuel_switch,
                                                         target_year=year)

    if not detailed:
        # get temporally disaggregated demand per application
        df_temp_apps = disagg_temporal_applications(source='gas', sector='CTS',
                                                    detailed=detailed,
                                                    state=None,
                                                    use_nuts3code=use_nuts3code,
                                                    disagg_ph=False,
                                                    use_blp=False,
                                                    use_slp_for_sh=False,
                                                    year=year)
        new_df = (df_temp_apps.mul(fuel_switch_projected.mean(), level=1)
                  .dropna(axis=1))

        return new_df
    else:
        # get temporally disaggregated demand per application
        test = pd.DataFrame(dtype='float')
        if state is not None:
            assert state in list(bl_dict().values()), ("'state' needs to be in "
                                               "['SH', 'HH', 'NI', 'HB', "
                                               "'NW', 'HE', 'RP', 'BW', "
                                               "'BY', 'SL', 'BE', 'BB', "
                                               "'MV', 'SN', 'ST', 'TH']")
            logger.info("Working on state {}.".format(str(state)))
            df_temp_apps = disagg_temporal_applications(source='gas',
                                                        sector='CTS',
                                                        detailed=detailed,
                                                        state=state,
                                                        use_nuts3code=use_nuts3code,
                                                        disagg_ph=False,
                                                        use_blp=False,
                                                        use_slp_for_sh=False,
                                                        year=year)
            new_df = (df_temp_apps.mul(fuel_switch_projected.mean(), level=2)
                      .dropna(axis=1))
            test = pd.concat([new_df, test], axis=1)
            return test
        else:
            for state in bl_dict().values():
                logger.info("Working on state {}.".format(str(state)))
                df_temp_apps = disagg_temporal_applications(source='gas',
                                                            sector='CTS',
                                                            detailed=detailed,
                                                            state=state,
                                                            use_nuts3code=use_nuts3code,
                                                            disagg_ph=False,
                                                            use_blp=False,
                                                            use_slp_for_sh=False,
                                                            year=year)
                new_df = (df_temp_apps.mul(fuel_switch_projected.mean(),
                                           level=2)
                          .dropna(axis=1))
                test = pd.concat([new_df, test], axis=1)
        return test



def disagg_temporal_industry_fuel_switch(df_gas_switch,
                                         state=None, **kwargs):
    """
    Temporally disaggregates industry gas demand, which will be switched to
    electricity or hydrogen, by state.

    Parameters
    -------
    df_gas_switch : pd.DataFrame
        Gas demand by branch, application and NUTS-3 region which will be
        replaced.
    state : str, default None
        Specifies state. Must by one of the entries of bl_dict().values(),
        ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
    Returns
    -------
    pd.DataFrame() : timestamp as index, multicolumns with nuts-3, branch and
        applications. uses shift load profiles for temporal disaggregation
        of df_gas_switch.

    """
    assert state in list(bl_dict().values()), ("'state' needs to be in "
                                               "['SH', 'HH', 'NI', 'HB', "
                                               "'NW', 'HE', 'RP', 'BW', "
                                               "'BY', 'SL', 'BE', 'BB', "
                                               "'MV', 'SN', 'ST', 'TH']")
    assert isinstance(state, str), "'state' needs to be a string."
    # check if a year was specified
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get("year", cfg["base_year"])
    # troughput values for the helper function, used for industrial disagg
    low = kwargs.get("low", 0.5)

    # make a dict with nuts3 as keys (lk) and nuts1 as values (bl)
    nuts3_list = df_gas_switch.index
    nuts1_list = [bl_dict().get(int(i[: -3]))
                  for i in df_gas_switch.index.astype(str)]
    nuts3_nuts1_dict = dict(zip(nuts3_list, nuts1_list))

    # creat multicolumn from df_gas_switch columns and index
    # count how many different applications there are
    amount_application = len(df_gas_switch.columns.unique(level=1))
    # count how many different wz (industrial/commercial branches) there are
    amount_wz = len(df_gas_switch.columns.unique(level=0))
    # count how many different regions there are
    regions = [k for k, v in nuts3_nuts1_dict.items() if v == state]
    amount_regions = len(regions)

    # create multicolumn for result DataFrame()
    multi_lk = [elem for elem in regions
                for _ in range(amount_application * amount_wz)]
    multi_wz = [elem for elem in df_gas_switch.columns.unique(level=0)
                for _ in range(amount_application)] * amount_regions
    multi_app = (list(df_gas_switch.columns.unique(level=1))
                 * amount_regions
                 * amount_wz)
    tuples = list(zip(*[multi_lk, multi_wz, multi_app]))
    multicolumn = pd.MultiIndex.from_tuples(tuples, names=["LK", "WZ",
                                                           "Anwendungen"])
    # create index for year in 15 min timesteps
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    # create new df for temporal disaggregated heat demand
    new_df = pd.DataFrame(index=idx, columns=multicolumn, dtype='float')

    # get shift load profiles for given state
    sp_bl = shift_load_profile_generator(state, low, year=year)

    # get normalized timeseries for temperature dependent gas demand for
    # industrial indoor heating, approximated with cts indoor heat with gas SLP
    # 'KO'
    if 'Raumwärme' in df_gas_switch.columns.unique(level=1):
        heat_norm, gas_total, gas_tempinde_norm = create_heat_norm_industry(
            state=state, slp='KO', year=year)
        # upsample heat_norm to quarter hours and interpolate, then normalize
        heat_norm = (heat_norm.resample('15T').asfreq()
                      .interpolate(method='linear',
                                  limit_direction='forward', axis=0))
        # extend DataFrame by 3 more periods
        extension = pd.DataFrame(index=pd.date_range(heat_norm.index[-1:]
                                                      .values[0], periods=4,
                                                      freq='15T')[-3:],
                                  columns=heat_norm.columns)
        heat_norm = heat_norm.append(extension).fillna(method='ffill')
        # normalize
        heat_norm = heat_norm.divide(heat_norm.sum(), axis=1)
        assert heat_norm.index.equals(idx), "The time-indizes are not aligned"

    # start assigning disaggregated demands to columns of new_df by multiplying
    # shift_load_profiles with yearly demands per region, branch and app
    assert sp_bl.index.equals(idx), "The time-indizes are not aligned"
    # nuts-3 (lk) per state
    i = 1  # lk counter
    for lk in new_df.columns.get_level_values(0).unique():
        logger.info("Working on LK {}/{}."
                    .format(i, len(new_df.columns
                                   .get_level_values(0)
                                   .unique())))
        i += 1
        # b=1
        for branch in (df_gas_switch.loc[lk]
                       .index.get_level_values(0).unique()):
            for app in df_gas_switch.loc[lk][branch].index:
                if app == 'Raumwärme':
                    new_df[lk, branch, app] = (
                        (df_gas_switch.loc[lk][branch, app]) * (heat_norm[lk]))
                else:
                    new_df.loc[:][lk, branch, app] = ((df_gas_switch
                                                    .loc[lk][branch, app])
                                                   * sp_bl[shift_profile_industry()
                                                           [branch]])
    # drop all columns with only zeros
    new_df = new_df.loc[:, (new_df != 0).any(axis=0)]
    # drop all columns that have only nan values
    new_df = new_df.dropna(axis=1, how='all')

    return new_df


def disagg_temporal_cts_fuel_switch(df_gas_switch, state=None, **kwargs):
    """
    Temporally disaggregates CTS gas demand, which will be switched to
    electricity or hydrogen, by state.

    Parameters
    -------
    df_gas_switch : pd.DataFrame
        Gas demand by branch, application and NUTS-3 region which will be
        replaced.
    state : str, default None
        Specifies state. Must by one of the entries of bl_dict().values(),
        ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
    Returns
    -------
    pd.DataFrame() : timestamp as index, multicolumns with nuts-3, branch and
        applications. temperature dependent and independent profiles from gas
        SLP for temporal disaggregation of df_gas_switch.

    Returns
    -------
    None.

    """
    assert state in list(bl_dict().values()), ("'state' needs to be in "
                                               "['SH', 'HH', 'NI', 'HB', "
                                               "'NW', 'HE', 'RP', 'BW', "
                                               "'BY', 'SL', 'BE', 'BB', "
                                               "'MV', 'SN', 'ST', 'TH']")
    assert isinstance(state, str), "'state' needs to be a string."
    # check if a year was specified
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get("year", cfg["base_year"])

    # make a dict with nuts3 as keys (lk) and nuts1 as values (bl)
    nuts3_list = df_gas_switch.index
    nuts1_list = [bl_dict().get(int(i[: -3]))
                  for i in df_gas_switch.index.astype(str)]
    nuts3_nuts1_dict = dict(zip(nuts3_list, nuts1_list))

    # create multicolumn from df_gas_switch columns and index
    # count how many different applications there are
    amount_application = len(df_gas_switch.columns.unique(level=1))
    # count how many different wz (industrial/commercial branches) there are
    amount_wz = len(df_gas_switch.columns.unique(level=0))
    # count how many different regions there are
    regions = [k for k, v in nuts3_nuts1_dict.items() if v == state]
    amount_regions = len(regions)

    # create multicolumn for result DataFrame()
    multi_lk = [elem for elem in regions
                for _ in range(amount_application * amount_wz)]
    multi_wz = [elem for elem in df_gas_switch.columns.unique(level=0)
                for _ in range(amount_application)] * amount_regions
    multi_app = (list(df_gas_switch.columns.unique(level=1))
                 * amount_regions
                 * amount_wz)
    tuples = list(zip(*[multi_lk, multi_wz, multi_app]))
    multicolumn = pd.MultiIndex.from_tuples(tuples, names=["LK", "WZ",
                                                           "Anwendungen"])
    # create index for year in 15 min timesteps
    idx = pd.date_range(start=str(year), end=str(year+1), freq='15T')[:-1]
    # create new df for temporal disaggregated heat demand
    new_df = pd.DataFrame(index=idx, columns=multicolumn, dtype='float')

    # get normalized timeseries for temperature dependent and temperature
    # independent gas demand in CTS
    heat_norm, gas_total, gas_tempinde_norm = create_heat_norm_cts(
        detailed=True, state=state, year=year)
    # upsample heat_norm to quarter hours and
    # interpolate, then normalize
    heat_norm = (heat_norm.resample('15T').asfreq()
                 .interpolate(method='linear',
                              limit_direction='forward', axis=0))
    # extend DataFrame by 3 more periods
    extension = pd.DataFrame(index=pd.date_range(heat_norm.index[-1:]
                                                 .values[0], periods=4,
                                                 freq='15T')[-3:],
                             columns=heat_norm.columns)
    heat_norm = heat_norm.append(extension).fillna(method='ffill')
    # normalize
    heat_norm = heat_norm.divide(heat_norm.sum(), axis=1)
    # upsample gas_tempinde_norm to quarter hours
    gas_tempinde_norm = (gas_tempinde_norm.resample('15T').asfreq()
                         .interpolate(method='linear',
                                      limit_direction='forward', axis=0))
    # extend DataFrame by 3 more periods
    extension = pd.DataFrame(index=pd.date_range(gas_tempinde_norm.index[-1:]
                                                 .values[0], periods=4,
                                                 freq='15T')[-3:],
                             columns=gas_tempinde_norm.columns)
    gas_tempinde_norm = (gas_tempinde_norm.append(extension)
                         .fillna(method='ffill'))
    # normalize
    gas_tempinde_norm = gas_tempinde_norm.divide(gas_tempinde_norm.sum(),
                                                 axis=1)

    # create temp disaggregated gas demands per nuts-3, branch and app
    i = 1  # lk counter
    for lk in new_df.columns.get_level_values(0).unique():
        logger.info("Working on LK {}/{}."
                    .format(i, len(new_df.columns
                                   .get_level_values(0)
                                   .unique())))
        i += 1
        for branch in (df_gas_switch.loc[lk]
                       .index.get_level_values(0).unique()):
            for app in (df_gas_switch.loc[lk][branch].index):
                if app == 'Raumwärme':
                    new_df[lk, branch, app] = (
                        (df_gas_switch.loc[lk][branch, app])
                        * (heat_norm[lk, branch]))
                else:
                    new_df[lk, branch, app] = (
                        (df_gas_switch.loc[lk][branch, app])
                        * (gas_tempinde_norm[lk, branch]))

    # drop all columns that have only nan values
    new_df.dropna(axis=1, how='all', inplace=True)

    return new_df


def temporal_cts_elec_load_from_fuel_switch(df_temp_gas_switch, p_ground=0.36,
                                            p_air=0.58, p_water=0.06):
    """
    Converts timeseries of gas demand per NUTS-3 and branch and application to
        electric consumption timeseries. Uses COP timeseries for heat
        applications. uses efficiency for mechanical energy.
    Parameters
    -------
    df_temp_gas_switch : pd.DataFrame()
        timestamp as index, multicolumns with nuts-3, branch and applications.
        contains temporally disaggregated gas demand for fuel switch
    p_ground, p_air, p_water : float, default 0.36, 0.58, 0.06
        percentage of ground/air/water heat pumps sum must be 1
    Returns
    -------
    None.

    """
    if p_ground + p_air + p_water != 1:
        raise ValueError("sum of percentage of ground/air/water heat pumps"
                         " must be 1")
    # get year from dataframe index
    year = df_temp_gas_switch.index[0].year
    # create new DataFrame for results
    df_temp_elec_from_gas_switch = pd.DataFrame(index=df_temp_gas_switch.index,
                                                columns=(df_temp_gas_switch
                                                         .columns),
                                                data=0)
    # create index slicer for data selection
    col = pd.IndexSlice
    # select heat applications which will be converted using cop series for
    # different temperatur levels. use efficiency to convert from gas to heat.
    # select indoor heating
    df_heating_switch = (df_temp_gas_switch.loc[:, col[:, :, ['Raumwärme']]]
                         * get_efficiency_level('Raumwärme'))
    # get the COP timeseries for indoor heating --> T=40°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=40,
                                                              source='ambient',
                                                              year=year)
    # assert that the years of COP TS and year of df_temp are aligned
    assert (air_floor_cop.index.year.unique() ==
            df_heating_switch.index.year.unique()),\
        ("The year of COP ts does not match the year of the heat demand ts")
    df_temp_indoor_heating = (p_ground * (df_heating_switch
                                          .div(ground_floor_cop, level=0)
                                          .fillna(method='ffill'))
                              + p_air * (df_heating_switch
                                         .div(air_floor_cop, level=0)
                                         .fillna(method='ffill'))
                              + p_water * (df_heating_switch
                                           .div(water_floor_cop, level=0)
                                           .fillna(method='ffill')))
    # select process heating
    df_heating_switch = (df_temp_gas_switch.loc[:, col[:, :, ['Prozesswärme']]]
                         * get_efficiency_level('Prozesswärme'))
    # get the COP timeseries for process heating --> T=70°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=70,
                                                              source='ambient',
                                                              year=year)
    df_temp_process_heat = (p_ground * (df_heating_switch.div(ground_floor_cop,
                                                              level=0)
                                        .fillna(method='ffill'))
                            + p_air * (df_heating_switch.div(air_floor_cop,
                                                             level=0)
                                       .fillna(method='ffill'))
                            + p_water * (df_heating_switch
                                         .div(water_floor_cop, level=0)
                                         .fillna(method='ffill')))
    # select warm water heating
    df_heating_switch = (df_temp_gas_switch.loc[:, col[:, :, ['Warmwasser']]]
                         * get_efficiency_level('Warmwasser'))
    # get the COP timeseries for warm water  --> T=55°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=55,
                                                              source='ambient',
                                                              year=year)
    df_temp_warm_water = (p_ground * (df_heating_switch.div(ground_floor_cop,
                                                            level=0)
                                      .fillna(method='ffill'))
                          + p_air * (df_heating_switch.div(air_floor_cop,
                                                           level=0)
                                     .fillna(method='ffill'))
                          + p_water * (df_heating_switch
                                       .div(water_floor_cop, level=0)
                                       .fillna(method='ffill')))
    # select Mechanical Energy
    df_mechanical_switch = ((df_temp_gas_switch
                            .loc[:, col[:, :, ['Mechanische Energie']]])
                            * (get_efficiency_level('Mechanische Energie')
                               / 0.9))  # HACK! 0.9 = electric motor efficiency
    # add all dataframes together for electric demand per nuts3, branch and app
    df_temp_elec_from_gas_switch = (df_temp_elec_from_gas_switch
                                    .add(df_temp_indoor_heating, fill_value=0)
                                    .add(df_temp_process_heat, fill_value=0)
                                    .add(df_temp_warm_water, fill_value=0)
                                    .add(df_mechanical_switch, fill_value=0))
    return df_temp_elec_from_gas_switch


def temporal_industry_elec_load_from_fuel_switch(df_temp_gas_switch,
                                                 p_ground=0.36, p_air=0.58,
                                                 p_water=0.06):
    """
    Calculates electric consumption temporally disaggregated gas consumption,
    which will be switched to power.

    Parameters
    -------
    df_temp_gas_switch : pd.DataFrame()
        timestamp as index, multicolumns with nuts-3, branch and applications.
        contains temporally disaggregated gas demand for fuel switch
    p_ground, p_air, p_water : float, default 0.36, 0.58, 0.06
        percentage of ground/air/water heat pumps sum must be 1

    Returns
    -------
    pd.DataFrame : 3 DataFrames with electricity consumption
    None.

    """
    if p_ground + p_air + p_water != 1:
        raise ValueError("sum of percentage of ground/air/water heat pumps"
                         " must be 1")
    # get year from dataframe index
    year = df_temp_gas_switch.index[0].year

    # create new DataFrame for results
    df_temp_elec_from_gas_switch = pd.DataFrame(index=df_temp_gas_switch.index,
                                                columns=(df_temp_gas_switch
                                                         .columns),
                                                data=0)
    # create index slicer for data selection
    col = pd.IndexSlice

    # read share of electrode heating system for heat between 100°C and 200°C
    PATH = data_in("dimensionless", "fuel_switch_keys.xlsx")
    df_electrode = pd.read_excel(PATH,
                                 sheet_name=("Gas2Power industry electrode"))
    df_electrode = (df_electrode
                    .loc[[isinstance(x, int) for x in df_electrode["WZ"]]]
                    .set_index("WZ")
                    .copy())

    # 1: get the COP timeseries for indoor heating --> T=40°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=40,
                                                              source='ambient',
                                                              year=year)
    # assert that the years of COP TS and year of df_temp are aligned
    assert (air_floor_cop.index.year.unique() ==
            df_temp_gas_switch.index.year.unique()),\
        ("The year of COP ts does not match the year of the heat demand ts")

    # select indoor heating demand to be converted to electric demand with cop.
    # use efficiency to convert from gas to heat.
    df_hp_heat = (df_temp_gas_switch
                  .loc[:, col[:, :, ['Raumwärme']]]
                  * get_efficiency_level('Raumwärme'))
    df_temp_hp_heating = (p_ground * (df_hp_heat.div(ground_floor_cop, level=0)
                                      .fillna(method='ffill'))
                          + p_air * (df_hp_heat
                                     .div(air_floor_cop, level=0)
                                     .fillna(method='ffill'))
                          + p_water * (df_hp_heat
                                       .div(water_floor_cop, level=0)
                                       .fillna(method='ffill')))

    # 2: get the COP timeseries for low temperature process heat --> T=80°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=80,
                                                              source='ambient',
                                                              year=year)
    # select low temperature heat to be converted to electric demand with cop
    df_hp_heat = (df_temp_gas_switch
                  .loc[:, col[:, :, ['Prozesswärme <100°C']]]
                  * get_efficiency_level('Prozesswärme <100°C'))
    df_temp_hp_low_heat = (p_ground * (df_hp_heat
                                       .div(ground_floor_cop, level=0)
                                       .fillna(method='ffill'))
                           + p_air * (df_hp_heat
                                      .div(air_floor_cop, level=0)
                                      .fillna(method='ffill'))
                           + p_water * (df_hp_heat
                                        .div(water_floor_cop, level=0)
                                        .fillna(method='ffill')))

    # 3: get the COP timeseries for high temperature process heat
    # Use 2 heat pumps to reach this high temperature level
    # 3.1: 1st stage: T_sink = 60°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=60,
                                                              source='ambient',
                                                              year=year)
    # select heat demand to be converted to electric demand with cop
    df_hp_heat = ((df_temp_gas_switch
                   .loc[:, col[:, :, ['Prozesswärme 100°C-200°C']]]
                   * get_efficiency_level('Prozesswärme 100°C-200°C'))
                  .multiply((1-df_electrode['Prozesswärme 100°C-200°C']),
                            axis=1, level=1))
    df_temp_hp_medium_heat_stage1 = (p_ground * (df_hp_heat
                                                 .div(ground_floor_cop,
                                                      level=0)
                                                 .fillna(method='ffill'))
                                     + p_air * (df_hp_heat
                                                .div(air_floor_cop, level=0)
                                                .fillna(method='ffill'))
                                     + p_water * (df_hp_heat
                                                  .div(water_floor_cop,
                                                       level=0)
                                                  .fillna(method='ffill')))
    # 3.2: 2nd stage: use heat from first stage
    # T_sink = 120°C, T_source = 60°C delta_T = 60°C
    high_temp_hp_cop = cop_ts(source='waste heat', delta_t=60, year=year)
    # select heat to be converted to electric demand with cop
    df_temp_hp_medium_heat_stage2 = (df_hp_heat.div(high_temp_hp_cop, level=0)
                                     .fillna(method='ffill'))
    # add energy consumption of both stages
    df_temp_hp_medium_heat = (df_temp_hp_medium_heat_stage1
                              .add(df_temp_hp_medium_heat_stage2,
                                   fill_value=0))

    # 4 calculate electric demand for electrode heaters
    df_electrode_switch = ((df_temp_gas_switch
                            .loc[:, col[:, :, ['Prozesswärme 100°C-200°C']]]
                            * get_efficiency_level('Prozesswärme 100°C-200°C'))
                           .multiply((df_electrode['Prozesswärme 100°C-200°C']),
                                     axis=1, level=1)
                           / 0.98)  # HACK! 0.98 = electrode heater efficiency

    # 5 get the COP timeseries for warm water  --> T=55°C
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=55,
                                                              source='ambient',
                                                              year=year)

    # select warm water heat to be converted to electric demand with cop
    df_hp_heat = (df_temp_gas_switch
                  .loc[:, col[:, :, ['Warmwasser']]]
                  * get_efficiency_level('Warmwasser'))
    df_temp_warm_water = (p_ground * (df_hp_heat.div(ground_floor_cop, level=0)
                                      .fillna(method='ffill'))
                          + p_air * (df_hp_heat.div(air_floor_cop, level=0)
                                     .fillna(method='ffill'))
                          + p_water * (df_hp_heat
                                       .div(water_floor_cop, level=0)
                                       .fillna(method='ffill')))
    # 6 select Mechanical Energy
    df_mechanical_switch = ((df_temp_gas_switch
                            .loc[:, col[:, :, ['Mechanische Energie']]])
                            * (get_efficiency_level('Mechanische Energie')
                               / 0.9))  # HACK! 0.9 = electric motor efficiency

    # # 7 select Industrial power plants
    # df_self_gen_switch = ((df_temp_gas_switch
    #                        .loc[:, col[:, :, ['Industriekraftwerke']]])
    #                       * 0.35)  # TBD HACK! Update get_efficiency_level()

    # add all dataframes together for electric demand per nuts3, branch and app
    df_temp_elec_from_gas_switch = (df_temp_elec_from_gas_switch
                                    .add(df_temp_hp_heating, fill_value=0)
                                    .add(df_temp_hp_low_heat, fill_value=0)
                                    .add(df_temp_hp_medium_heat, fill_value=0)
                                    .add(df_temp_warm_water, fill_value=0)
                                    .add(df_mechanical_switch, fill_value=0)
                                    .add(df_electrode_switch, fill_value=0))
    # df_temp_elec_from_gas_switch = (df_temp_elec_from_gas_switch
    #                                 .loc[:, (df_temp_elec_from_gas_switch != 0)
    #                                      .any(axis=0)])
    

    return df_temp_elec_from_gas_switch#, df_temp_hp_medium_heat, df_electrode_switch]


def calculate_consumption_after_switch(df_old, df_switch, source):
    """
    

    Returns
    -------
    None.

    """
    if source == 'power':
        df_total = df_old.add(df_switch)
    else:
        df_total = df_old.sub(df_switch)

    return df_total


def disagg_temporal_applications_hp(source, sector, detailed=False, state=None,
                                    use_nuts3code=False, disagg_ph=False,
                                    use_blp=False, use_hp=True, **kwargs):
    """
    Perform dissagregation based on applications of the final energy usage.
    Uses temperature dependent timeseries for heat pumps for electricity
    consumption of indoor heating in CTS sector.

    Parameters
    ----------
    source : str
        must be one of ['power', 'gas']
    sector : str
        must be one of ['CTS', 'industry']
    detailed : bool, default False
        Throughput to functions disagg_temporal_industry(),
        disagg_temporal_gas_CTS() and disagg_temporal_power_CTS(). If True
        energy use per branch and disctrict get disaggreagated.
        Otherwise just the energy use per district
    use_nuts3code : bool, default False
        throughput for spatial disaggregation functions
        If True use NUTS-3 codes as region identifiers.
    disagg_ph : bool, default False
        If True: returns industrial gas consumption fpr process heat by
                 temperature level
    state : str, default None
        Defines state for disaggregation. Needs to be defined, if detailed=True
    use_blp: bool, default False
        If True, branch load profiles (blp) are used if available. blp are
        based on measured consumption data gathered during DemandRegio project.
    use_hp : bool, default True
        If True, use HP profiles for electricity consumption for space heating
        in CTS.

    Returns
    -------
    pd.DataFrame
        index = datetimeindex
        columns = Districts / (Branches) / Applications
    """
    # Step1: Read Input data
    # variable check
    assert (source in ['power', 'gas']), "`source` must be in ['power', 'gas']"
    assert (sector in ['CTS', 'industry']),\
        "`sector` must be in ['CTS', 'industry']"

    # check if a year was specified
    cfg = kwargs.get('cfg', get_config())
    year = kwargs.get("year", cfg["base_year"])
    # troughput values for the helper function, used for industrial disagg
    low = kwargs.get("low", 0.5)
    no_self_gen = kwargs.get("no_self_gen", False)

    # perfom spatial disagg of demand per consumer group and application
    df_app = disagg_applications_eff(source, sector, disagg_ph,
                                     use_nuts3code, no_self_gen, year=year)
    # count how many different applications there are
    amount_application = len(df_app.columns.unique(level=1))
    # count how many different wz (industrial/commercial branches) there are
    amount_wz = len(df_app.columns.unique(level=0))

    if not detailed:  # check if result should be detailed
        # Create temporal dataset
        # industry
        if sector == "industry":
            if use_blp:     # check if blp should be used
                assert (source in ['power']), ("`source` must be in set to "
                                               "'power' if use_blp=True")
                ec = disagg_temporal_industry_blp(source, detailed,
                                                  use_nuts3code, low,
                                                  no_self_gen, year=year)
            else:
                ec = disagg_temporal_industry(source, detailed, use_nuts3code,
                                              low, no_self_gen, year=year)
        # CTS
        else:
            if source == "gas":
                # this one has a different methodology
                ec = disagg_temporal_gas_CTS(detailed, use_nuts3code,
                                             year=year)
                # wrong column names are corrected
                ec.columns = ec.columns.set_names(["LK"])
            # power
            else:
                if use_blp:     # check if blp should be used
                    ec = disagg_temporal_power_CTS_blp(detailed, use_nuts3code,
                                                       year=year)
                else:
                    ec = disagg_temporal_power_CTS(detailed, use_nuts3code,
                                                   year=year)
                if use_hp:
                    hp_load_norm = create_hp_load(detailed,
                                                  use_nuts3code=False,
                                                  year=year, state=state)
                    # upsample hp_load to quarter hours and interpolate, then
                    # normalize
                    hp_load_norm = (hp_load_norm.resample('15T').asfreq()
                                    .interpolate(method='linear',
                                                 limit_direction='forward',
                                                 axis=0))
                    # extend DataFrame by 3 more periods, because it would be
                    # too short otherwise
                    extension = pd.DataFrame(index=pd.date_range(
                        hp_load_norm.index[-1:].values[0], periods=4,
                        freq='15T')[-3:], columns=hp_load_norm.columns)
                    hp_load_norm = (hp_load_norm.append(extension)
                                    .fillna(method='ffill'))
                    # normalize
                    hp_load_norm = hp_load_norm.divide(hp_load_norm.sum(),
                                                       axis=1)
        # Create temporal dataset with multiindex
        # creating the multiindex
        multi_lk = [elem for elem in list(ec.columns)
                    for _ in range(amount_application)]
        multi_app = list(df_app.columns.unique(level=1)) * len(ec.columns)
        tuples = list(zip(*[multi_lk, multi_app]))
        index = pd.MultiIndex.from_tuples(tuples, names=["LK", "Anwendung"])

        # new df with multiindex columns and datetime as index
        new_df = pd.DataFrame(columns=index, index=ec.index, dtype='float')

        # print info on the process
        logger.info("Working on disaggregating the applications for each "
                    "NUTS-3 region.")

        # percentage-values from the averages of the spatial disagg function
        # by region and application to use for multiplication
        percentages = (df_app.groupby(level=1, axis=1).mean()
                       .div(df_app.groupby(level=1, axis=1).mean().sum(axis=1),
                            axis=0))

        # for every lk multiply the consumption with the percentual use for
        # that application
        i = 1  # lk counter
        if use_hp:
            apps_no_heat = df_app.columns.unique(level=1).drop('Raumwärme')
            for lk in ec.columns:
                # provide info how far along the function is
                if i % 50 == 0:
                    logger.info("Working on LK {}/{}."
                                .format(i+1, len(new_df.columns
                                                 .get_level_values(0)
                                                 .unique())))
                i += 1
                for app in apps_no_heat:
                    new_df[lk, app] = percentages.loc[lk, app] * ec[lk]
                # for ambient heating use different energy
                new_df[lk, 'Raumwärme'] = (df_app.sum(axis=1, level=1)
                                           .loc[lk, 'Raumwärme']
                                           * hp_load_norm[lk])
        else:
            for lk in ec.columns:
                if i % 50 == 0:
                    logger.info("Working on LK {}/{}."
                                .format(i+1, len(new_df.columns
                                                 .get_level_values(0)
                                                 .unique())))
                i += 1
                for app in df_app.columns.unique(level=1):
                    new_df[lk, app] = percentages.loc[lk, app] * ec[lk]

        # Für detailed: Raumwärme für jeden LK und WZ
        # ((df_app.reorder_levels(order=[1,0], axis=1).loc[:, 'Raumwärme'])

    else:  # results will be "detailed"
        assert state in list(bl_dict().values()), ("'state' needs to be in "
                                                   "['SH', 'HH', 'NI', 'HB', "
                                                   "'NW', 'HE', 'RP', 'BW', "
                                                   "'BY', 'SL', 'BE', 'BB', "
                                                   "'MV', 'SN', 'ST', 'TH']")
        assert isinstance(state, str), "'state' needs to be a string."
        # create temporal dataset
        # industry
        if sector == "industry":
            if use_blp:
                assert (source in ['power']), ("`source` must be in set to "
                                               "'power' if use_blp=True")
                ec = disagg_temporal_industry_blp_by_state(source, detailed,
                                                           use_nuts3code, low,
                                                           no_self_gen,
                                                           state=state,
                                                           year=year)
            else:
                ec = disagg_temporal_industry_by_state(source, detailed,
                                                       use_nuts3code, low,
                                                       no_self_gen,
                                                       state=state, year=year)
        # CTS
        else:
            if source == "gas":
                ec = disagg_temporal_gas_CTS_by_state(detailed, use_nuts3code,
                                                      state=state, year=year)
                # wrong column names are corrected
                ec.columns = ec.columns.set_names(["LK", "WZ"])
            # power
            else:
                if use_blp:
                    ec = disagg_temporal_power_CTS_blp_by_state(detailed,
                                                                use_nuts3code,
                                                                state=state,
                                                                year=year)
                else:
                    ec = disagg_temporal_power_CTS_by_state(detailed,
                                                            use_nuts3code,
                                                            state=state,
                                                            year=year)
                if use_hp:
                    hp_load_norm = create_hp_load(detailed,
                                                  use_nuts3code=False,
                                                  year=year, state=state)
                    # upsample hp_load to quarter hours and interpolate, then
                    # normalize
                    hp_load_norm = (hp_load_norm.resample('15T').asfreq()
                                    .interpolate(method='linear',
                                                 limit_direction='forward',
                                                 axis=0))
                    # extend DataFrame by 3 more periods, because it would be
                    # too short otherwise
                    extension = pd.DataFrame(index=pd.date_range(
                        hp_load_norm.index[-1:].values[0], periods=4,
                        freq='15T')[-3:], columns=hp_load_norm.columns)
                    hp_load_norm = (hp_load_norm.append(extension)
                                    .fillna(method='ffill'))
                    # normalize
                    hp_load_norm = hp_load_norm.divide(hp_load_norm.sum(),
                                                       axis=1)
        # number of regions
        regions = list(ec.columns.get_level_values(0).unique())
        amount_regions = len(regions)
        # creating the multiindex
        multi_lk = [elem for elem in regions
                    for _ in range(amount_application * amount_wz)]
        multi_wz = [elem for elem in df_app.columns.unique(level=0)
                    for _ in range(amount_application)] * amount_regions
        multi_app = (list(df_app.columns.unique(level=1))
                     * amount_regions
                     * amount_wz)
        tuples = list(zip(*[multi_lk, multi_wz, multi_app]))
        columns = pd.MultiIndex.from_tuples(tuples, names=["LK", "WZ",
                                                           "Anwendungen"])

        # new df with multiindex columns and datetime as index
        new_df = pd.DataFrame(columns=columns, index=ec.index, dtype='float')
        # optional, give memory usage info
        # logger.info('Approximate memory usage of Dataframe in MB will be: '
        #           + str(new_df.memory_usage(deep=True).sum()/1000000))

        # percentage-values from the averages of the spatial disagg function
        # by region, industrial branch and app to use for multiplication
        percentages = (df_app.div(df_app.sum(axis=1, level=0), level=0).mean())

        i = 1  # lk counter
        # for every lk and WZ multiply the consumption with the percentual
        # use for that application
        if use_hp:  # if heat pump load profile should be used
            apps_no_heat = df_app.columns.unique(level=1).drop('Raumwärme')
            # for every region
            for lk in new_df.columns.get_level_values(0).unique():
                # provide info how far along the function is
                if i % 4 == 0:
                    logger.info("Working on LK {}/{}."
                                .format(i+1, len(new_df.columns
                                                 .get_level_values(0)
                                                 .unique())))
                i += 1
                for branch in new_df.columns.get_level_values(1).unique():
                    for app in apps_no_heat:
                        new_df[lk, branch, app] = (percentages.loc[branch, app]
                                                   * ec[lk, branch])
                    new_df[lk, branch, 'Raumwärme'] = ((df_app.loc[lk]
                                                       [branch, 'Raumwärme'])
                                                       * (hp_load_norm
                                                          [lk, branch]))
        else:  # HP load profile is not used
            # for every region
            for lk in new_df.columns.get_level_values(0).unique():
                # provide info how far along the function is
                if i % 50 == 0:
                    logger.info("Working on LK {}/{}."
                                .format(i+1, len(new_df.columns
                                                 .get_level_values(0)
                                                 .unique())))
                i += 1
                for branch in new_df.columns.get_level_values(1).unique():
                    for app in new_df.columns.get_level_values(2).unique():
                        new_df[lk, branch, app] = (percentages.loc[branch, app]
                                                   * ec[lk, branch])

    # Plausibility check:
    msg = ('The sum of consumptions (={:.3f}) and the sum of disaggrega'
           'ted consumptions (={:.3f}) do not match! Please check algorithm!')
    if detailed:
        # adding the complete consumption for every given branch before the
        # disaggregation
        total_sum = 0
        for branch in new_df.columns.get_level_values(1).unique():
            total_sum += ec.xs(branch, level="WZ", axis=1).sum().sum()
    else:
        # consumption trough all timesteps in every district
        total_sum = ec.sum().sum()
    # total sum of all disaggregated consumptions
    disagg_sum = new_df.sum().sum()
    assert np.isclose(total_sum, disagg_sum), msg.format(total_sum, disagg_sum)

    return new_df


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
    # create path for reading existing heat load profiles
    if not detailed:
        path = data_out('CTS_heat_norm_' + str(year) + '.csv')
    else:
        path = data_out('CTS_heat_norm_' + str(state) + '_'
                        + str(year) + '.csv')

    # get normalised heat load profiles
    if os.path.exists(path) and not detailed:
        logger.info('Reading existing heat norm timeseries for detailed: '
                    + str(detailed))
        heat_norm = pd.read_csv(path, index_col=0, header=[0])
        heat_norm.index = pd.to_datetime(heat_norm.index)
        heat_norm.columns = heat_norm.columns.astype(int)
    if os.path.exists(path) and detailed:
        logger.info('Reading existing heat norm timeseries for detailed: '
                    + str(detailed))
        heat_norm = pd.read_csv(path, index_col=0, header=[0, 1])
        heat_norm.index = pd.to_datetime(heat_norm.index)
        heat_norm.columns = heat_norm.columns.set_levels(
            [heat_norm.columns.levels[0].astype(int),
             heat_norm.columns.levels[1].astype(int)])
    else:
        logger.info('Creating heat norm timeseries for detailed: '
                    + str(detailed))
        heat_norm, gas_total, gas_tempinde = (create_heat_norm_cts(
            detailed=detailed, year=year, state=state))
        heat_norm.to_csv(path)

    # Creating COP timeseries
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(year=year)

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


def create_heat_norm_cts(detailed=False, state=None, **kwargs):
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
                                            use_nuts3code=False, year=year)
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
                                                     use_nuts3code=False,
                                                     year=year)
        gas_tempinde.to_csv(path)  # TBD Moved to function disagg_temporal gas
        gas_tempinde.columns = gas_tempinde.columns.astype(int)
    else:
        logger.info('Disaggregating detailed temperature independent gas'
                    ' consumption for state: ' + str(state))
        gas_tempinde = (
            disagg_temporal_gas_CTS_water_by_state(detailed=detailed,
                                                   use_nuts3code=False,
                                                   state=state, year=year))

    # create space heating timeseries: difference between total heat demand
    # and water heating demand
    heat_norm = (gas_total - gas_tempinde).clip(lower=0)

    temp_allo = resample_t_allo(year=year)
    # clip heat demand above heating threshold
    if detailed:
        df = (heat_norm
              .droplevel([1], axis=1)[temp_allo[temp_allo > 15].isnull()]
              .fillna(0))
        df.columns = heat_norm.columns
        heat_norm = df.copy()
    else:
        # set heat demand to 0 if temp_allo is higher then 15°C
        heat_norm = heat_norm[temp_allo[temp_allo > 15].isnull()].fillna(0)

    # normalise
    heat_norm = heat_norm.divide(heat_norm.sum(axis=0), axis=1)
    gas_tempinde_norm = gas_tempinde.divide(gas_tempinde.sum(axis=0), axis=1)

    # if not detailed:
    #     # safe as csv
    #     heat_norm.to_csv(data_out('CTS_heat_norm_' + str(year) + '.csv'))
    # else:
    #     heat_norm.to_csv(data_out('CTS_heat_norm_' + str(state) + '_'
    #                               + str(year) + '.csv'))

    return heat_norm, gas_total, gas_tempinde_norm


# def create_heat_norm_industry(state=None, slp='KO', df_gv_lk=None, **kwargs):
#     """
#     Creates normalised heat demand timeseries for CTS per NUTS-3, and branch
#     if detailed is set to True.

#     Parameters
#     ----------
#     detailed : bool, default False
#         If True heat demand per branch and disctrict is calculated.
#         Otherwise just the heat demand per district.
#     state : str, default None
#         Specifies state. Only needed if detailed=True. Must by one of the
#         entries of bl_dict().values(),
#         ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
#          'BB', 'MV', 'SN', 'ST', 'TH']
#     slp : str, default "KO"
#         musst be one of ['BD', 'BH', 'GA', 'GB', 'HA', 'KO', 'MF', 'MK', 'PD']

#     Returns
#     -------

#     """
#     # get base year
#     cfg = kwargs.get('cfg', get_config())
#     year = kwargs.get('year', cfg['base_year'])
#     # Idea: create normalized heat profile from gas SLP for offices but use
#     # industry consumption
#     if df_gv_lk is None:
#         gv_lk = (disagg_applications_eff(source="gas", sector="industry",
#                                          use_nuts3code=False, year=year))
#     else:
#         gv_lk = df_gv_lk

#     col = pd.IndexSlice
#     gv_lk_total = gv_lk.copy()
#     gv_lk_tempinde = gv_lk.loc[:, col[:, ['Warmwasser',
#                                           'Mechanische Energie',
#                                           'Prozesswärme']]]

#     # get daily allocation temperature
#     temperatur_df = t_allo(year=year)
#     # clip at 15 for hot water, below 15°C the water heating demand is assumed
#     # to be constant
#     temperatur_df_clip = temperatur_df.clip(15)
#     # check if gap year
#     if ((year % 4 == 0) & (year % 100 != 0) | (year % 4 == 0)
#             & (year % 100 == 0) & (year % 400 == 0)):
#         hours = 8784
#     else:
#         hours = 8760

#     # make a dict with nuts3 as keys (lk) and nuts1 as values (bl)
#     nuts3_list = temperatur_df.columns
#     nuts1_list = [bl_dict().get(int(i[: -3]))
#                   for i in temperatur_df.columns.astype(str)]
#     nuts3_nuts1_dict = dict(zip(nuts3_list, nuts1_list))

#     # count how many different regions there are
#     regions = [int(k) for k, v in nuts3_nuts1_dict.items() if v == state]

#     # new DataFrames for results
#     gas_total = pd.DataFrame(columns=regions,
#                              index=pd.date_range((str(year) + '-01-01'),
#                                                  periods=hours, freq='H'))
#     gas_temp_inde = pd.DataFrame(columns=regions,
#                                  index=pd.date_range((str(year) + '-01-01'),
#                                                      periods=hours, freq='H'))

#     # get weekday-factors per day
#     F_wd = (gas_slp_weekday_params(state, year=year)
#             .set_index('Date')['FW_'+str(slp)]).to_frame()
#     # get h-value per day
#     h_slp = h_value(slp, [str(i) for i in regions], temperatur_df)
#     # get h-value for hot water per day
#     h_slp_water = h_value_water(slp, [str(i) for i in regions],
#                                 temperatur_df_clip)

#     # multiply h_values and week day values per day
#     tw = pd.DataFrame(np.multiply(h_slp.values, F_wd.values),
#                       index=h_slp.index, columns=h_slp.columns.astype(int))
#     tw_water = pd.DataFrame(np.multiply(h_slp_water.values, F_wd.values),
#                             index=h_slp_water.index,
#                             columns=h_slp_water.columns.astype(int))

#     # normalize
#     tw_norm = tw/tw.sum()
#     tw_water_norm = tw_water/tw_water.sum()
#     # set DatetimeIndex
#     tw_norm.index = pd.DatetimeIndex(tw_norm.index)
#     tw_water_norm.index = pd.DatetimeIndex(tw_water_norm.index)
#     # multiply with gas demand per region
#     ts_total = tw_norm.multiply(gv_lk_total.sum(axis=1).loc[regions])
#     ts_water = tw_water_norm.multiply(gv_lk_tempinde.sum(axis=1).loc[regions])

#     # extend by one day because when resampling to hours later, this day
#     # is lost otherwise
#     last_day = ts_total.copy()[-1:]
#     last_day.index = last_day.index + timedelta(1)
#     ts_total = (ts_total.append(last_day).resample('H').pad()[:-1])
#     # extend tw_water by one day and resample
#     ts_water = (ts_water.append(last_day).resample('H').pad()[:-1])

#     # get temperature dataframe for hourly disaggregation
#     t_allo_df = temperatur_df[[str(i) for i in regions]]
#     for col in t_allo_df.columns:
#         t_allo_df[col].values[t_allo_df[col].values < -15] = -15
#         t_allo_df[col].values[(t_allo_df[col].values > -15)
#                               & (t_allo_df[col].values < -10)] = -10
#         t_allo_df[col].values[(t_allo_df[col].values > -10)
#                               & (t_allo_df[col].values < -5)] = -5
#         t_allo_df[col].values[(t_allo_df[col].values > -5)
#                               & (t_allo_df[col].values < 0)] = 0
#         t_allo_df[col].values[(t_allo_df[col].values > 0)
#                               & (t_allo_df[col].values < 5)] = 5
#         t_allo_df[col].values[(t_allo_df[col].values > 5)
#                               & (t_allo_df[col].values < 10)] = 10
#         t_allo_df[col].values[(t_allo_df[col].values > 10)
#                               & (t_allo_df[col].values < 15)] = 15
#         t_allo_df[col].values[(t_allo_df[col].values > 15)
#                               & (t_allo_df[col].values < 20)] = 20
#         t_allo_df[col].values[(t_allo_df[col].values > 20)
#                               & (t_allo_df[col].values < 25)] = 25
#         t_allo_df[col].values[(t_allo_df[col].values > 25)] = 100
#         t_allo_df = t_allo_df.astype('int32')
#     # for tempindependent consumption of ts_water t_allo_df = 100
#     t_allo_water_df = t_allo_df.copy()
#     t_allo_water_df.values[:] = 100

#     # rewrite calendar for better data handling later
#     calender_df = (gas_slp_weekday_params(state, year=year)
#                    [['Date', 'MO', 'DI', 'MI', 'DO', 'FR', 'SA', 'SO']])
#     # add temperature data to calendar
#     temp_calender_df = (pd.concat([calender_df.reset_index(),
#                                    t_allo_df.reset_index()], axis=1))
#     temp_calender_water_df = (pd.concat([calender_df.reset_index(),
#                                          t_allo_water_df.reset_index()],
#                                         axis=1))
#     # add weekdays to calendar
#     temp_calender_df['Tagestyp'] = 'MO'
#     temp_calender_water_df['Tagestyp'] = 'MO'
#     for typ in ['DI', 'MI', 'DO', 'FR', 'SA', 'SO']:
#         (temp_calender_df.loc[temp_calender_df[typ], 'Tagestyp']) = typ
#         (temp_calender_water_df
#          .loc[temp_calender_water_df[typ], 'Tagestyp']) = typ

#     # get hourly percentages per slp and region, here only one slp is used
#     for lk in regions:
#         # for calendar for gas total
#         temp_cal = temp_calender_df.copy()
#         temp_cal = temp_cal[['Date', 'Tagestyp', str(lk)]].set_index("Date")
#         last_hour = temp_cal.copy()[-1:]
#         last_hour.index = last_hour.index + timedelta(1)
#         temp_cal = temp_cal.append(last_hour)
#         temp_cal = temp_cal.resample('H').pad()
#         temp_cal = temp_cal[:-1]
#         temp_cal['Stunde'] = pd.DatetimeIndex(temp_cal.index).time
#         temp_cal = temp_cal.set_index(["Tagestyp", str(lk), 'Stunde'])
#         f = ('Lastprofil_{}.xls'.format(slp))
#         slp_profil = pd.read_excel(data_in('temporal',
#                                            'Gas Load Profiles', f))
#         slp_profil = pd.DataFrame(slp_profil.set_index(['Tagestyp',
#                                                 'Temperatur\nin °C\nkleiner']))
#         slp_profil.columns = pd.to_datetime(slp_profil.columns,
#                                             format='%H:%M:%S')
#         slp_profil.columns = pd.DatetimeIndex(slp_profil.columns).time
#         slp_profil = slp_profil.stack()
#         temp_cal['Prozent'] = [slp_profil[x] for x in temp_cal.index]
#         # multiplication of gas demand with hourly factors
#         gas_total[int(lk)] = (ts_total[lk].values
#                               * temp_cal['Prozent'].values/100)
#         # for calendar for calendar gas temp independent
#         temp_cal = temp_calender_water_df.copy()
#         temp_cal = temp_cal[['Date', 'Tagestyp', str(lk)]].set_index("Date")
#         last_hour = temp_cal.copy()[-1:]
#         last_hour.index = last_hour.index + timedelta(1)
#         temp_cal = temp_cal.append(last_hour)
#         temp_cal = temp_cal.resample('H').pad()
#         temp_cal = temp_cal[:-1]
#         temp_cal['Stunde'] = pd.DatetimeIndex(temp_cal.index).time
#         temp_cal = temp_cal.set_index(["Tagestyp", str(lk), 'Stunde'])
#         f = ('Lastprofil_{}.xls'.format(slp))
#         slp_profil = pd.read_excel(data_in('temporal',
#                                            'Gas Load Profiles', f))
#         slp_profil = pd.DataFrame(slp_profil.set_index(['Tagestyp',
#                                                 'Temperatur\nin °C\nkleiner']))
#         slp_profil.columns = pd.to_datetime(slp_profil.columns,
#                                             format='%H:%M:%S')
#         slp_profil.columns = pd.DatetimeIndex(slp_profil.columns).time
#         slp_profil = slp_profil.stack()
#         temp_cal['Prozent'] = [slp_profil[x] for x in temp_cal.index]
#         # multiplication of temperatur independent gas demand with hourly
#         # factors
#         gas_temp_inde[int(lk)] = (ts_water[lk].values
#                                   * temp_cal['Prozent'].values/100)

#     # create space heating timeseries: difference between total heat demand
#     # and water heating demand
#     heat_norm = (gas_total - gas_temp_inde).clip(lower=0)

#     temp_allo = (resample_t_allo(temp_df=temperatur_df,
#                                  year=year)[[(i) for i in regions]])
#     heat_norm = heat_norm[temp_allo[temp_allo > 15].isnull()].fillna(0)

#     # normalise
#     heat_norm = heat_norm.divide(heat_norm.sum(axis=0), axis=1)
#     gas_tempinde_norm = gas_temp_inde.divide(gas_temp_inde.sum(axis=0), axis=1)

#     return heat_norm, gas_total, gas_tempinde_norm


def cop_ts(sink_t=40, source='ambient', delta_t=None, cf=0.85, **kwargs):
    """
    Creates COP timeseries for ground/air/water heat pumps with floor heating.

    Parameters
    ----------
    sink_t : float, default 40
        temperature level of heat sink
    source : str, must be in ['ambient', 'waste heat'], default 'ambient'
        defines heat source for heat pump. If 'ambient', sink_t must be
        defined. if 'waste heat' delta_t must be defined.
    delata_t : float, default None.
        must be defined, if source is set to 'waste heat'. defines temperature
        difference between sink and source for heat pump. should not exceed 80.
    cf : float, default 0.85
        correction factor to be multiplied with COP-series to account for
        real world real-world effects, as opposed to ideal conditions under
        which the initial regression data was obtained.
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
    assert source in ['ambient', 'waste heat'], ("'source' needs to be in "
                                                 "['ambient', 'waste heat'].")
    # get base year (if a future year, assing historical weather year)
    cfg = kwargs.get('cfg', get_config())
    base_year = kwargs.get('year', cfg['base_year'])
    year = hist_weather_year().get(base_year)
    while year > 2019:
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
    if source == 'ambient':
        # sink temperature
        sink_t = pd.DataFrame(index=ground_t.index,
                              columns=ground_t.columns,
                              data=sink_t)
        # create cop timeseries based on temperature difference between source
        # and sink
        air_floor_cop = cop_curve((sink_t - air_t), 'air')
        ground_floor_cop = cop_curve((sink_t - ground_t), 'ground')
        water_floor_cop = cop_curve((sink_t - water_t), 'water')

        # change columns to int
        air_floor_cop.columns = air_floor_cop.columns.astype(int)
        ground_floor_cop.columns = ground_floor_cop.columns.astype(int)
        water_floor_cop.columns = water_floor_cop.columns.astype(int)

        return air_floor_cop*cf, ground_floor_cop*cf, water_floor_cop*cf

    # if source == 'waste heat' 
    else:
        assert isinstance(delta_t, (int, float)), ("'delta_t' needs to be a "
                                                   "number.")
        # temperature difference between source and sink
        delta_t = pd.DataFrame(index=ground_t.index,
                               columns=ground_t.columns,
                               data=delta_t)
        # create cop timeseries based on temperature difference between source
        # and sink. use regression function from Arpagaus et al. (2018)
        # "Review - High temperature heat pumps: Market overview, state of
        #  the art, research Status, refrigerants, and application potentials",
        # Energy, 2018
        high_temp_hp_cop = 68.455*(delta_t.pow(-0.76))

        high_temp_hp_cop.columns = high_temp_hp_cop.columns.astype(int)

        return high_temp_hp_cop*cf


# %% Utility functions


def projection_fuel_switch_share(df_fuel_switch, target_year):
    """
    Projects fuel switch share by branch to target year.

    Parameters
    ----------
    df_fuel_switch : pd.DataFrame()
        Data which is projected.
    target_year: int
        Year for which the share should be projected.

    Returns
    -------
    None.

    """
    start_year = int(2019)
    if target_year <= start_year:
        logger.info('Target year is lower than base year. No projection is'
                    ' done. Target year:' + str(target_year) + ' and base'
                    ' year: ' + str(start_year))
        # as there is no projection the share of fuel which is switched away
        # from is 0.
        for col in df_fuel_switch.columns:
            df_fuel_switch[col].values[:] = 0

        return df_fuel_switch
    elif target_year <= 2045:
        # define yearly step from today to 2045
        df_scaling = df_fuel_switch.div(2045-start_year)
        # project to target year
        df_fuel_switch_projected = df_scaling*(target_year - start_year)
        return df_fuel_switch_projected
    else:
        return df_fuel_switch


def read_fuel_switch_share(sector, switch_to):
    """
    Read fuel switch shares by branch from input data for year 2045.

    Parameters
    -------
    sector : str
        must be one of ['CTS', 'industry']
    switch_to: str
        must be one of ['power', 'hydrogen']
    Returns
    -------
    pd.DataFrame()

    """
    assert (sector in ['CTS', 'industry']),\
        "`sector` must be in ['CTS', 'industry']"
    # reading and preapring the table
    PATH = data_in("dimensionless", "fuel_switch_keys.xlsx")
    if sector == "CTS":
        assert (switch_to in ['power']),\
            "`switch_to` must be 'power' for CTS sector."
        df_fuel_switch = pd.read_excel(PATH, sheet_name=("Gas2Power CTS 2045"),
                                       skiprows=1)
    if sector == 'industry':
        assert (switch_to in ['power', 'hydrogen']),\
            "`switch_to` must be in ['power', 'hydrogen'] for industry sector."
        if switch_to == 'power':
            df_fuel_switch = pd.read_excel(PATH,
                                           sheet_name=("Gas2Power "
                                                       "industry 2045"),
                                           skiprows=1)
        else:
            df_fuel_switch = pd.read_excel(PATH,
                                           sheet_name=("Gas2Hydrogen"
                                                       " industry 2045"),
                                           skiprows=1)
    # cleaning the table
    # selecting only the rows with WZ and not the name columns
    df_fuel_switch = (df_fuel_switch
                      .loc[[isinstance(x, int) for x in df_fuel_switch["WZ"]]]
                      .set_index("WZ")
                      .copy())
    return df_fuel_switch


# def resample_t_allo(temp_df=None, **kwargs):
#     """
#     Resamples the allocation temperature to match other dataframes in plot
#     Parameters:
#     temp_df : pd.DataFrame with temperature data, default None
#     """
#     # get base year
#     cfg = kwargs.get('cfg', get_config())
#     year = kwargs.get('year', cfg['base_year'])

#     # change index to datetime
#     if ((year % 4 == 0)
#             & (year % 100 != 0)
#             | (year % 4 == 0)
#             & (year % 100 == 0)
#             & (year % 400 == 0)):
#         periods = 35136
#     else:
#         periods = 35040
#     date = pd.date_range((str(year) + '-01-01'), periods=periods / 4, freq='H')
#     if temp_df is not None:
#         df = temp_df
#     else:
#         df = t_allo(year=year)
#     df.index = pd.to_datetime(df.index)
#     last_hour = df.copy()[-1:]
#     last_hour.index = last_hour.index + timedelta(1)
#     df = df.append(last_hour)

#     # resample
#     df = df.resample('H').pad()
#     df = df[:-1]
#     df.index = date
#     df.columns = df.columns.astype(int)

#     return df


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
    delta_t.clip(lower=15, inplace=True)
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
