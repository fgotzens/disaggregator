from .spatial import disagg_applications, disagg_households_power
from .config import (data_in, data_out, get_config, dict_region_code,
                     hist_weather_year, bl_dict)
from .data import (database_shapes, ambient_T, CTS_power_slp_generator,
                   households_per_size, t_allo)
from .temporal import (disagg_temporal_gas_CTS, disagg_temporal_gas_CTS_water,
                       disagg_temporal_gas_households,
                       disagg_temporal_power_CTS)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
cfg = get_config()


def create_heat_norm(detailed=False, **kwargs):
    """
    Creates normalised heat demand timeseries for CTS and HH

    Parameters
    ----------
    detailed : bool, default False
        If True heat demand per branch and disctrict is calculated.
        Otherwise just the heat demand per district.

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

    else:
        logger.info('Disaggregating total gas consumption')
        gas_total = disagg_temporal_gas_CTS(detailed=detailed,
                                            use_nuts3code=False)
        if not detailed:
            gas_total.to_csv(path)
            gas_total.columns = gas_total.columns.astype(int)

    # create timeseries for temperature independent gas consumption
    path = data_out('CTS_gas_tempinde_' + str(year) + '.csv')
    if os.path.exists(path) and not detailed:
        logger.info('Reading existing temperature independent gas temporal '
                    'disaggregated timeseries.')
        gas_tempinde = pd.read_csv(path, index_col=0)
        gas_tempinde.index = pd.to_datetime(gas_tempinde.index)
        gas_tempinde.columns = gas_tempinde.columns.astype(int)

    else:
        logger.info('Disaggrating temperature independent gas consumption')
        gas_tempinde = disagg_temporal_gas_CTS_water(detailed=detailed,
                                                     use_nuts3code=False)
        if not detailed:
            gas_tempinde.to_csv(path)
            gas_tempinde.columns = gas_tempinde.columns.astype(int)

    # create space heating ts: difference between total heat demand
    # and water heating demand
    heat_norm = (gas_total - gas_tempinde).clip(lower=0)

    t_allo = resample_t_allo()
    # clip heat demand above heating threshold
    if detailed:
        df = (heat_norm
              .droplevel([1], axis=1)[t_allo[t_allo > 13].isnull()]
              .fillna(0))
        df.columns = heat_norm.columns
        heat_norm = df
    else:
        heat_norm = heat_norm[t_allo[t_allo > 13].isnull()].fillna(0)

    # normalise
    heat_norm = heat_norm.divide(heat_norm.sum(axis=0), axis=1)

    if not detailed:
        # safe as csv
        heat_norm.to_csv(data_out('CTS_heat_norm_' + str(year) + '.csv'))

    return heat_norm, gas_total, gas_tempinde


def create_heat_norm_old(sector, detailed=False, **kwargs):
    """
    Creates normalised heat demand timeseries for CTS and HH

    Parameters
    ----------
    sector : str
        must be one of ['HH','CTS']
    detailed : bool, default False
        If True heat demand per branch and disctrict is calculated.
        Otherwise just the heat demand per district.

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
    year = kwargs.get('year', cfg['base_year'])

    if sector not in ['CTS', 'HH']:
        raise ValueError("`sector` must be in ['CTS', 'HH']")

    # create timeseries total gas consumption
    path = data_out(sector + '_gas_total_' + str(year) + '.csv')
    if os.path.exists(path) and detailed == False:
        logger.info('Reading existing total gas temporal disaggregated timeseries.')
        gas_total = pd.read_csv(path, index_col=0)
        gas_total.index = pd.to_datetime(gas_total.index)
        gas_total.columns = gas_total.columns.astype(int)

    else:
        logger.info('Disaggrating total gas consumption')
        if sector == 'CTS':
            gas_total = disagg_temporal_gas_CTS(detailed=detailed, use_nuts3code=False)
            print(gas_total)
            if detailed == False:
                gas_total.to_csv(path)
                gas_total.columns = gas_total.columns.astype(int)
        if sector == 'HH':
            gas_total = disagg_temporal_gas_households(use_nuts3code=False)
            gas_total.to_csv(path)
            gas_total.columns = gas_total.columns.astype(int)

    # create timeseries for temperature independent gas consumption	
    path = data_out(sector + '_gas_tempinde_' + str(year) + '.csv')
    if os.path.exists(path) and detailed == False:
        logger.info('Reading existing temperature independent gas temporal disaggregated timeseries.')
        gas_tempinde = pd.read_csv(path, index_col=0)
        gas_tempinde.index = pd.to_datetime(gas_tempinde.index)
        gas_tempinde.columns = gas_tempinde.columns.astype(int)

    else:
        logger.info('Disaggrating temperature independent gas consumption')
        if sector == 'CTS':
            gas_tempinde = disagg_temporal_gas_CTS_water(detailed=detailed, use_nuts3code=False)
            if detailed == False:
                gas_tempinde.to_csv(path)
                gas_tempinde.columns = gas_tempinde.columns.astype(int)
        if sector == 'HH':
            gas_tempinde = disagg_temporal_gas_households_water(use_nuts3code=False)
            gas_tempinde.to_csv(path)
            gas_tempinde.columns = gas_tempinde.columns.astype(int)

    # create space heating ts, difference between total heat demand and water heating demand
    heat_norm = (gas_total - gas_tempinde).clip(lower=0)

    t_allo = resample_t_allo()
    # clip heat demand above heating threshold
    if detailed == True:
        df = heat_norm.droplevel([1], axis=1)[t_allo[t_allo > 13].isnull()].fillna(0)
        df.columns = heat_norm.columns
        heat_norm = df
    else:
        heat_norm = heat_norm[t_allo[t_allo > 13].isnull()].fillna(0)

    # normalise
    heat_norm = heat_norm.divide(heat_norm.sum(axis=0), axis=1)

    if detailed == False:
        # safe as csv
        heat_norm.to_csv(data_out(sector + '_heat_norm_' + str(year) + '.csv'))

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

    date = pd.date_range((str(base_year) + '-01-01'), periods=periods / 4, freq='H')

    # get source temperatures
    soil_t = soil_temp(year)
    soil_t.index = date

    # heat loss between soil and brine
    ground_t = soil_t.sub(5)

    # get air temperature
    air_t = ambient_T(year=year)
    air_t = change_nuts3_to_ags(air_t)
    air_t.index = date

    # create water temperatur DataFrame (constant temperature 0f 10°C, heat loss between water and brine)
    water_t = pd.DataFrame(index=ground_t.index, columns=ground_t.columns, data=10 - 5)

    # sink temperature of 40°C (floor heatin)
    floor_t = pd.DataFrame(index=ground_t.index, columns=ground_t.columns, data=40)

    # read cop parameters (from When2Heat, Ruhnau et al.)
    cop_params = pd.read_csv(data_in('heat', 'cop_parameters.csv'), sep=';', decimal=',', header=0, index_col=0).apply(
        pd.to_numeric, downcast='float')

    # create cop timeseries based on temperature difference between source and sink
    air_floor_cop = cop_curve((floor_t - air_t), 'air')
    ground_floor_cop = cop_curve((floor_t - ground_t), 'ground')
    water_floor_cop = cop_curve((floor_t - water_t), 'water')

    return air_floor_cop, ground_floor_cop, water_floor_cop


def RW_ec(sector, p_ground=0.36, p_air=0.58, p_water=0.06,
          RW_scenario=0, detailed=False, WP_Netz=None, **kwargs):
    """
	Creates timeseries energy demand of heat pumps [MWh]
	
	Parameters
	----------
	sector : str
		must be one of ['HH','CTS']
	p_ground, p_air, p_water : float, default 0.36, 0.58, 0.06
		percentage of ground/air/water heat pumps
		sum must be 1
	RW_scenario : float, default 0
	    future energy use for space heating [TWH]
		must be higher than inital energy consumption for space heating
	detailed : bool, default False
		If True heat demand per branch and disctrict is calculated.
		Otherwise just the heat demand per district.
	WP_Netz : pd.DataFrame
	    timeseries of reduced energy net consumption for Households with battery- and bufferstorage
		Output of function prosumer_hh_bl
		index = datetimeindex
        columns = Districts
		
	Returns
	-------
	pd.DataFrame
		index = datetimeindex
		columns = Districts
	"""

    if sector not in ['CTS', 'HH']:
        raise ValueError("`sector` must be in ['CTS', 'HH']")

    if p_ground + p_air + p_water != 1:
        raise ValueError("sum of percentage of ground/air/water heat pumps must be 1")

    # get base year
    year = kwargs.get('year', cfg['base_year'])
    path = data_out(sector + '_heat_norm' + str(year) + '.csv')

    # get normalised heat timeseries
    if os.path.exists(path) and detailed == False:
        logger.info('Reading existing heat norm timeseries.')
        heat_norm = pd.read_csv(path, index_col=0, header=[0])
        heat_norm.index = pd.to_datetime(heat_norm.index)
        heat_norm.columns = heat_norm.columns.astype(int)

    else:
        logger.info('Creating heat norm timeseries.')
        heat_norm, gas_total, gas_tempinde = create_heat_norm(detailed=detailed, sector=sector)

    # Creating COP timeseries
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts()

    # Get total energy consumption for space heating per district
    if sector == 'CTS':
        ec_CTS_appl = disagg_applications(source="power", sector="CTS", use_nuts3code=False)
        ec_RW = ec_CTS_appl.reorder_levels(order=[1, 0], axis=1).loc[:, 'Raumwärme']
    if sector == 'HH':
        df_spat = disagg_households_power('households', weight_by_income=True) * 1e3
        df_spat = change_nuts3_to_ags(df_spat.T)
        ec_RW = df_spat.sum(axis=0) * 0.07  # 7% energy consumption for space heating in 2015, from AGEB (2017)

    # distribute additional consumption among LK
    if RW_scenario > 0:
        if RW_scenario * 1e6 < ec_RW.sum().sum():
            raise ValueError("RW_scenario must be higher than inital energy consumption for space heating")

        diff = RW_scenario * 1e6 - ec_RW.sum().sum()
        if sector == 'CTS':
            RW_share_lk = ec_RW.sum(axis=1) / ec_RW.sum().sum()
            RW_share_wz = ec_RW.divide(ec_RW.sum(axis=1), axis=0)
            ec_RW = ec_RW + RW_share_wz.multiply(RW_share_lk, axis=0) * diff
        if sector == 'HH':
            RW_share_lk = ec_RW / ec_RW.sum()
            ec_RW = ec_RW + RW_share_lk * diff

    # compute energy consumption timeseries for heat pumps with COP
    ec_heat = (p_ground * heat_norm.div(ground_floor_cop, level=0)
               + p_air * heat_norm.div(air_floor_cop, level=0)
               + p_water * heat_norm.div(water_floor_cop, level=0))
    ec_heat = ec_heat.divide(ec_heat.sum(axis=0), axis=1)

    if detailed and sector == 'CTS':
        ec_dict = ec_RW.to_dict('index')
        for col in ec_heat:
            ec_heat[col] = ec_heat[col].multiply(ec_dict[col[0]][col[1]])
    else:
        if sector == 'CTS':
            ec_heat = ec_RW.sum(axis=1) * ec_heat
        if sector == 'HH':
            ec_heat = ec_RW * ec_heat
            # substract reductions in grid consumption due to storage use
            if type(WP_Netz) == pd.DataFrame:
                ec_heat.loc[:, WP_Netz.columns] = ec_heat.loc[:, WP_Netz.columns] + (WP_Netz / 10e3).resample('H').sum()

    return ec_heat


def create_heat_ts_EFH(Wohnfläche, JAZ=3.7, spez_Wärm=60, **kwargs):
    """
    Creates normalised timeseries of energy demand for heat pumps in single-family houses
	
	Parameters
	----------
	Wohnfläche : float
		living space of the household
	JAZ : float, default 3.7
		annual COP of air heat pump
	spez_Wärm : float, default 60
		specific heat demand of the household
		
	Returns
	-------
	ec_heat_ts : pd.DataFrame
		normalised energy consumption
		index = datetimeindex
		columns = Districts
	air_floor_cop
	    COP timeseries for air heat pump and floor heating
	    index = datetimeindex
		columns = Districts
    """
    # get base year
    year = kwargs.get('year', cfg['base_year'])

    # create timeseries total gas consumption
    path = data_out('gas_temp_norm_' + str(year) + '.csv')
    if os.path.exists(path):
        logger.info('Reading existing total gas temporal disaggregated timeseries.')
        gas_temp_norm = pd.read_csv(path, index_col=0)
        gas_temp_norm.index = pd.to_datetime(gas_temp_norm.index)
    else:
        logger.info('Disaggrating total gas consumption')
        gas_temp = disagg_temporal_gas_households_EFH()
        gas_temp = gas_temp.loc[:, gas_temp.columns.str.contains('EFH')]
        gas_temp_norm = gas_temp / gas_temp.sum(axis=0)
        gas_temp_norm.to_csv(path)

    # create timeseries temperature independent gas consumption
    path = data_out('tempinde_temp_norm_' + str(year) + '.csv')
    if os.path.exists(path):
        logger.info('Reading existing temperature independent gas temporal disaggregated timeseries.')
        tempinde_temp_norm = pd.read_csv(path, index_col=0)
        tempinde_temp_norm.index = pd.to_datetime(tempinde_temp_norm.index)
    else:
        logger.info('Disaggrating temperature independent gas consumption')
        tempinde_temp = disagg_temporal_gas_households_water_EFH()
        tempinde_temp = tempinde_temp.loc[:, tempinde_temp.columns.str.contains('EFH')]
        tempinde_temp_norm = tempinde_temp / tempinde_temp.sum(axis=0)
        tempinde_temp_norm.to_csv(path)

    # create space heating ts, difference between total heat demand and water heating demand
    heat_norm = (gas_temp_norm - tempinde_temp_norm).clip(lower=0)
    heat_norm.columns = heat_norm.columns.str.split('_').str[0].astype(int)
    # clip heat demand above heating threshold
    t_allo = resample_t_allo()
    heat_norm = heat_norm[t_allo[t_allo > 13].isnull()].fillna(0)
    heat_norm = heat_norm.resample('15Min').interpolate().append(
        pd.DataFrame(heat_norm.loc[heat_norm.iloc[-1:, :].index.repeat(3)]).set_index(
            pd.date_range((str(year) + '-12-31 23:15:00'), periods=3, freq='0.25H')))

    # create COP timeseries
    air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts()

    # resample COP to 15 min resolution
    air_floor_cop = air_floor_cop.resample('15Min').interpolate().append(
        pd.DataFrame(air_floor_cop.loc[air_floor_cop.iloc[-1:, :].index.repeat(3)]).set_index(
            pd.date_range((str(year) + '-12-31 23:15:00'), periods=3, freq='0.25H')))

    ec_heat = heat_norm.div(air_floor_cop, level=0)
    ec_heat = ec_heat.divide(ec_heat.sum())

    # scale timeseries with total energy consumption of the heat pump
    ec_heat_ts = (spez_Wärm * Wohnfläche * ec_heat) / JAZ

    return ec_heat_ts, air_floor_cop


def create_PV_ts(jvlh, PV_peak, **kwargs):
    """
	Create PV timeseries for one single-family household [kWh]
	
	Parameters
	----------
	jvlh : float
		annual full load hours of the state
	PV_peak : float
		installed PV capacity
		
	Returns
	-------
	pd.Series
		index = datetimeindex
	"""

    # get base year
    year = kwargs.get('year', cfg['base_year'])
    hist_year = year
    while hist_year > 2017:
        hist_year = hist_weather_year().get(hist_year)

    # read PV timeseries from transmission system operator 50Hertz
    path = data_in('heat', 'PV_' + str(hist_year) + '_50Hertz.csv')
    PV = pd.read_csv(path, header=4, sep=';', usecols=[0, 1, 2, 3])

    # set datetime index
    if ((year % 4 == 0)
            & (year % 100 != 0)
            | (year % 4 == 0)
            & (year % 100 == 0)
            & (year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040

    date = pd.date_range((str(year) + '-01-01'), periods=periods, freq='0.25H')
    PV.index = date
    PV = PV.loc[:, 'MW']

    # scale to household generation
    PV_norm = PV / PV.sum()
    PV_lk = PV_norm * jvlh * PV_peak

    return PV_lk


def use_storage(df, s_max, b_max):
    """
	Create storage timeseries using surplus PV energy
	
	Parameters
	----------
	df : pd.DataFrame
		index = datetimeindex
		columns = ['PV-Ertrag','COP','HH Netzbezug mit Speicher','HH Netzbezug ohne Speicher'
                   'Stand Batteriespeicher','WP Netzbezug mit Speicher','WP Netzbezug ohne Speicher',
				   'Stand Pufferspeicher','PV Netzeinspeisung']
	s_max : float
		maximum capacity of buffer storage [kWh]
	b_max : float
		maximum capacity of battery storage [kWh]
		
	Returns
	-------
	pd.DataFrame
		index and columns same as input df
	"""

    # maximum input and output power
    in_out_max = b_max * 0.48 * 0.25

    for timestep, row in df.iterrows():

        # PV Erzeugung reicht aus um HH Strom zu decken
        if row['HH Netzbezug mit Speicher'] < row['PV-Ertrag']:
            diff = row['PV-Ertrag'] - row['HH Netzbezug mit Speicher']
            # Deckung Haushaltsstrom
            df.at[timestep, 'HH Netzbezug mit Speicher'] = 0

        # PV Erzeugung reicht nicht aus um HH Strom zu decken,
        # ggf. Nutzung von Strom aus Batteriespeicher
        else:
            b_out = min(in_out_max,
                        (row['HH Netzbezug mit Speicher'] - row['PV-Ertrag']),
                        row['Stand Batteriespeicher'])

            df.at[timestep, 'HH Netzbezug mit Speicher'] -= (row['PV-Ertrag'] + b_out)
            df.at[timestep, 'Stand Batteriespeicher'] -= b_out

            diff = 0

        # Überschussstrom reicht nicht aus, um Nachfrage der WP zu decken
        if diff < row['WP Netzbezug mit Speicher']:
            df.at[timestep, 'WP Netzbezug mit Speicher'] -= diff
            diff = 0

            # Nutzung der Wärme aus Pufferspeicher
            if row['Stand Pufferspeicher'] > 0:

                # Wärme im Pufferspeicher reicht aus, um Bedarf zu decken
                if row['Stand Pufferspeicher'] > (df.at[timestep, 'WP Netzbezug mit Speicher'] * row['COP']):
                    df.at[timestep, 'Stand Pufferspeicher'] -= (
                            df.at[timestep, 'WP Netzbezug mit Speicher'] * row['COP'])
                    df.at[timestep, 'WP Netzbezug mit Speicher'] = 0

                # Wärme im Pufferspeicher reicht nicht aus, um Bedarf zu decken
                else:
                    df.at[timestep, 'WP Netzbezug mit Speicher'] -= (row['Stand Pufferspeicher'] / row['COP'])
                    df.at[timestep, 'Stand Pufferspeicher'] = 0

            # Nutzung des Stroms aus Batteriespeicher
            if df.at[timestep, 'Stand Batteriespeicher'] > 0:
                b_out = min(in_out_max,
                            df.at[timestep, 'Stand Batteriespeicher'],
                            df.at[timestep, 'WP Netzbezug mit Speicher'])

                df.at[timestep, 'Stand Batteriespeicher'] -= b_out
                df.at[timestep, 'WP Netzbezug mit Speicher'] -= b_out

            # Überschussstrom reicht aus, um Nachfrage der Wärmepumpe zu decken
        else:

            diff -= row['WP Netzbezug mit Speicher']
            df.at[timestep, 'WP Netzbezug mit Speicher'] = 0

            # Einspeicherung in Batteriespeicher
            b_in = min(0.9 * in_out_max,
                       0.9 * diff,
                       0.9 * b_max - df.at[timestep, 'Stand Batteriespeicher'])

            df.at[timestep, 'Stand Batteriespeicher'] += b_in
            diff -= 1.1 * b_in

            # restliches wird ggf. eingespeichert (Speicher wird innerhalb der Heizperiode immer voll gemacht)
            if df.loc[(timestep + pd.Timedelta(value='15min')):(timestep + pd.Timedelta(days=3)),
               'WP Netzbezug mit Speicher'].sum() > 0:
                w = min(row['COP'] * df.loc[:, 'WP Netzbezug mit Speicher'].max(),
                        row['COP'] * diff,
                        s_max - df.at[timestep, 'Stand Pufferspeicher'])  # produzierbare Wärme

                df.at[timestep, 'Stand Pufferspeicher'] += w
                diff -= w / row['COP']

        df.at[timestep, 'PV Netzeinspeisung'] = diff

        # Speicherstände werden in den nächsten Zeitschritt übernommen
        df.at[(timestep + pd.Timedelta(value='15min')), 'Stand Pufferspeicher'] = max(0, df.at[
            timestep, 'Stand Pufferspeicher'] - 0.02)
        df.at[(timestep + pd.Timedelta(value='15min')), 'Stand Batteriespeicher'] = 0.999 * df.at[
            timestep, 'Stand Batteriespeicher']
    return df


def prosumer_ts(state, jvlh, PV_peak, Wohnfläche, sv, s_vol, b_max, JAZ=3.7, spez_Wärm=60):
    """
	Creates prosumer timeseries for a single-family house with heat pump, pv, battery and buffer storage [kWh]
	
	Parameters
	----------
	state : str
		must be one of ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV',
                        'NI', 'NW', 'RP', 'SL', 'SN',' ST', 'SH', 'TH']
	jvlh : float
		annual full load hours of the state [kWh/kWp]
	PV_peak : float
		installed PV capacity [kWp]
	Wohnfläche : float
		living space of the household [m²]
	sv : float
		annual power comsumption of household [kWh]
	s_vol : float
		capacitiy of buffer storage [l]
	b_max : float
		maximum capacitiy of battery
	JAZ : float, default 3.7
		annual COP of air heat pump
	spez_Wärm : float, default 60
		specific heat demand of the household
		
	Returns
	-------
    pd.DataFrame
		index = datetimeindex
		columns = ['PV-Ertrag','COP','HH Netzbezug mit Speicher','HH Netzbezug ohne Speicher'
                   'Stand Batteriespeicher','WP Netzbezug mit Speicher','WP Netzbezug ohne Speicher',
				   'Stand Pufferspeicher','PV Netzeinspeisung']
	"""
    if state not in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', ' ST', 'SH', 'TH']:
        raise ValueError(
            "`state` must be in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV','NI', 'NW', 'RP', 'SL', 'SN',' ST', 'SH', 'TH']")

    # create PV timeseries
    PV_bl = create_PV_ts(jvlh, PV_peak)

    # create space heating and COP time series
    WP_ts, air_floor_cop = create_heat_ts_EFH(Wohnfläche, JAZ, spez_Wärm)
    ec_heat_ts_bl = WP_ts.loc[:, WP_ts.columns.astype(str).str[:-3].astype(int) ==
                                 list(bl_dict().keys())[list(bl_dict().values()).index(state)]]
    # air_floor_cop_bl = air_floor_cop.loc[:,air_floor_cop.columns.astype(str).str[:-3].astype(int) ==
    # list(bl_dict().keys())[list(bl_dict().values()).index(state)]]

    results_dict = {}

    # create timeseries for conventional power comsumption
    slp = CTS_power_slp_generator(state)
    HH_bl = sv * slp['H0']

    # Create storage timeseries using surplus PV energy
    for lk in ec_heat_ts_bl.columns:
        df = pd.DataFrame(data={'PV-Ertrag': PV_bl,
                                'COP': air_floor_cop[lk],
                                'HH Netzbezug mit Speicher': HH_bl,
                                'HH Netzbezug ohne Speicher': HH_bl,  # nur für Plot
                                'Stand Batteriespeicher': pd.Series(index=slp.index, data=0),
                                'WP Netzbezug mit Speicher': ec_heat_ts_bl[lk],
                                'WP Netzbezug ohne Speicher': ec_heat_ts_bl[lk],  # nur für plot
                                'Stand Pufferspeicher': pd.Series(index=slp.index, data=0),
                                'PV Netzeinspeisung': pd.Series(index=slp.index, data=0)})

        df = df.astype(np.float64)
        df = use_storage(df,
                         s_max=(s_vol * 1.163 * (40 - 35)) / 1000,
                         b_max=b_max)
        results_dict[lk] = df.iloc[:-1, :]

    return results_dict


def Eigenverbrauchsanteil(results_dict, lk):
    return (results_dict[lk]['PV-Ertrag'].sum() - results_dict[lk]['PV Netzeinspeisung'].sum()) / results_dict[lk][
        'PV-Ertrag'].sum()


def Autarkiegrad(results_dict, lk):
    return (results_dict[lk]['PV-Ertrag'].sum() - results_dict[lk]['PV Netzeinspeisung'].sum()) / (
            results_dict[lk]['HH Netzbezug ohne Speicher'].sum() + results_dict[lk][
        'WP Netzbezug ohne Speicher'].sum())


def prosumer_hh_bl(state, total=2140000, hh_size=[4], **kwargs):
    """
	Creates prosumer timeseries for one state with single-family houses with 
	heat pump, pv, battery and buffer storage [kWh]
	
	Parameters
	----------
	state : str
		must be one of ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV',
                        'NI', 'NW', 'RP', 'SL', 'SN',' ST', 'SH', 'TH']
	total : int, default 2140000
		total number of prosumer households in Germany in year (default for 2030)
	
	hh_size : list of int
		must be in [2,3,4,5]
		
	Returns
	-------
    pd.DataFrame
	    reductions in grid consumption for space heating application due to storage use
		index = datetimeindex
		columns = Districts
	"""

    if state not in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', ' ST', 'SH', 'TH']:
        raise ValueError(
            "`state` must be in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV','NI', 'NW', 'RP', 'SL', 'SN',' ST', 'SH', 'TH']")

    if any(size not in [2, 3, 4, 5] for size in hh_size):
        raise ValueError("hh_size must be in [2,3,4,5]")

    # read excel file with predefined household charateristics
    path = data_in('heat', 'Prosumer_HH.xlsx')
    BL_spez = pd.read_excel(path, index_col=0, header=0, sheet_name='BL')
    HH_spez = pd.read_excel(path, index_col=0, header=0, sheet_name='HH')

    # calculate number of prosumer hh in given state
    number_prosumer = int((BL_spez['Anzahl der Solarstromspeicher'][state] /
                           BL_spez['Anzahl der Solarstromspeicher'].sum()) * total)

    # calculate number of prosumer hh per hh size and lk
    hh_per_size = households_per_size(nuts_3=True)
    hh_per_size.rename(index=dict_region_code(level='lk', keys='natcode_nuts3',
                                              values='ags_lk'), inplace=True)
    hh_per_size.index.name = None
    hh_per_size_bl = hh_per_size.assign(über_5=hh_per_size[5] + hh_per_size[6]).drop([1, 5, 6], axis=1).rename(
        columns={'über_5': 5}).loc[hh_per_size.index.astype(str).str[:-3].astype(int) ==
                                   list(bl_dict().keys())[list(bl_dict().values()).index(state)], :]
    prosumer_bl = (
            hh_per_size_bl.loc[:, hh_size] / hh_per_size_bl.loc[:, hh_size].sum().sum() * number_prosumer).astype(
        int)

    # create results DataFrame and set datetime index
    year = kwargs.get('year', cfg['base_year'])
    if ((year % 4 == 0)
            & (year % 100 != 0)
            | (year % 4 == 0)
            & (year % 100 == 0)
            & (year % 400 == 0)):
        periods = 35136
    else:
        periods = 35040
    date = pd.date_range((str(year) + '-01-01'), periods=periods, freq='0.25H')
    WP_Netz = pd.DataFrame(columns=prosumer_bl.index, index=date, data=0)
    HH_Netz = pd.DataFrame(columns=prosumer_bl.index, index=date, data=0)
    PV_Ertrag = pd.DataFrame(columns=prosumer_bl.index, index=date, data=0)
    PV_Netz = pd.DataFrame(columns=prosumer_bl.index, index=date, data=0)

    # Creates prosumer timeseries per hh size
    for hh in hh_size:
        results_dict = prosumer_ts(jvlh=BL_spez.loc[state, 'durchschnittliche Jahresvolllaststunden'],
                                   PV_peak=HH_spez.loc[hh, 'PV Leistung'],
                                   Wohnfläche=HH_spez.loc[hh, 'Wohnfläche'],
                                   sv=HH_spez.loc[hh, 'Stromverbrauch'],
                                   s_vol=HH_spez.loc[hh, 'Speichervolumen'],
                                   b_max=HH_spez.loc[hh, 'Batteriespeicher'],
                                   state=state)

        # scale to total grid reduction
        for lk in results_dict.keys():
            WP_Netz.loc[:, lk] = WP_Netz[lk].add(
                prosumer_bl.loc[lk, hh] * (results_dict[lk]['WP Netzbezug mit Speicher'] - results_dict[lk][
                    'WP Netzbezug ohne Speicher']),
                axis=0)
            HH_Netz.loc[:, lk] = HH_Netz[lk].add(
                prosumer_bl.loc[lk, hh] * (results_dict[lk]['HH Netzbezug mit Speicher'] - results_dict[lk][
                    'HH Netzbezug ohne Speicher']),
                axis=0)
            PV_Ertrag.loc[:, lk] = PV_Ertrag[lk].add(prosumer_bl.loc[lk, hh] * (results_dict[lk]['PV-Ertrag']), axis=0)
            PV_Netz.loc[:, lk] = PV_Netz[lk].add(prosumer_bl.loc[lk, hh] * (results_dict[lk]['PV Netzeinspeisung']),
                                                 axis=0)
    return WP_Netz, HH_Netz, PV_Ertrag, PV_Netz


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


# -----------------------------------------------------------
# Following functions for heat demand and COP are completly or partly taken from
# https://github.com/oruhnau/when2heat (Ruhnau et al.)

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
    # The low-resolution values are applied to all high-resolution values up to the next low-resolution value
    # In particular, the last low-resolution value is extended up to where the next low-resolution value would be

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
    input_path = data_in('heat', 'ERA_temperature_' + str(year) + '.nc')

    # -----------------------------------------------
    # from read.weather (When2Heat)

    # read weather nc
    nc = Dataset(input_path)
    time = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    latitude = nc.variables['latitude'][:]
    longitude = nc.variables['longitude'][:]
    variable = nc.variables['stl4'][:]

    # Transform to pd.DataFrame
    df = pd.DataFrame(data=variable.reshape(len(time), len(latitude) * len(longitude)),
                      index=pd.Index(num2date(time, time_units), name='time'),
                      columns=pd.MultiIndex.from_product([latitude, longitude],
                                                         names=('latitude', 'longitude')))
    # ------------------------------------------------

    # upsample from 6h resolution
    df = upsample_df(df, '60min')

    # ------------------------------------------------
    # from plot.choropleth_map (disaggregator)

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


# from cop.spatial_cop (When2Heat)
def cop_curve(delta_t, source_type):
    """
	Creates cop timeseries based on temperature difference between source and sink
	
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

    cop_params = pd.read_csv(data_in('heat', 'cop_parameters.csv'), sep=';', decimal=',', header=0, index_col=0).apply(
        pd.to_numeric, downcast='float')
    delta_t.clip(lower=13, inplace=True)
    return sum(cop_params.loc[i, source_type] * delta_t ** i for i in range(3))


# ----------------------Plot-Functions---------------------------

def resample_t_allo(**kwargs):
    """
	Resamples the allocation temperature to match other dataframes in plot
	"""

    # get base year
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


def plot_gas_demand(tempinde, space, t_allo, lk=11000, start=0, end=365):
    """
	Plots the disaggregated gas demand
	
	Parameters
	----------
	tempinde : pd.DataFrame
	space : pd.DataFrame
	t_allo : pd.DataFrame
	lk : int, default 11000 (Berlin)
	start : int, default 0
		day of year at the beginning of the plotted interval
	end : int, default 365
		day of year at the end of the plotted interval
	"""

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    tempinde.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax)
    space.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax, figsize=(15, 5))
    t_allo.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax2, color='green')
    plt.axhline(y=13, color='r', linestyle='-')

    ax.set_ylabel('MWh')
    ax.set_ylim(0)
    ax2.set_ylabel('°C')
    fig.legend(['Gasnachfrage für PW, WW und ME', 'Gasnachfrage für RW', 'Allokationstemperatur', 'Heizgrenze'],
               loc='lower center', fontsize=15, ncol=4, bbox_to_anchor=(0.5, 0.9))

    plt.show()


def plot_source_temps(lk=11000, **kwargs):
    """
	Plots the heat source temperatures of water, air and ground
	Parameters
	----------
	lk : int, default 11000 (Berlin)

	"""

    # get base year
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

    # get source temperatures
    soil_t = soil_temp(year)
    soil_t.index = date

    air_t = ambient_T(year=year)
    air_t = change_nuts3_to_ags(air_t)
    air_t.index = date

    water_t = pd.DataFrame(index=soil_t.index, columns=soil_t.columns, data=5)

    # plot
    fig, ax = plt.subplots(figsize=(15, 4))
    water_t.loc[:, lk].plot(ax=ax)
    air_t.loc[:, lk].plot(ax=ax, zorder=1)
    soil_t.loc[:, lk].plot(ax=ax)
    fig.legend(['Grundwasser', 'Luft', 'Boden'],
               loc='lower center', fontsize=15, ncol=3, bbox_to_anchor=(0.5, 0.9))
    ax.set_ylabel('Temperatur [°C]')


def plot_COP():
    """
	Plots the COP depending on the temperature difference
	"""
    diff = pd.Series(data=range(0, 60))
    water_cop = cop_curve(diff, 'water')
    air_cop = cop_curve(diff, 'air')
    ground_cop = cop_curve(diff, 'ground')

    fig, ax = plt.subplots(figsize=(7, 4))

    water_cop.plot(ax=ax)
    air_cop.plot(ax=ax)
    ground_cop.plot(ax=ax)

    fig.legend(['Grundwasser', 'Luft', 'Boden'],
               loc='lower center', fontsize=15, ncol=3, bbox_to_anchor=(0.5, 0.9))
    ax.set_ylabel('COP')
    ax.set_xlabel('Temperaturdifferenz')


def plot_power_and_heat_pumps(ec_heat, start=0, end=365, **kwargs):
    """
	Plots the total power consumption and power consumption for heat pumps
	
	Parameters
	----------
	ec_heat : pd.DataFrame
	start : int, default 0
		day of year at the beginning of the plotted interval
	end : int, default 365
		day of year at the end of the plotted interval
	"""
    # get base year
    year = kwargs.get('year', cfg['base_year'])

    # get total power consumption
    path = data_out('sv_CTS_' + str(year) + '.csv')
    if os.path.exists(path):
        sv = pd.read_csv(path, index_col=0)
        sv.index = pd.to_datetime(sv.index)
    else:
        sv = disagg_temporal_power_CTS(detailed=False, use_nuts3code=False)
        sv.to_csv(data_out(path))

    # plot
    df = pd.DataFrame(data=[sv.resample('H').sum().sum(axis=1).iloc[start * 24:end * 24],
                            ec_heat.sum(axis=1).iloc[start * 24:end * 24]]).T
    df.columns = ['Gesamt', 'Raumwärme']
    df.plot(kind='area', stacked=False, figsize=(15, 4), alpha=1, ylabel='MWh')
    plt.legend(loc='lower center', fontsize=15, ncol=3, bbox_to_anchor=(0.5, 1))
    plt.show()


def plot_heat_pump_source(ec_heat_ground, ec_heat_air, ec_heat_water, start=0, end=365, lk=11000, **kwargs):
    """
	Plots the power consumption of heat pumps with different source types
	
	Parameters
	----------
	ec_heat_ground : pd.DataFrame
	ec_heat_air : pd.DataFrame
	ec_heat_water : pd.DataFrame
	start : int, default 0
		day of year at the beginning of the plotted interval
	end : int, default 365
		day of year at the end of the plotted interval
	lk : int, default 11000 (Berlin)
	"""
    fig, ax = plt.subplots(figsize=(15, 6), nrows=2, ncols=1)

    # get base year
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

    # get source temperatures
    soil_t = soil_temp(year)
    soil_t.index = date

    air_t = ambient_T(year=year)
    air_t = change_nuts3_to_ags(air_t)
    air_t.index = date

    water_t = pd.DataFrame(index=soil_t.index, columns=soil_t.columns, data=5)

    # plot
    ec_heat_water.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax[0])
    ec_heat_air.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax[0])
    ec_heat_ground.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax[0])

    water_t.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax[1])
    air_t.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax[1])
    soil_t.loc[:, lk].iloc[start * 24:end * 24].plot(ax=ax[1])

    fig.legend(['Grundwasser', 'Luft', 'Boden'],
               loc='lower center', fontsize=15, ncol=3, bbox_to_anchor=(0.5, 1))
    fig.tight_layout()

    ax[0].set_ylabel('Normierte Stromnachfrage der WP [kWh_el]', labelpad=25, fontsize=13)
    ax[1].set_ylabel('Temperatur [°C]', fontsize=13)


def plot_prosumer(results_dict, columns, start=0, end=365, lk=None):
    """
	Plots the results of the prosumer_ts function.
	
	Parameters
	----------
	results_dict : dict
	columns : list of str
		Columns of the results_dict that should be plotted.
		Must be in ['PV-Ertrag', 'COP', 'HH Netzbezug mit Speicher', 'HH Netzbezug ohne Speicher',
       'Stand Batteriespeicher', 'WP Netzbezug mit Speicher',
       'WP Netzbezug ohne Speicher', 'Stand Pufferspeicher', 'PV Netzeinspeisung']
	start : int, default 0
		day of year at the beginning of the plotted interval
	end : int, default 365
		day of year at the end of the plotted interval
	lk : int
		if not given, first lk of the results_dict keys
	"""

    # get first lk of the results_dict keys
    if type(lk) != int:
        lk = list(results_dict.keys())[0]

    if any(c not in results_dict[lk].columns for c in columns):
        raise ValueError(
            "columns must be in ['PV-Ertrag', 'COP', 'HH Netzbezug mit Speicher', 'HH Netzbezug ohne Speicher','Stand Batteriespeicher', 'WP Netzbezug mit Speicher','WP Netzbezug ohne Speicher', 'Stand Pufferspeicher', 'PV Netzeinspeisung']")

    # plot
    cmap = plt.cm.get_cmap('tab20')
    color_dict = {'PV-Ertrag': 'olive', 'COP': cmap(9 / 21), 'HH Netzbezug mit Speicher': cmap(2 / 21),  # cmap(17/21)
                  'HH Netzbezug ohne Speicher': cmap(1 / 21), 'Stand Batteriespeicher': cmap(7 / 21),
                  'WP Netzbezug mit Speicher': 'navajowhite', 'WP Netzbezug ohne Speicher': 'tab:orange',
                  # cmap(4/21),cmap(3/21),
                  'Stand Pufferspeicher': cmap(11 / 21), 'PV Netzeinspeisung': cmap(18 / 21)}

    fig, ax = plt.subplots(figsize=(15, 3))

    if any(c in ['Stand Pufferspeicher', 'Stand Batteriespeicher'] for c in columns):
        ax2 = ax.twinx()

    for c in columns:
        if c not in ['Stand Batteriespeicher', 'Stand Pufferspeicher']:
            results_dict[lk].loc[:, c].iloc[start * 96:end * 96].plot(ax=ax, color=color_dict[c])
        else:
            results_dict[lk].loc[:, c].iloc[start * 96:end * 96].plot(ax=ax2, color=color_dict[c])

    fig.legend(columns, loc='lower center', fontsize=15, ncol=len(columns), bbox_to_anchor=(0.5, 0.9))
    ax.set_ylabel('kWh')

    if all(c in ['Stand Pufferspeicher', 'Stand Batteriespeicher'] for c in columns):
        ax2.set_ylabel('kWh_el oder kWh_th')
    elif 'Stand Pufferspeicher' in columns:
        ax2.set_ylabel('kWh_th')
    elif 'Stand Batteriespeicher' in columns:
        ax2.set_ylabel('kWh_el')


def plot_heating_threshold(ec_heat, lk=11000, **kwargs):
    """
	Visualises the heat pumps electricity demand outside the heating threshold.
	
	Parameters
	----------
	ec_heat : pd.DataFrame
	lk : int, default 11000 (Berlin)
	"""
    # get allocation temperature
    t_allo = resample_t_allo()

    # filter for timesteps with temperature over heating threshold
    temp = t_allo.loc[:, lk].copy()
    temp.loc[temp > 13] = np.nan

    # filter for timesteps with heating demand except air temp being over 15°C
    RW_warm = ec_heat.loc[:, lk].copy()
    RW_kalt = ec_heat.loc[:, lk].copy()
    RW_warm.loc[temp.loc[temp.isnull()].index] = np.nan
    RW_warm.loc[RW_warm == 0] = np.nan
    RW_kalt.loc[RW_kalt == 0] = np.nan

    # plot
    fig, ax = plt.subplots(figsize=(15, 3))
    RW_kalt.plot(ax=ax)
    RW_warm.plot(ax=ax)
    plt.legend(['Allokationstemperatur größer als Heizgrenze', 'Allokationstemperatur kleiner als Heizgrenze'],
               loc='lower center', fontsize=15, ncol=2, bbox_to_anchor=(0.5, 1))
    plt.ylabel('MWh')
    plt.show()

    print(str(round((RW_kalt.sum() - RW_warm.sum()) * 100 / RW_kalt.sum(),
                    2)) + ' % der Wärmepumpen-Stromnachfrage ist außerhalb der Heizgrenze')
