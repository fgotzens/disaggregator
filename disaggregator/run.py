# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:19:30 2022

@author: Paul Verwiebe
"""
# %% Imports
from disaggregator.spatial import *
from disaggregator.heat import *
from disaggregator.temporal import *
from disaggregator.plot import *

import logging
logger = logging.getLogger(__name__)

# %% Parameters
years = range(2001, 2019, 1)
year = 2018
source = 'power'
sector = 'CTS'
sectors = ['CTS', 'industry']
sources = ['power', 'gas']
disagg_ph = True
use_nuts3code = False
no_self_gen = False
detailed=False

#test


# %% Script - Part 01: Demonstration
# Goal create Datasets for final plots

# create DataFrame for National results
multi_sectors = [sector for sector in sectors for source in sources]
multi_sources = list(sources)*len(sectors)

tuples = list(zip(*[multi_sectors, multi_sources]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Sektoren",
                                                       "Energieträger"])

df_results = pd.DataFrame(index=years, columns=multicolumn)

# create results 01, 02
for year in years:
    print("Working on year: " + str(year))
    for sector in sectors:
        for source in sources:
            df = disagg_applications_eff(source, sector, disagg_ph,
                                         use_nuts3code, no_self_gen, year=year)
            df_results.loc[year][sector, source] = df.sum().sum()
# save results
df_results.to_csv("../data_out/Diss/02_results_Bundesebene_2019_2050.csv")

# create DataFrame with regional results 03
multi_sectors = [sector for sector in sectors for source in sources]
multi_sources = list(sources)*len(sectors)

tuples = list(zip(*[multi_sectors, multi_sources]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Sektoren",
                                                       "Energieträger"])

df_results = pd.DataFrame(columns=multicolumn)
# create results 03
for sector in sectors:
    for source in sources:
        df = disagg_CTS_industry(source, sector, use_nuts3code=True,
                                 no_self_gen=False, year=year)[0].sum()
        df_results[sector, source] = df
# save results    
df_results.to_csv("../data_out/Diss/03_results_regio_2018.csv")

# create DataFrame with regional results 05

df_results_power = pd.DataFrame()
df_results_gas = pd.DataFrame()

# create results 04
df_results_power_cts = pd.DataFrame()
df_results_gas_cts = pd.DataFrame()
df_results_power_industry = pd.DataFrame()
df_results_gas_industry = pd.DataFrame()

# create results
# using blp
df_results_power_cts = disagg_temporal_power_CTS_blp(detailed,use_nuts3code,
                                                     year=year).sum(axis=1)
df_results_gas_cts = disagg_temporal_gas_CTS(detailed, use_nuts3code,
                                             year=year)

df_results_power_industry = disagg_temporal_industry_blp(source='power',
                                                         detailed=False,
                                                         use_nuts3code=False,
                                                         low=0.5,
                                                         no_self_gen=False,
                                                         year=year)
df_results_gas_industry = disagg_temporal_industry(source='gas',
                                                   detailed=False,
                                                   use_nuts3code=False,
                                                   low=0.5, no_self_gen=False,
                                                   year=year)

# for 
df_results_power_cts.to_csv("./data_out/Diss/04_results_lastgänge_regions_strom_cts_2018.csv")
df_results_gas_cts.to_csv("./data_out/Diss/04_results_lastgänge_regions_gas_cts_2018.csv")

df_results_power_industry.to_csv("./data_out/Diss/04_results_lastgänge_regions_strom_industry_2018.csv")
df_results_gas_industry.to_csv("./data_out/Diss/04_results_lastgänge_regions_gas_industry_2018.csv")

# ODER für 04 gas
df_results_gas_industry = pd.DataFrame()
df_results_gas_industry = disagg_temporal_applications(source='gas',
                                                       sector='industry',
                                                       detailed=False,
                                                       use_nuts3code=False,
                                                       use_slp_for_sh=True,
                                                       year=2018)

(df_results_gas_industry.sum(axis=1)).to_csv("../data_out/Diss/04_results_lastgänge_regions_gas_industry_2018_with_KO.csv")


# create results 05
for sector in sectors:
    for source in sources:
        df = disagg_CTS_industry(source, sector, use_nuts3code=True,
                                 no_self_gen=False, year=year)[0]
        if source == 'power':
            df_results_power = pd.concat([df_results_power, df])
        else:
            df_results_gas = pd.concat([df_results_gas, df])

df_max_gas = pd.concat([df_results_gas.max(), df_results_gas.idxmax()],
                       axis=1, keys=['Größter Gasverbrauch [MWh]', 'WZ'])

df_max_power = pd.concat([df_results_power.max(), df_results_power.idxmax()],
                         axis=1, keys=['Größter Stromverbrauch [MWh]', 'WZ'])

# save results    
df_max_gas.to_csv("../data_out/Diss/05_results_regio_gas_2018.csv")
df_max_power.to_csv("../data_out/Diss/05_results_regio_power_2018.csv")

# create DataFrame with regional results 07
df_results=pd.DataFrame()
df=pd.DataFrame()
top = 10

# create results 07
for sector in sectors:
    for source in sources:
        df = disagg_applications(source, sector, disagg_ph, use_nuts3code,
                                 no_self_gen, year=year)
        df_results = (df.sum()[df.sum().sum(level=0).nlargest(top).index]
                      .unstack())
        df_results.to_csv("../data_out/Diss/07_results_Anwendungen_WZ_2018_"
                          + str(sector) + "_" + str(source) + ".csv")
        
#.plot(kind='bar', stacked=True)

# create results 08
years = range(2015, 2051, 5)

multi_sectors = [sector for sector in sectors for year in years]
multi_years= list(years)*len(sectors)

tuples = list(zip(*[multi_years, multi_sectors]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "Sektoren"])

df_results = pd.DataFrame(columns=multicolumn)

for year in years:
    print("Working on year: " + str(year))
    for sector in sectors:
        df_results[year, sector] = sector_fuel_switch_fom_gas(sector=sector,
                                                              switch_to='power',
                                                              year=year).sum(axis=1)
df_results = df_results.rename(index=dict_region_code(keys='ags_lk',
                                         values='natcode_nuts3'))

df_results.to_csv("../data_out/Diss/08_cts_industry_gas_to_replace_with_elec_regions_years.csv")

# continuation from 08 create results 09
df_results_2 = pd.DataFrame(columns=multicolumn)
df_results_gas_left = pd.DataFrame(columns=multicolumn)

for year in years:
    print("Working on year: " + str(year))
    for sector in sectors:
        df_results_2[year, sector] = disagg_applications_eff(source='gas', sector=sector,
                                     disagg_ph=True, no_self_gen=False, use_nuts3code=True,
                                     year=year).sum(axis=1)
        df_results_gas_left[year, sector] = df_results_2[year, sector]-df_results[year, sector]

df_results_gas_left.to_csv("../data_out/Diss/09_cts_industry_gas_left_after_switch_regions_years.csv")

# results 10
years = range(2015, 2051, 5)
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetischer Erdgasverbrauch', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_sectors = [sector for sector in sectors for year in years]
multi_years= list(years)*len(sectors)

tuples = list(zip(*[multi_years, multi_sectors]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "Sektoren"])

# multi_app = list(apps)*len(sectors)

# tuples = list(zip(*[multi_sectors, multi_app]))
# multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Sektoren", "Anwendungen"])

df_results = pd.DataFrame(index=apps, columns=multicolumn)

# create results
for year in years:
    for sector in sectors:
            df_results[year, sector] = sector_fuel_switch_fom_gas(sector=sector,
                                                              switch_to='power',
                                                              year=year).groupby(level=1, axis=1).sum().sum()
df_results.to_csv("../data_out/Diss/10_results_sectors_gas_replaced_by_elec_applications_years.csv")

# results 11-13 CTS
years = [2020]
states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetischer Erdgasverbrauch', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "LK"])

df_results_app_cts = pd.DataFrame(index=apps, columns=multicolumn)
df_results_load_curve_cts = pd.DataFrame(columns=multicolumn)
df_results_gas_curve_cts = pd.DataFrame(columns=multicolumn)
# for cts gas--> elec
for year in years:
    df_fs_cts_to_power = sector_fuel_switch_fom_gas(sector='CTS', switch_to='power', year=year)
    df_results_regio_cts = pd.DataFrame(index= df_fs_cts_to_power.index, columns=multicolumn)
    df_results_wz_cts = pd.DataFrame(index=df_fs_cts_to_power.columns.unique(level=0), columns=multicolumn)

    for state in states:
        temporal_df_fs_cts_to_power = disagg_temporal_cts_fuel_switch(df_fs_cts_to_power, state=state, year=year)
        hp_temporal_df_fs_cts_to_power = temporal_cts_elec_load_from_fuel_switch(temporal_df_fs_cts_to_power)
        # saving results
        df_results_app_cts[year, state] = hp_temporal_df_fs_cts_to_power.groupby(level=[2], axis=1).sum().sum()
        df_results_wz_cts[year, state] = hp_temporal_df_fs_cts_to_power.groupby(level=[1], axis=1).sum().sum()
        df_results_regio_cts[year, state] = hp_temporal_df_fs_cts_to_power.groupby(level=[0], axis=1).sum().sum()
        df_results_load_curve_cts[year, state] = hp_temporal_df_fs_cts_to_power.sum(axis=1)
        df_results_gas_curve_cts[year, state] = temporal_df_fs_cts_to_power.sum(axis=1)

    df_results_app_cts.to_csv("../data_out/Diss/13_results_elec_from_switch_cts_applications_" + str(year) + ".csv")
    df_results_wz_cts.to_csv("../data_out/Diss/12_results_elec_from_switch_cts_wz_" + str(year) + ".csv")
    df_results_regio_cts.sum(axis=1).to_csv("../data_out/Diss/11_results_elec_from_switch_cts_regions_" + str(year) + ".csv")
    df_results_load_curve_cts.to_csv("../data_out/Diss/14_results_elec_from_switch_cts_load_curve_" + str(year) + ".csv")
    df_results_gas_curve_cts.to_csv("../data_out/Diss/14_results_gas_to_switch_cts_gas_curve_" + str(year) + ".csv")



# results 11-13 Industry
years = [2020]
states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetischer Erdgasverbrauch', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "LK"])

df_results_app_industry = pd.DataFrame(index=apps, columns=multicolumn)
df_results_load_curve_industry = pd.DataFrame(columns=multicolumn)
df_results_gas_curve_industry = pd.DataFrame(columns=multicolumn)
# for cts gas--> elec
for year in years:
    df_fs_industry_to_power = sector_fuel_switch_fom_gas(sector='industry', switch_to='power', year=year)
    df_results_regio_industry = pd.DataFrame(index=df_fs_industry_to_power.index, columns=multicolumn)
    df_results_wz_industry = pd.DataFrame(index=df_fs_industry_to_power.columns.unique(level=0), columns=multicolumn)

    for state in states:
        temporal_df_fs_industry_to_power = disagg_temporal_industry_fuel_switch(df_fs_industry_to_power, state=state, year=year)
        hp_temporal_df_fs_industry_to_power = temporal_industry_elec_load_from_fuel_switch(temporal_df_fs_industry_to_power)
        # saving results
        df_results_app_industry[year, state] = hp_temporal_df_fs_industry_to_power.groupby(level=[2], axis=1).sum().sum()
        df_results_wz_industry[year, state] = hp_temporal_df_fs_industry_to_power.groupby(level=[1], axis=1).sum().sum()
        df_results_regio_industry[year, state] = hp_temporal_df_fs_industry_to_power.groupby(level=[0], axis=1).sum().sum()
        df_results_load_curve_industry[year, state] = hp_temporal_df_fs_industry_to_power.sum(axis=1)
        df_results_gas_curve_industry[year, state] = temporal_df_fs_industry_to_power.sum(axis=1)

    df_results_app_industry.to_csv("../data_out/Diss/13_results_elec_from_switch_industry_applications_" + str(year) + ".csv")
    df_results_wz_industry.to_csv("../data_out/Diss/12_results_elec_from_switch_industry_wz_" + str(year) + ".csv")
    df_results_regio_industry.sum(axis=1).to_csv("../data_out/Diss/11_results_elec_from_switch_industry_regions_" + str(year) + ".csv")
    df_results_load_curve_industry.to_csv("../data_out/Diss/14_results_elec_from_switch_industry_load_curve_" + str(year) + ".csv")
    df_results_gas_curve_industry.to_csv("../data_out/Diss/14_results_gas_to_switch_industry_gas_curve_" + str(year) + ".csv")

# results 15 - CTS
years = range(2015, 2046, 5)
source='power'

df_results_elec_app_cts = pd.DataFrame(columns=years)
df_results_elec_wz_cts = pd.DataFrame(columns=years)
df_results_elec_regio_cts = pd.DataFrame(columns=years)

for year in years:
    df = disagg_applications_eff(source=source, sector='CTS', disagg_ph=True, use_nuts3code=True, no_self_gen=False, year=year)
    df_results_elec_wz_cts[year] = df.groupby(level=[0],axis=1).sum().sum()
    df_results_elec_app_cts[year] = df.groupby(level=[1],axis=1).sum().sum()
    df_results_elec_regio_cts[year] = df.sum(axis=1)

df_results_elec_wz_cts.to_csv("../data_out/Diss/15_cts_electricity_before_switch_years_wz.csv")
df_results_elec_app_cts.to_csv("../data_out/Diss/15_cts_electricity_before_switch_years_apps.csv")
df_results_elec_regio_cts.to_csv("../data_out/Diss/15_cts_electricity_before_switch_years_regions.csv")

# results 15 - industry
years = range(2015, 2046, 5)
source='power'

df_results_elec_app_industry = pd.DataFrame(columns=years)
df_results_elec_wz_industry = pd.DataFrame(columns=years)
df_results_elec_regio_industry = pd.DataFrame(columns=years)

for year in years:
    df = disagg_applications_eff(source=source, sector='industry', disagg_ph=True, use_nuts3code=True, no_self_gen=False, year=year)
    df_results_elec_wz_industry[year] = df.groupby(level=[0],axis=1).sum().sum()
    df_results_elec_app_industry[year] = df.groupby(level=[1],axis=1).sum().sum()
    df_results_elec_regio_industry[year] = df.sum(axis=1)

df_results_elec_wz_industry.to_csv("../data_out/Diss/15_industry_electricity_before_switch_years_wz.csv")
df_results_elec_app_industry.to_csv("../data_out/Diss/15_industry_electricity_before_switch_years_apps.csv")
df_results_elec_regio_industry.to_csv("../data_out/Diss/15_industry_electricity_before_switch_years_regions.csv")

# 17 Lastgang vor switch
years = [2025, 2035, 2045]

for year in years:
    df_results_app_industry_elec=disagg_temporal_applications(source='power', sector='industry', detailed=False,
                                     state=None, use_nuts3code=False, disagg_ph=True, use_blp=True, year=year).sum(level=1, axis=1)
    df_results_app_industry_elec.to_csv("../data_out/Diss/17_load_curve_before_switch_elec_industry_"+str(year)+".csv")
    
    df_results_app_industry_gas=disagg_temporal_applications(source='gas', sector='industry', detailed=False,
                                     state=None, use_nuts3code=False, disagg_ph=True, use_blp=False, year=year).sum(level=1, axis=1)
    df_results_app_industry_gas.to_csv("../data_out/Diss/17_load_curve_before_switch_gas_industry_"+str(year)+".csv")

    df_results_app_cts_elec=disagg_temporal_applications(source='power', sector='CTS', detailed=False,
                                     state=None, use_nuts3code=False, disagg_ph=True, use_blp=True, year=year).sum(level=1, axis=1)
    df_results_app_cts_elec.to_csv("../data_out/Diss/17_load_curve_before_switch_elec_cts_"+str(year)+".csv")
    
    df_results_app_cts_gas=disagg_temporal_applications(source='gas', sector='CTS', detailed=False,
                                     state=None, use_nuts3code=False, disagg_ph=True, use_blp=False, year=year).sum(level=1, axis=1)
    df_results_app_cts_gas.to_csv("../data_out/Diss/17_load_curve_before_switch_gas_cts_"+str(year)+".csv")

# 18 results
years = range(2020, 2046, 5)
sector = 'industry'
switch_to = 'hydrogen'
ets = ['Gas', 'Wasserstoff']

multi_years = [elem for elem in years for et in ets]
multi_et = list(ets)*len(years)
tuples = list(zip(*[multi_years, multi_et]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "Energieträger"])


df_results_regions = pd.DataFrame(columns=multicolumn)
df_results_apps = pd.DataFrame(columns=multicolumn)
df_results_wz = pd.DataFrame(columns=multicolumn)

for year in years:
    df = sector_fuel_switch_fom_gas(sector, switch_to, year=year)
    df_results_regions[year, 'Gas'] = df.sum(axis=1)
    df_results_apps[year, 'Gas'] = df.sum(level=1, axis=1).sum()
    df_results_wz[year, 'Gas'] = df.sum(level=0, axis=1).sum()
    
    df_hydro = hydrogen_after_switch(df)
    df_results_regions[year, 'Wasserstoff'] = df_hydro.sum(axis=1)
    df_results_apps[year, 'Wasserstoff'] = df_hydro.sum(level=1, axis=1).sum()
    df_results_wz[year, 'Wasserstoff'] = df_hydro.sum(level=0, axis=1).sum() 

df_results_regions.to_csv("../data_out/Diss/18_results_gas_replaced_hydro_industry_regions_years.csv")
df_results_apps.to_csv("../data_out/Diss/18_results_gas_replaced_hydro_industry_apps_years.csv")
df_results_wz.to_csv("../data_out/Diss/18_results_gas_replaced_hydro_industry_wz_years.csv")

# 19 - continuation of 18
col = pd.IndexSlice
df_result_electrolyzer_regions = ((df_results_regions.loc[:, col[:, 'Wasserstoff']]/0.7)
                                  .rename(columns={"Wasserstoff" : "Strombedarf Elektrolyse"}, level=1))
df_result_electrolyzer_apps = ((df_results_apps.loc[:, col[:, 'Wasserstoff']]/0.7)
                               .rename(columns={"Wasserstoff" : "Strombedarf Elektrolyse"}, level=1))
df_result_electrolyzer_wz = ((df_results_wz.loc[:, col[:, 'Wasserstoff']]/0.7)
                             .rename(columns={"Wasserstoff" : "Strombedarf Elektrolyse"}, level=1))

df_results_regions.to_csv("../data_out/Diss/19_results_electrolysis_industry_regions_years.csv")
df_results_apps.to_csv("../data_out/Diss/19_results_electrolysis_industry_apps_years.csv")
df_results_wz.to_csv("../data_out/Diss/19_results_electrolysis_industry_wz_years.csv")    

#  
# results 11-13 Industry GAS
from disaggregator.heat import *
years = [2025]
states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetischer Erdgasverbrauch', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "LK"])

df_results_gas_curve_industry = pd.DataFrame(columns=multicolumn)
# for cts gas--> elec
for year in years:
    df_fs_industry_to_power = sector_fuel_switch_fom_gas(sector='industry', switch_to='power', year=year)

    for state in states:
        temporal_df_fs_industry_to_power = disagg_temporal_industry_fuel_switch(df_fs_industry_to_power, state=state, year=year)

        # saving results
        df_results_gas_curve_industry[year, state] = temporal_df_fs_industry_to_power.sum(axis=1)

    df_results_gas_curve_industry.to_csv("../data_out/Diss/14_results_gas_to_switch_industry_gas_curve_" + str(year) + "_no_KO.csv")

years = [2035]
states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetischer Erdgasverbrauch', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "LK"])

df_results_gas_curve_industry = pd.DataFrame(columns=multicolumn)
# for cts gas--> elec
for year in years:
    df_fs_industry_to_power = sector_fuel_switch_fom_gas(sector='industry', switch_to='power', year=year)

    for state in states:
        temporal_df_fs_industry_to_power = disagg_temporal_industry_fuel_switch(df_fs_industry_to_power, state=state, year=year)

        # saving results
        df_results_gas_curve_industry[year, state] = temporal_df_fs_industry_to_power.sum(axis=1)

    df_results_gas_curve_industry.to_csv("../data_out/Diss/14_results_gas_to_switch_industry_gas_curve_" + str(year) + "_no_KO.csv")

years = [2045]
states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetischer Erdgasverbrauch', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "LK"])

df_results_gas_curve_industry = pd.DataFrame(columns=multicolumn)
# for cts gas--> elec
for year in years:
    df_fs_industry_to_power = sector_fuel_switch_fom_gas(sector='industry', switch_to='power', year=year)

    for state in states:
        temporal_df_fs_industry_to_power = disagg_temporal_industry_fuel_switch(df_fs_industry_to_power, state=state, year=year)

        # saving results
        df_results_gas_curve_industry[year, state] = temporal_df_fs_industry_to_power.sum(axis=1)

    df_results_gas_curve_industry.to_csv("../data_out/Diss/14_results_gas_to_switch_industry_gas_curve_" + str(year) + "_no_KO.csv")

# IDEE für CTS mal durchspielen für BE oder HH /MV
# industry fuel switch komplett für 2045 müsste doch 0 rauskommen oder nicht?
from disaggregator.temporal import*
from disaggregator.heat import *

sector='CTS'
switch_to='power'
source='gas'
year=2045
state='BE'
detailed=True
use_nuts3code=False

years = [2025, 2035,2045]
for year in years:
    df_gas_to_switch = disagg_temporal_fuel_switch_simple_cts('power', detailed=False,
                                                              year=year)
    df_gas_to_switch = df_gas_to_switch.loc[:, (df_gas_to_switch != 0).any(axis=0)]
    df_gas_to_switch.to_csv('../data_out/Diss/14_results_gas_to_switch_industry_gas_curve_' + str(year) +"_new_simple.csv")

# df_pre_swtich = disagg_temporal_applications(source, sector, detailed=False,
#                                  state=None, use_nuts3code=False,
#                                  disagg_ph=False, use_blp=False,
#                                  use_slp_for_sh=False, year=year)
df_pre_swtich_state = disagg_temporal_applications(source, sector, detailed=True,
                                  state=state, use_nuts3code=False,
                                  disagg_ph=False, use_blp=False,
                                  use_slp_for_sh=False, year=year)
df_pre_swtich_state=df_pre_swtich_state.loc[:, (df_pre_swtich_state != 0).any(axis=0)]

df_gas_to_switch = sector_fuel_switch_fom_gas(sector, switch_to, year=year)


df_gas_to_switch_lc = disagg_temporal_cts_fuel_switch(df_gas_to_switch, state='BE', year=year)
df_gas_to_switch_lc_h = df_gas_to_switch_lc.copy().resample('H').sum()

df_gas_to_switch_lc2 = disagg_temporal_fuel_switch_simple_cts('power', detailed=True,
                                                              state='BE', year=year)

df_gas_to_switch_lc2 = df_gas_to_switch_lc2.loc[:, (df_gas_to_switch_lc2 != 0).any(axis=0)]

(df_pre_swtich_state.sum(axis=1) - df_gas_to_switch_lc_h.sum(axis=1)).sum()

(df_pre_swtich_state.sum(axis=1) - df_gas_to_switch_lc2.sum(axis=1)).plot()

fig, ax = plt.subplots()
df_pre_swtich_state.sum(axis=1).plot(ax=ax)
df_gas_to_switch_lc2.sum(axis=1).plot(ax=ax)
#df_gas_to_switch_lc_h.sum(axis=1).plot(ax=ax)
#(df_pre_swtich_state.sum(axis=1) - df_gas_to_switch_lc_h.sum(axis=1)).plot(ax=ax)
(df_pre_swtich_state.sum(axis=1) - df_gas_to_switch_lc2.sum(axis=1)).plot(ax=ax)


[, df_gas_to_switch_lc2.sum(axis=1)].plot()
# IDEE mit KO mal durchspielen für BE oder HH
# industry fuel switch komplett für 2045 müsste doch 0 rauskommen oder nicht?
from disaggregator.temporal import*

sector='industry'
switch_to='power'
source='gas'
year=2045
state='MV'

df_ind_ps_KO = disagg_temporal_applications(source, sector, detailed=True,
                                 state=state, use_nuts3code=False,
                                 disagg_ph=True, use_blp=False, 
                                 use_slp_for_sh=True, year=year)

df_ind_fs_e = sector_fuel_switch_fom_gas(sector, switch_to, year=year)
df_ind_fs_e_c = disagg_temporal_industry_fuel_switch(df_ind_fs_e, state=state, year=year)
switch_to='hydrogen'
df_ind_fs_h = sector_fuel_switch_fom_gas(sector, switch_to, year=year)
df_ind_fs_h_c = disagg_temporal_industry_fuel_switch(df_ind_fs_h, state=state, year=year)

(df_ind_ps_KO.sum(axis=1)-df_ind_fs_e_c.sum(axis=1)-df_ind_fs_h_c.sum(axis=1)).plot(ylim=(-1,1))

df_ind_ps_KO.sum(axis=1).sum()-df_ind_fs_e_c.sum(axis=1).sum()-df_ind_fs_h_c.sum(axis=1).sum()


df_ind_ps_KO.loc[:, col[:,:,'Raumwärme']].sum(axis=1).plot()

(df_ind_ps_KO.groupby(level=2,axis=1).sum().sum()/df_ind_ps_KO.groupby(level=2,axis=1).sum().sum().sum())*100

df_ind_ps_KO.groupby(level=2,axis=1).sum().sum()/1000

df_ind_ps = pd.read_csv("../data_out/Diss/17_load_curve_before_switch_gas_industry_2045_new_with_KO.csv",
                        header=[0], index_col=[0])#.dropna().sum(axis=1)
from disaggregator.temporal import *
df_ind_ps_noKO = disagg_temporal_applications(source, sector, detailed=True,
                                 state='BE', use_nuts3code=False,
                                 disagg_ph=False, use_blp=False, year=year)

df_ind_ps_KO = disagg_temporal_applications(source, sector, detailed=True,
                                 state='BE', use_nuts3code=False,
                                 disagg_ph=False, use_blp=False, 
                                 use_slp_for_sh=True, year=year)
df_ind_ps_noKO.sum(axis=1).plot()
tw_water_norm.plot()
df_ind_fs_e = sector_fuel_switch_fom_gas(sector, switch_to, year=year)
df_ind_fs_e_c = disagg_temporal_industry_fuel_switch(df_ind_fs_e, state='BE', year=year)
df_ind_fs_h_c

sector='industry'
switch_to='power'
year=2035
df_gas_switch = sector_fuel_switch_fom_gas(sector, switch_to, year=year)
state='BE'



from disaggregator.temporal import *
df_ind_ps_noKO = disagg_temporal_applications(source, sector, detailed=False,
                                 state='BE', use_nuts3code=False,
                                 disagg_ph=False, use_blp=False, year=year)

df_ind_ps_KO = disagg_temporal_applications(source, sector, detailed=False,
                                 state='BE', use_nuts3code=False,
                                 disagg_ph=False, use_blp=False, 
                                 use_slp_for_sh=True, year=year)


df_app.loc[11000, col[:, 'Raumwärme']].sum()#.sum()
df_ind_ps_noKO.loc[:, col[11000, 'Raumwärme']].sum().sum()
df_ind_ps_KO.loc[:, col[11000, 'Raumwärme']].plot()

(df_app.groupby(level=1, axis=1).sum().sum()/1000000).div(df_app.groupby(level=1, axis=1).sum().sum().sum()/1000000)*100


neu=pd.DataFrame(index=test.index, columns=test.columns, dtype='float')
gd_region_sh = new_df.loc[:, col[:, :, ['Raumwärme']]].sum()
(gd_region_sh[13003].values).dot((heat_norm[13003]))

######

gd_regio_sh = new_df.loc[:, col[:, :, ['Raumwärme']]]
gd_regio_sh_sum = gd_regio_sh.sum()
for lk in test.columns.get_level_values(0):
    gd_regio_sh.loc[:][lk] = pd.DataFrame(data=(np.outer(heat_norm[lk].values,
                                                  gd_regio_sh_sum[lk].values)), 
             columns=gd_regio_sh[lk].columns, index=heat_norm.index)
    
gd_regio_sh.sum().sum()
new_df.loc[:, col[:, :, ['Raumwärme']]].sum().sum()

######
(np.outer(heat_norm.values,gd_region_sh.values))
 
 
for lk in test.columns.get_level_values(0):
    

# %% tests

sector= 'industry'
switch_to = 'power'
year = 2045

df_gas_switch_2020 = sector_fuel_switch_fom_gas(sector, switch_to, year=year)

df_gas_switch_2015 = sector_fuel_switch_fom_gas(sector, switch_to, year=year)

PATH = data_in("dimensionless", "fuel_switch_keys.xlsx")
df_electrode = pd.read_excel(PATH,
                                 sheet_name=("Gas2Power industry electrode"))
df_electrode = (df_electrode
                    .loc[[isinstance(x, int) for x in df_electrode["WZ"]]]
                    .set_index("WZ")
                    .copy())
col = pd.IndexSlice

df_PW = df_gas_switch_2015.loc[:, col[:, 'Prozesswärme 100°C-200°C']]

df_PW.groupby(axis=1, level=0).sum().sum().multiply(1-df_electrode.T).T.sum().sum()/1000000

df_gas_switch_2015.sum().sum()/10000000

air_floor_cop, ground_floor_cop, water_floor_cop = cop_ts(sink_t=40,
                                                              source='ambient',
                                                              year=year)
air_floor_cop1, ground_floor_cop1, water_floor_cop1 = cop_ts(sink_t=40,
                                                              source='ambient',
                                                              year=year, cf=1)


   ###############################################
   

# results 11-13 Industry
years = [2045]
#years = [2020]

#states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
#         'BB', 'MV', 'SN', 'ST', 'TH']
states = ['BE']
apps = ['Industriekraftwerke', 'Mechanische Energie',
       'Nichtenergetische Nutzung', 'Prozesswärme','Prozesswärme 100°C-200°C',
       'Prozesswärme 200°C-500°C', 'Prozesswärme <100°C',
       'Prozesswärme >500°C', 'Raumwärme', 'Warmwasser']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "LK"])

df_results_app_industry = pd.DataFrame(index=apps, columns=multicolumn)
df_results_app_industry_hp=pd.DataFrame()
df_results_wz_industry_hp=pd.DataFrame()
df_results_regio_industry_hp=pd.DataFrame()
df_results_app_industry_elec=pd.DataFrame()
df_results_wz_industry_elec=pd.DataFrame()
df_results_regio_industry_elec=pd.DataFrame()


# for cts gas--> elec
for year in years:
    df_results_load_curve_industry = pd.DataFrame(columns=states)
    df_results_gas_curve_industry = pd.DataFrame(columns=states)

    df_fs_industry_to_power = sector_fuel_switch_fom_gas(sector='industry', switch_to='power', year=year)
    df_results_regio_industry = pd.DataFrame(index=df_fs_industry_to_power.index, columns=multicolumn)
    df_results_wz_industry = pd.DataFrame(index=df_fs_industry_to_power.columns.unique(level=0), columns=multicolumn)

    for state in states:
        temporal_df_fs_industry_to_power = disagg_temporal_industry_fuel_switch(df_fs_industry_to_power, state=state, year=year)
        hp_temporal_df_fs_industry_to_power = temporal_industry_elec_load_from_fuel_switch(temporal_df_fs_industry_to_power)[0]
        hp_temporal_df_fs_industry_to_power_hp = temporal_industry_elec_load_from_fuel_switch(temporal_df_fs_industry_to_power)[1]
        hp_temporal_df_fs_industry_to_power_elec = temporal_industry_elec_load_from_fuel_switch(temporal_df_fs_industry_to_power)[2]


        # saving results
        df_results_app_industry[year, state] = hp_temporal_df_fs_industry_to_power.groupby(level=[2], axis=1).sum().sum()
        df_results_wz_industry[year, state] = hp_temporal_df_fs_industry_to_power.groupby(level=[1], axis=1).sum().sum()
        df_results_regio_industry[year, state] = hp_temporal_df_fs_industry_to_power.groupby(level=[0], axis=1).sum().sum()
       
        # saving results
        df_results_app_industry_hp[state] = hp_temporal_df_fs_industry_to_power_hp.groupby(level=[2], axis=1).sum().sum()
        df_results_wz_industry_hp = hp_temporal_df_fs_industry_to_power_hp.groupby(level=[1], axis=1).sum().sum()
        df_results_regio_industry_hp = hp_temporal_df_fs_industry_to_power_hp.groupby(level=[0], axis=1).sum().sum()
       
       
        # saving results
        df_results_app_industry_elec = hp_temporal_df_fs_industry_to_power_elec.groupby(level=[2], axis=1).sum().sum()
        df_results_wz_industry_elec = hp_temporal_df_fs_industry_to_power_elec.groupby(level=[1], axis=1).sum().sum()
        df_results_regio_industry_elec = hp_temporal_df_fs_industry_to_power_elec.groupby(level=[0], axis=1).sum().sum()
       



   
   ########################
col = pd.IndexSlice

years = range(2020, 2046, 5)
year = 2015
sector = 'industry'
switch_to = 'hydrogen'
 
df_hydro_switch = sector_fuel_switch_fom_gas(sector, switch_to, year=year)

df_hydro = df_hydro_switch.copy()
df_hydro.loc[:, col[:, 'Nichtenergetische Nutzung']] = (
        df_hydro.loc[:, col[:, 'Nichtenergetische Nutzung']]
        * (get_efficiency_level('Nichtenergetische Nutzung')))


source ='power'
sector='industry'
disagg_ph=False
use_nuts3code=False
no_self_gen=False
year=2018

df_app_2018_ind = disagg_applications_eff(source, sector, disagg_ph=False,
                            use_nuts3code=False, no_self_gen=False, year=year)

df_app_2018_cts = disagg_applications_eff(source, sector='CTS', disagg_ph=False,
                            use_nuts3code=False, no_self_gen=False, year=year)

df_share = df_app_2018.sum().groupby(level=0).sum()/(df_app_2018.sum().sum())

df_wz = disagg_CTS_industry(source, sector,
                        use_nuts3code=False, no_self_gen=False, year=year)

col = pd.IndexSlice

years = [2025, 2035, 2045]

year=2045
states = ['SH', 'HH', 'NI', 'HB', 'NW', 'HE', 'RP', 'BW', 'BY', 'SL', 'BE',
         'BB', 'MV', 'SN', 'ST', 'TH']

multi_states = [elem for elem in states for year in years]
multi_years = list(years)*len(states)
tuples = list(zip(*[multi_years, multi_states]))
multicolumn = pd.MultiIndex.from_tuples(tuples, names=["Jahre", "BL"])



# %% OLD Ideas



