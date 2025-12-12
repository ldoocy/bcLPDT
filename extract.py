"""
@author: Lauren Doocy, Nathan Tefft

This script extracts data from the raw FARS data files to be used for Levitt & Porter (2001) replication and bias correction.
Selected variables are included, and the data definitions are harmonized across years.
Accident, vehicle, and person dataframes are constructed and stored in csv files for later use in the replication.

This script has been validated for FARS datasets from 1982 to 2023.
The crash matching process required for bias correction requires a one year crash buffer
-- the extract.py must include +/- 1 years of crash data beyond the indended years for the estimate.py analysis.
"""

import os, numpy, pandas, shutil, us, zipfile, utils, time, random, math
from timezonefinder import TimezoneFinder
from geopy import distance

firstYear = 1982
lastYear = 2023

find_closest= True # perform nearest crash search for all single vehicle crashes
num_closest = 5 # if find_closest = True, define set of closest crashes for each crash

# load US state abbreviations for later merge
df_states = pandas.DataFrame.from_dict(us.states.mapping('fips', 'abbr'),orient='index',columns=['state_abbr'])
df_states.loc[11] = ['DC'] # manually add DC to list
df_states = df_states[df_states.index.notnull()]
df_states.index = df_states.index.astype(int)

# time zone, city, county state, reference datasets
tz_ref = pandas.read_csv('data/tz_bystate.csv', index_col='state_abbr')
dst_ref = pandas.read_csv('data/daylight_savings.csv', index_col='year')
county_ref = pandas.read_csv('data/county_data.csv', index_col='county')
city_ref = pandas.read_csv('data/city_data.csv', index_col='city')
state_ref = pandas.read_csv('data/nbr_state.csv', index_col='state_abbr')

# Initialize analytic dataframe for the crash, vehicle, and person datasets
fars_datasets = ['accident', 'vehicle', 'person', 'Miper']
dataset_ids = ['st_case','veh_no','per_no', 'per_no']

df_list={}
for dataset in fars_datasets:
    df_list[dataset]=pandas.DataFrame()

for yr in range(firstYear,lastYear+1): 
    print('Extracting data from ' + str(yr) + '.' )

    # extract accident, vehicle, person, and multiple imputation files
    zipfile.ZipFile('data/FARS' + str(yr) + 'NationalCSV.zip', 'r').extractall(path='data/extracted')
    filepath = 'data/extracted/accident.csv' #correct extracting issue with some years
    if os.path.isfile(filepath):
        filepath = 'data/extracted/'
    else:
        filepath = 'data/extracted/FARS' + str(yr) + 'NationalCSV/'

    df_list_yr={}
    index_list=['crash_id']
    for (dataset, id) in zip(fars_datasets,dataset_ids):
        file = open(filepath + dataset + '.csv', errors='ignore')
        df_list_yr[dataset]=pandas.read_csv(file,low_memory=False)
        file.close()
    
        df_list_yr[dataset].columns = df_list_yr[dataset].columns.str.lower() # make all columns lowercase
        df_list_yr[dataset]['year'] = numpy.full(len(df_list_yr[dataset].index), yr) # standardize the year variable to 4 digits
        df_list_yr[dataset]['crash_id'] = str(yr) + df_list_yr[dataset]['st_case'].astype(str) # provide standard crash id

        if dataset not in ['accident','Miper']:
            index_list.append(id)
        df_list_yr[dataset][index_list] = df_list_yr[dataset][index_list].astype('int') # set the indices as integers
        df_list_yr[dataset].set_index(index_list, inplace=True) # set the multiindex
        df_list_yr[dataset].index.set_names(index_list, inplace=True)

    shutil.rmtree(path='data/extracted') # clean up temporary extractions folder
    
    # Modify missing data
    df_list_yr['accident'].loc[df_list_yr['accident'].hour==99, 'hour'] = numpy.nan
    df_list_yr['accident'].loc[df_list_yr['accident'].hour==24, 'hour'] = 0
    df_list_yr['accident'].loc[df_list_yr['accident'].day_week==9, 'day_week'] = numpy.nan

    if yr <= 1990: # add pedestrian variable
        df_list_yr['accident']['peds'] = numpy.nan
    if yr <= 2000: # add lat/long variable
        df_list_yr['accident']['latitude'] = numpy.nan
        df_list_yr['accident']['longitud'] = numpy.nan
    else:
        df_list_yr['accident'].loc[df_list_yr['accident'].latitude.isin([77.7777,88.8888,99.9999]), 'latitude'] = numpy.nan
        df_list_yr['accident'].loc[df_list_yr['accident'].longitud.isin([777.7777,888.8888,999.9999]), 'longitud'] = numpy.nan

    # Manipulating accident data
    df_list_yr['accident'] = df_list_yr['accident'][df_list_yr['accident'].ve_forms.isin(range(1,3))] # remove accidents not involving 1 or 2 vehicles
    df_list_yr['accident']['quarter'] = numpy.ceil(df_list_yr['accident']['month']/3) # create quarter variable
    df_list_yr['accident'] = df_list_yr['accident'].merge(df_states,how='left',left_on='state',right_index=True) # merge in state abbreviations
    df_list_yr['accident']['day_type'] = 'weekday' # initiate all days at weekday
    df_list_yr['accident'].loc[df_list_yr['accident'].day_week.isin([1,7]), 'day_type'] = 'weekend' # set weekend on Saturday/Sunday
    df_list_yr['accident'].loc[(df_list_yr['accident'].day_week == 2) & (df_list_yr['accident'].hour < 5), 'day_type'] = 'weekend' # set early Monday weekend
    df_list_yr['accident'].loc[(df_list_yr['accident'].day_week == 6) & (df_list_yr['accident'].hour >= 20), 'day_type'] = 'weekend' # set late Friday weekend
    df_list_yr['accident'].loc[df_list_yr['accident'].hour.isin(range(5,12)), 'time_day'] = 'morning' # set time of day hours - morning
    df_list_yr['accident'].loc[df_list_yr['accident'].hour.isin(range(12,20)), 'time_day'] = 'afternoon' # set afternoon crashes
    df_list_yr['accident'].loc[(df_list_yr['accident'].hour.isin(range(0,5))) | (df_list_yr['accident'].hour.isin(range(20,24))), 'time_day'] = 'night' # set night crashes

    # create ids for approximate location merging
    df_list_yr['accident']['county_id'] = df_list_yr['accident'].state_abbr.astype(str) + df_list_yr['accident'].county.astype(str)
    df_list_yr['accident']['city_id'] = df_list_yr['accident'].state_abbr.astype(str) + df_list_yr['accident'].city.astype(str)
    city_ref['id'] = city_ref['state_abbr'].astype(str) + city_ref.index.astype(str)
    county_ref['id'] = county_ref['state_abbr'].astype(str) + county_ref.index.astype(str)

    # merge approximate locations and timezones 
    df_list_yr['accident'] = df_list_yr['accident'].reset_index().merge(city_ref.drop(columns=['state','state_abbr','city_name', 'county','county_name']).rename(columns={'latitude':'approx_latitude', 'longitud':'approx_longitud'}),how='left',left_on='city_id',right_on='id').set_index(df_list_yr['accident'].index.names)
    df_list_yr['accident'] = df_list_yr['accident'].reset_index().merge(county_ref[['id','latitude','longitud','timezone','RUCC_'+str(math.floor(yr/10)*10)]].rename(columns={'latitude':'approx_latitude2', 'longitud':'approx_longitud2', 'RUCC_'+str(math.floor(yr/10)*10):'rucc'}),how='left',left_on='county_id',right_on='id').set_index(df_list_yr['accident'].index.names)
    df_list_yr['accident']['approx_latitude'] = df_list_yr['accident'].approx_latitude.fillna(df_list_yr['accident'].approx_latitude2) # fill in county information for rural crashes/no city reported
    df_list_yr['accident']['approx_longitud'] = df_list_yr['accident'].approx_longitud.fillna(df_list_yr['accident'].approx_longitud2)
    df_list_yr['accident']['latitude'] = df_list_yr['accident'].latitude.fillna(df_list_yr['accident'].approx_latitude) # replace missing location with city/county centroid
    df_list_yr['accident']['longitud'] = df_list_yr['accident'].longitud.fillna(df_list_yr['accident'].approx_longitud)
    
    # approximate distance between provided and approximate location
    df_list_yr['accident']['approx_dist'] = utils.approximate_distance(df_list_yr['accident']['latitude'],df_list_yr['accident']['longitud'],df_list_yr['accident']['approx_latitude'],df_list_yr['accident']['approx_longitud'])
    df_list_yr['accident'] = df_list_yr['accident'].loc[df_list_yr['accident'].approx_dist<150] # remove crashes further than 150 miles from approx location
    df_list_yr['accident'] = df_list_yr['accident'].loc[(df_list_yr['accident'].approx_dist<100) | (df_list_yr['accident'].state_abbr.isin(['AZ','CA','NV']))]

    # update timezones for crashes occuring in counties with nonuniform time zones
    df_list_yr['accident'].loc[df_list_yr['accident'].state_abbr == 'AK', 'timezone'] = 'AKST' # manually set Alaska timezone to avoid incorporation issues
    df_list_yr['accident'].loc[(df_list_yr['accident'].state_abbr == 'AK') & (df_list_yr['accident'].latitude < 55) & (df_list_yr['accident'].longitud < -169.5), 'timezone'] = 'HST'
    multi_counties = df_list_yr['accident'][(df_list_yr['accident'].timezone=='MULTI') | df_list_yr['accident'].timezone.isna()][['latitude', 'longitud']]
    multi_counties['timezone_b'] = multi_counties.apply(lambda x: utils.get_timezone(x['latitude'], x['longitud']), axis=1)
    df_list_yr['accident'] = df_list_yr['accident'].merge(multi_counties.drop(columns=['latitude', 'longitud']),how='left',left_index=True,right_index=True)
    df_list_yr['accident'].loc[df_list_yr['accident'].timezone=='MULTI','timezone'] = numpy.nan
    df_list_yr['accident']['timezone'] = df_list_yr['accident'].timezone.fillna(df_list_yr['accident'].timezone_b)
    df_list_yr['accident']['standard_hour'] = df_list_yr['accident'].apply(lambda x: utils.univeral_time(x['hour'], x['timezone'],x['month'], x['day'], list(dst_ref.loc[yr])), axis=1) # create universal time for comparing accross time zones

    # keep relevant crash variables and append to accident dataframe
    df_list_yr['accident'] = df_list_yr['accident'][['year','state_abbr','county','city','peds','timezone','month','day','quarter','day_week','day_type','time_day',
                                                    'hour','minute','standard_hour','rucc','ve_forms','persons','latitude','longitud']]
    print('Count of crashes: ' + str(len(df_list_yr['accident'])))
    df_list['accident'] = pandas.concat([df_list['accident'],df_list_yr['accident']])

    # Manipulating vehicle data
    if yr <= 2008: 
        df_list_yr['vehicle']['occupants'] = df_list_yr['vehicle']['ocupants']
    else:
        df_list_yr['vehicle']['occupants'] = df_list_yr['vehicle']['numoccs']
    if yr <= 2015:
        df_list_yr['vehicle'].loc[df_list_yr['vehicle'].occupants>=99, 'occupants'] = numpy.nan
    else:
        df_list_yr['vehicle'].loc[df_list_yr['vehicle'].occupants>=97, 'occupants'] = numpy.nan 
        
    if yr <= 2017:
        for vt in ['acc','sus','dwi','spd','oth']:
            df_list_yr['vehicle'].loc[df_list_yr['vehicle']['prev_' + vt] > 97, 'prev_' + vt] = numpy.nan # previous violations
    else:
        for vt in ['acc','sus1','sus2','sus3','dwi','spd','oth']:
            df_list_yr['vehicle'].loc[df_list_yr['vehicle']['prev_' + vt] > 97, 'prev_' + vt] = numpy.nan # include all subdivisions of suspension

    # keep relevant vehicle variables and append to vehicle dataframe
    if yr <= 2017:
        df_list_yr['vehicle'] = df_list_yr['vehicle'][['prev_acc','prev_sus','prev_dwi','prev_spd','prev_oth','dr_drink','occupants']]
    else:
        sus_subdiv = ['prev_sus1', 'prev_sus2', 'prev_sus3']
        df_list_yr['vehicle']['prev_sus'] = df_list_yr['vehicle'][sus_subdiv].sum(axis=1) # combine all suspension subdivisions into one variable
        df_list_yr['vehicle'] = df_list_yr['vehicle'][['prev_acc','prev_sus','prev_dwi','prev_spd','prev_oth','dr_drink','occupants']] # add vehicle information, year
    print('Count of vehicles: ' + str(len(df_list_yr['vehicle'])))
    df_list['vehicle'] = pandas.concat([df_list['vehicle'],df_list_yr['vehicle']])

    # Manipulating person variables
    if yr <= 1990: #standardize alcohol test result
        df_list_yr['person']['alcohol_test_result'] = df_list_yr['person']['test_res']
    else:
        df_list_yr['person']['alcohol_test_result'] = df_list_yr['person']['alc_res']
    
    if yr >= 2015:
        df_list_yr['person']['alcohol_test_result'] = df_list_yr['person']['alcohol_test_result']/10
    
    df_list_yr['person'].loc[df_list_yr['person'].alcohol_test_result>=95, 'alcohol_test_result'] = numpy.nan    

    for vn in ['alc_det','atst_typ','race']: # create variables if necessary
        if vn not in df_list_yr['person'].columns:
            df_list_yr['person'][vn] = numpy.nan
    
    if yr <= 2008:
        df_list_yr['person'].loc[df_list_yr['person'].age==99, 'age'] = numpy.nan # age
    else:
        df_list_yr['person'].loc[df_list_yr['person'].age>=998, 'age'] = numpy.nan # age
    
    df_list_yr['person'] = df_list_yr['person'][df_list_yr['person'].ve_forms.isin(range(1,3))] # remove people inolved in 3+ vehicle crashes
    df_list_yr['person']['age_lt15'] = df_list_yr['person']['age'] < 15 # less than 15 defined as child for our purposes
    df_list_yr['person'].loc[df_list_yr['person'].sex.isin([8,9]), 'sex'] = numpy.nan # sex
    df_list_yr['person'].loc[df_list_yr['person'].race==99, 'race'] = numpy.nan # race
    df_list_yr['person'].loc[df_list_yr['person'].seat_pos>=98, 'seat_pos'] = numpy.nan # seat position
    
    # clean mulptiple imputation variables, e.g. harmonize names, ensure correct datatypes, and record missing variables 
    df_list_yr['Miper'] = df_list_yr['Miper'].rename(columns={'p1':'mibac1','p2':'mibac2','p3':'mibac3','p4':'mibac4','p5':'mibac5','p6':'mibac6','p7':'mibac7','p8':'mibac8','p9':'mibac9','p10':'mibac10'}) # rename bac columns
    df_list_yr['Miper']['seat_pos'] = 11  
    df_list_yr['person'] = df_list_yr['person'].merge(df_list_yr['Miper'].drop(columns=['year','st_case']),how='left',on=['crash_id','veh_no','seat_pos']) # merge multiply imputed bac values into person dataframe

    # keep relevant person variables and append to person dataframe
    df_list_yr['person'] = df_list_yr['person'][['year','ve_forms','seat_pos','drinking','alc_det','atst_typ','alcohol_test_result','race','age','age_lt15','sex','mibac1','mibac2','mibac3','mibac4','mibac5','mibac6','mibac7','mibac8','mibac9','mibac10']]
    print('Count of persons: ' + str(len(df_list_yr['person'])))
    df_list['person'] = pandas.concat([df_list['person'],df_list_yr['person']])

# find closest crashes to all single vehicle crashes for bias corrected values
if find_closest == True:
    crash_header = []
    for col in range(1, num_closest+1):
        crash_header.append('closest_crash' + str(col))

    single_veh_crash = df_list['accident'][df_list['accident']['ve_forms'] == 1] #remove multicar crashes
    single_veh_crash = single_veh_crash[single_veh_crash['latitude'].notnull()] # remove crash without location info
    single_veh_crash = single_veh_crash[single_veh_crash['hour'].notnull()] # remove crashes without timestamp

    for yr in range(firstYear+1, lastYear):
        print('Finding closest one vehicle crashes in', yr)

        crash_yr = single_veh_crash[single_veh_crash['year'] == yr]
        crash_buffer = single_veh_crash[(single_veh_crash['year'] <= yr+1) & (single_veh_crash['year'] >= yr-1)] # keep crash in 1 year buffer

        dist_yr = []
        neighbor_none = 0
        for crash, crash_info in crash_yr.iterrows():
            state, county = crash_info['state_abbr'], crash_info['county']
            counties = county_ref[county_ref['state_abbr'] == state]
            try:
                nbr_st, nbr_counties = counties.loc[county]['nbr_st'].split(';'), counties.loc[county]['nbr_counties'].split(';')
            except: # include entire state for unincorperated counties
                nbr_st, nbr_counties = [state], [county]
            # create geographic county radius, remove crashess ourside of time boundary week/weekend day, +/- 1 year, +/- 1 hour
            crash_radius = utils.create_crash_radius(crash, crash_buffer, state, county, nbr_st, nbr_counties) #dataset of crashes in county radius
            crash_radius = utils.find_similar(crash, crash_info, crash_radius) # remove crashes outside of time buffer
            closest_crash, dist = utils.find_closest(crash_info['latitude'], crash_info['longitud'], crash_radius, num_closest)

            if dist > 10 or len(crash_radius.index) < num_closest: # check wider radius for crashes with large distance to nearest crash
                nbr_st = state_ref.loc[state, 'nbr_st'].split(';')
                crash_radius = utils.create_statewide_radius(crash, crash_buffer, crash_info['state_abbr'], nbr_st)
                crash_radius = utils.find_similar(crash, crash_info, crash_radius)
                closest_crash, dist = utils.find_closest(crash_info['latitude'], crash_info['longitud'], crash_radius, num_closest)

            dist_yr.append(dist) # create list of distance to nearest crash for summary purposes
            df_list['accident'].loc[crash,'closest_dist'] = dist
            df_list['accident'].loc[crash,crash_header] = closest_crash

# summarize the constructed dataframes and save to csv files
if not os.path.exists('summary_data'):
    os.makedirs('summary_data')
for dfn in ['accident', 'vehicle', 'person']:
    print('Describing dataframe ' + dfn)
    print(df_list[dfn].describe())
    df_list[dfn].to_csv('summary_data/df_' + dfn + '.csv')

# Delete dictionaries so garbage collector releases memory from dataframes
df_list.clear()
df_list_yr.clear()