import pandas, numpy, time, us, random, math, warnings, sklearn
from timezonefinder import TimezoneFinder
from geopy import distance
from math import radians
from statsmodels.base.model import GenericLikelihoodModel

# more efficient, slightly less accurate distance approximation
def approximate_distance(lat1, lon1, lat2, lon2):
	lat1, lat2, lon1, lon2 = lat1.apply(lambda x : radians(x)),lat2.apply(lambda x : radians(x)),lon1.apply(lambda x : radians(x)),lon2.apply(lambda x : radians(x))
	dlat = lat2-lat1
	dlon = lon2-lon1
	return 2 * 3956 * numpy.arcsin(numpy.sqrt(numpy.sin(dlat/2)**2 + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(dlon/2)**2))


def get_timezone(latitude, longitud):
	tz_code = {'America/New_York' : 'EST', 'America/Detroit' : 'EST', 'America/Chicago' : 'CST','America/Denver' : 'MST','America/Phoenix' : 'AZST', 'America/Boise' : 'MST','America/Los_Angeles' : 'PST'}
	tzf = TimezoneFinder()
	tz = tzf.timezone_at(lng=longitud, lat=latitude)
	return tz_code[tz]


def univeral_time(hour, timezone, month, day, dst_ref):
	tz_standard = {'EST': 0, 'CST': 1, 'MST': 2, 'PST': 3, 'AKST': 4, 'HST': 5, 'NA': 0}
	if timezone == 'AZST':
		month_start, day_start, month_end, day_end = dst_ref[0], dst_ref[1], dst_ref[2], dst_ref[3]
		timezone = 'MST'
		if month in [i for i in range(month_start, month_end+1)]:
			if month == month_start:
				if day >= day_start:
					timezone = 'PST'
			elif month == month_end:
				if day < day_end:
					timezone = 'PST'
			else:
				timezone = 'PST'
	return (hour + tz_standard[timezone])%24


# get all crashes within county and neighboring counties of given crash
def create_crash_radius(crash, df_crash, state, county, nbr_st, nbr_counties): #create crash radius of of neighboring counties
	crash_radius = df_crash[(df_crash['state_abbr'] == state) & (df_crash['county'] == county)]
	for (st, cnty) in zip(nbr_st, nbr_counties):
		if int(cnty) != county:
			crash_radius = pandas.concat([crash_radius,df_crash[(df_crash['state_abbr'] == st) & (df_crash['county'] == int(cnty))]])
	return crash_radius.drop(index=crash) # remove itself from the dataset


# get all crashes within state and neighboring states of given crash
def create_statewide_radius(crash, df_crash, state, nbr_st):
	crash_radius = df_crash[df_crash['state_abbr'] == state]
	if len(set(nbr_st)) != 1: # if crash is near a state border, include bordering state data
		for st in list(set(nbr_st)):
			if st != state:
				crash_radius = pandas.concat([crash_radius,df_crash[df_crash['state_abbr'] == st]])
	return crash_radius.drop(index=crash) # remove itself from the dataset


# get all crashes +/- 1 year, +/- 1 hour, similar week(end) status
def find_similar(crash, crash_info, df_crash, day_type='day_type'): # day_type matches week/weekend, day_week for specific day of the week
	day_week, year, month, day, hour, minute = crash_info['day_week'], crash_info['year'], crash_info['month'], crash_info['day'], crash_info['hour'], crash_info['minute']
	if (day_week, hour) == (6,19) or (day_week, hour) == (2,5): # first/last hour of weekday
		df_crash = df_crash[df_crash['day_week'].isin([2,3,4,5,6])]
	elif (day_week, hour) == (6,20): # first hour of weekend
		df_crash = df_crash[df_crash['day_week'].isin([6,7,1])]
	elif (day_week, hour) == (2,4): # last hour of weekend
		df_crash = df_crash[df_crash['day_week'].isin([7,1,2])]
	else:
		df_crash = df_crash[df_crash[day_type] == crash_info[day_type]] # keep week/weekend

	# retain all crashess in  +/-1 year window
	df_crash_yr = df_crash[df_crash['year'] == year]
	df_crash_yr = pandas.concat([df_crash_yr, df_crash[(df_crash['year'] == year-1) & (df_crash['month'] > month)]])
	df_crash_yr = pandas.concat([df_crash_yr, df_crash[(df_crash['year'] == year-1) & (df_crash['month'] == month) & (df_crash['day'] >= day)]])
	df_crash_yr = pandas.concat([df_crash_yr, df_crash[(df_crash['year'] == year+1) & (df_crash['month'] < month)]])
	df_crash_yr = pandas.concat([df_crash_yr, df_crash[(df_crash['year'] == year+1) & (df_crash['month'] == month) & (df_crash['day'] <= day)]])

	# retain all crashes in +/-1 hour window
	hr = 'hour'
	if (len(set(df_crash['timezone'])) > 1):
		hr = 'standard_hour' # use standardized time for crash sets across different timezones
	standard_hour = crash_info[hr]
	crash_similar = df_crash_yr[df_crash_yr[hr] == standard_hour] # keep crash in same hour
	crash_similar = pandas.concat([crash_similar, df_crash_yr[(df_crash_yr[hr] == ((standard_hour+1)%24)) & (df_crash_yr['minute'] <= minute)]])
	crash_similar = pandas.concat([crash_similar, df_crash_yr[(df_crash_yr[hr] == ((standard_hour-1)%24)) & (df_crash_yr['minute'] >= minute)]])

	return crash_similar


def find_closest(crash_lat, crash_lon, df_crash, num_closest):
	if len(df_crash.index) > 1:
		df_crash = df_crash.sample(frac=1) # randomize order of crashes

		# calculate distance to each crash in radius, sort by distance
		df_crash['latitude_eval'] = crash_lat
		df_crash['longitud_eval'] = crash_lon
		df_crash['dist_btw'] = approximate_distance(df_crash['latitude_eval'],df_crash['longitud_eval'],df_crash['latitude'],df_crash['longitud'])
		df_crash = df_crash.sort_values(by=['dist_btw'])

		closest_crash = list(df_crash.index)[:num_closest]
		if len(closest_crash) < num_closest: # report nan values when less than expected number of neighbors is returned
			for missing in range(0,num_closest-len(closest_crash)):
				closest_crash.append(numpy.nan)
		dist = df_crash['dist_btw'].iloc[0] # record closest crash

	else: # return nan values for crashes with no neighbors
		closest_crash, dist = [numpy.nan]*num_closest, numpy.nan

	return closest_crash, dist


def find_missing_data(df_accident,df_vehicle,df_driver,incl_drinking=False):
	# collect missing info about the crash
	df_acc_miss = pandas.DataFrame(index=df_accident.index)
	df_acc_miss['miss_hour'] = df_accident['hour'].isnull()
	df_acc_miss['miss_day_week'] = df_accident['day_week'].isnull()
	#df_acc_miss['miss_state'] = df_accident['state'].isnull()

	# collect missing info about the vehicle
	df_veh_miss = pandas.DataFrame(index=df_vehicle.index)
	df_veh_miss['miss_minor_blemishes'] = (df_vehicle['prev_acc'].isnull() | df_vehicle['prev_spd'].isnull() | df_vehicle['prev_oth'].isnull()) 
	df_veh_miss['miss_major_blemishes'] = (df_vehicle['prev_sus'].isnull() | df_vehicle['prev_dwi'].isnull()) 
	df_veh_miss['miss_any_blemishes'] = (df_veh_miss['miss_minor_blemishes'] | df_veh_miss['miss_major_blemishes'])

	# collect missing info about the driver
	df_dr_miss = pandas.DataFrame(index=df_driver.index)
	df_dr_miss['miss_age'] = (df_driver['age'].isnull()) | (df_driver['age'] < 13) # set child drivers as missing values
	df_dr_miss['miss_sex'] = df_driver['sex'].isnull()
	if incl_drinking == True:
		df_dr_miss['miss_drinking_status'] = df_driver['mibac1'].isnull()

	df_missing = df_acc_miss.merge(df_veh_miss.merge(df_dr_miss,how='left',on='crash_id'),how='left',on='crash_id').groupby(['crash_id']).any()
	df_missing['miss_any'] = df_missing.any(axis='columns')

	return df_missing


def get_analytic_sample(df_accident,df_vehicle,df_person,bac_threshold,drinking_definition,st_yr_threshold=1,year=-1,num_closest=5,mireps=10,
						state=False,day_type=False,first_year=-1,last_year=-1,earliest_hour=-1,latest_hour=-1):

	if year in range(1982,2100): # restrict sample to selected year or years
		analytic_sample = df_accident[df_accident['year']==year]
		df_person = df_person[df_person['year'].isin(range(year-1,year+2))]
	elif first_year in range(1982,2100) and last_year in range(1982,2100):
		analytic_sample = df_accident[df_accident['year'].isin(range(first_year,last_year+1))]
		df_person = df_person[df_person['year'].isin(range(first_year-1,last_year+2))]
	else:
		analytic_sample = df_accident[df_accident['year'].isin(range(1983,2100))]
		df_person = df_person[df_person['year'].isin(range(1982,2100))]
	df_crash_miss = analytic_sample.copy()


	# implement hours, state, week/end restriction
	if earliest_hour in range(0,25) and latest_hour in range(0,25): # implement hours range restriction
		if earliest_hour < latest_hour:
			analytic_sample = analytic_sample.loc[(analytic_sample['hour']>=earliest_hour) & (analytic_sample['hour']<latest_hour)]
		else:
			analytic_sample = analytic_sample.loc[(analytic_sample['hour']>=earliest_hour) | (analytic_sample['hour']<latest_hour)]
	if state in list(us.states.mapping('abbr','name').keys()) or state == 'DC':
		analytic_sample = analytic_sample.loc[analytic_sample['state_abbr']==state]
	if day_type in ['weekday', 'weekend']:
		analytic_sample = analytic_sample.loc[analytic_sample['day_type']==day_type]

	# predefine headers
	mibac_header = [f'mibac{i}' for i in range(1,mireps+1)]
	mibac2_header = {'mibac1':'mibac1_2', 'mibac2':'mibac2_2', 'mibac3':'mibac3_2', 'mibac4':'mibac4_2', 'mibac5':'mibac5_2', 'mibac6':'mibac6_2', 'mibac7':'mibac7_2', 'mibac8':'mibac8_2', 'mibac9':'mibac9_2', 'mibac10':'mibac10_2'}
	mibacb_header = {'mibac1':'mibac1_b', 'mibac2':'mibac2_b', 'mibac3':'mibac3_b', 'mibac4':'mibac4_b', 'mibac5':'mibac5_b', 'mibac6':'mibac6_b', 'mibac7':'mibac7_b', 'mibac8':'mibac8_b', 'mibac9':'mibac9_b', 'mibac10':'mibac10_b'}

	if drinking_definition == 'police_report_primary': # correct bac and mibac values for officer-judged drinking involved
		df_person.loc[df_person['drinking']==0, mibac_header] = [0]*(mireps) # officer reports drinking not involved, assumed bac sober
		if bac_threshold == 0:
			df_person.loc[df_person['drinking']==1, mibac_header] = [12]*(mireps) # officer reports drinking involved, assumed drinking

	# get drinking status of all drivers involved in crash
	df_driver = df_person.loc[df_person['seat_pos']==11] # keep only drivers from the person file
	for mirep in range(1,mireps+1): # fill in bac test results for missing mibac results
		df_driver.loc[:,'mibac'+str(mirep)] = df_driver.loc[:,'mibac'+str(mirep)].fillna(df_driver.loc[:,'alcohol_test_result'].apply(numpy.floor))

	df_missing = find_missing_data(df_crash_miss,df_vehicle,df_driver)
	df_driver = df_driver[['ve_forms','veh_no','alcohol_test_result']+mibac_header]

	# match closest crash drinking status for single vehicle crashes
	df_driver1 = df_driver.loc[(df_driver['ve_forms']==1) & (df_driver['veh_no']==1)]
	df_driver1 = df_driver1.merge(analytic_sample[[f'closest_crash{i}' for i in range(1, num_closest+1)]],how='left',left_index=True,right_index=True)
	df_driver1 = df_driver1.merge(df_driver1.drop(columns=['ve_forms','veh_no','alcohol_test_result']+[f'closest_crash{i}' for i in range(1, num_closest+1)]).rename(columns=mibac2_header), how='left', left_on='closest_crash1',right_index=True)

	# compile drinking status of both drivers in 2 vehicle crashes
	df_driver2 = df_driver.loc[df_driver['ve_forms']==2]
	df_driver_grouped = df_driver2.groupby(level=0) # group drivers by vehicle 1 and vehicle 2
	df_driver2 = df_driver_grouped.last()[df_driver_grouped.last()['veh_no']==2].rename(columns=mibac2_header)
	df_driver2 = df_driver_grouped.first()[df_driver_grouped.first()['veh_no']==1].merge(df_driver2.drop(columns=['ve_forms','veh_no','alcohol_test_result']),how='left',left_index=True,right_index=True)

	# omit applicable drinking values, fill with next closest crash drinking status
	if bac_threshold > 0:
		for mirep in range(1,mireps+1): #omit applicable drinking values
			df_driver1.loc[df_driver1['mibac'+str(mirep)].isin(range(1,bac_threshold)), 'mibac'+str(mirep)] = numpy.nan
			df_driver2.loc[df_driver2['mibac'+str(mirep)].isin(range(1,bac_threshold)), 'mibac'+str(mirep)] = numpy.nan
			df_driver2.loc[df_driver2['mibac'+str(mirep)+'_2'].isin(range(1,bac_threshold)), 'mibac'+str(mirep)+'_2'] = numpy.nan
		for crash_num in range(2,num_closest+1): #omit and replace with next closest crash
			df_driver1 = df_driver1.merge(df_driver1[[f'mibac{i}' for i in range(1,mireps+1)]].rename(columns=mibacb_header), how='left', left_on='closest_crash'+str(crash_num),right_index=True)
			for mirep in range(1,mireps+1):
				df_driver1.loc[df_driver1['mibac'+str(mirep)+'_2'].isin(range(1,bac_threshold)), 'mibac'+str(mirep)+'_2'] = numpy.nan
				df_driver1['mibac'+str(mirep)+'_2'] = df_driver1['mibac'+str(mirep)+'_2'].fillna(df_driver1['mibac'+str(mirep)+'_b'])
			df_driver1 = df_driver1.drop(columns=[f'mibac{i}_b' for i in range(1,mireps+1)])

	# compile drinking status of one and two vehicle crashes and merge with crash data
	df_driver = pandas.concat([df_driver1,df_driver2])
	analytic_sample = analytic_sample.merge(df_driver.drop(columns=['ve_forms', 'veh_no']+[f'closest_crash{i}' for i in range(1, num_closest+1)]),how='inner',left_index=True,right_index=True)
	
	analytic_sample = analytic_sample[~analytic_sample.index.duplicated(keep='first')] #remove duplicates
	analytic_sample = analytic_sample[~analytic_sample[mibac_header].isnull().all(axis=1)] #remove crashes without valid drinking information
	analytic_sample = analytic_sample[~analytic_sample[[f'mibac{i}_2' for i in range(1,mireps+1)]].isnull().all(axis=1)] # remove crashes without valid paired drinking information

	# eliminate crashes in state-year pairs with missing data above threshold
	df_missing = analytic_sample[['year','state_abbr']].merge(df_missing[['miss_any']],how='left',right_index=True,left_index=True) # merge in missing data values
	df_missing = df_missing.groupby(['state_abbr','year']).mean() # determine proportion of state-year missing values
	analytic_sample = analytic_sample.reset_index().merge(df_missing[['miss_any']],how='left',on=['year','state_abbr']).set_index(analytic_sample.index.names)
	analytic_sample = analytic_sample.loc[analytic_sample['miss_any']<=st_yr_threshold]

	return analytic_sample[['month','day','ve_forms','state_abbr','year','hour','day_type','time_day','latitude','longitud']+mibac_header+[f'mibac{i}_2' for i in range(1,mireps+1)]]


def get_estimation_sample(analytic_sample, bac_threshold, bc_mixing, mirep):
	estimation_sample = analytic_sample[bc_mixing+['ve_forms','mibac'+str(mirep), 'mibac'+str(mirep)+'_2']].rename(columns={'mibac'+str(mirep): 'bac1', 'mibac'+str(mirep)+'_2': 'bac2'})

	if bac_threshold == 0:
		bac_threshold = 1 # change BAC threshold to minimum drinking value

	estimation_sample['a1_s'] = 0
	estimation_sample.loc[(estimation_sample['ve_forms']==1) & (estimation_sample['bac1']==0),'a1_s'] = 1 # single vehicle nondrinking driver

	estimation_sample['a1_d'] = 0
	estimation_sample.loc[(estimation_sample['ve_forms']==1) & (estimation_sample['bac1']>=bac_threshold),'a1_d'] = 1 # single vehicle drinking driver

	estimation_sample['a1_d_d'] = 0
	estimation_sample.loc[(estimation_sample['ve_forms']==1) & (estimation_sample['bac2']>=bac_threshold),'a1_d_d'] = 1 # drinking status of nearest single vehicle crash

	estimation_sample['a2_ss'] = 0
	estimation_sample.loc[(estimation_sample['ve_forms']==2) & (estimation_sample['bac1']==0) & (estimation_sample['bac2']==0),'a2_ss'] = 1 # 2 nondrinking drivers

	estimation_sample['a2_sd'] = 0
	estimation_sample.loc[(estimation_sample['ve_forms']==2) & (estimation_sample['bac1']>=bac_threshold) & (estimation_sample['bac2']==0),'a2_sd'] = 1 # two vehicle crash, one drinking driver
	estimation_sample.loc[(estimation_sample['ve_forms']==2) & (estimation_sample['bac1']==0) & (estimation_sample['bac2']>=bac_threshold),'a2_sd'] = 1 # two vehicle rash, one drinking driver

	estimation_sample['a2_dd'] = 0
	estimation_sample.loc[(estimation_sample['ve_forms']==2) & (estimation_sample['bac1']>=bac_threshold) & (estimation_sample['bac2']>=bac_threshold),'a2_dd'] = 1 # two vehicle crash, two drinking drivers

	estimation_sample['a1_total'] = estimation_sample['a1_s'] + estimation_sample['a1_d']
	estimation_sample['a2_total'] = estimation_sample['a2_ss'] + estimation_sample['a2_sd'] + estimation_sample['a2_dd']

	return estimation_sample[bc_mixing+['a1_s','a1_d','a1_total','a1_d_d','a2_ss','a2_sd','a2_dd','a2_total']]


def get_estimation_cells(estimation_sample,bc_mixing):
	if len(bc_mixing) > 0:
		estimation_cells = estimation_sample.groupby(delta_mixing).sum().reset_index()[delta_mixing]
		estimation_sample_a1 = estimation_sample[estimation_sample['a1_total']==1]
		for cell, cell_info in estimation_cells.iterrows(): # fit delta values for observational units
			sample_crashes = estimation_sample_a1.copy()
			for (parameter, value) in zip(delta_mixing, list(cell_info)[:4]):
				sample_crashes = sample_crashes[sample_crashes[parameter]==value]
			if len(sample_crashes.index) > 0:
				drinking_status = sample_crashes['a1_d'].values.reshape(len(sample_crashes.index), 1)
				neighbor_drink = sample_crashes['a1_d_d'].values.reshape(len(sample_crashes.index), 1)

				regr = sklearn.linear_model.LinearRegression()
				regr.fit(drinking_status, neighbor_drink)
				estimation_cells.loc[cell,'a1_delta'] = round(1+regr.coef_[0][0],3)
		estimation_sample = estimation_sample.groupby(bc_mixing).sum().merge(estimation_cells.set_index(delta_mixing),how='left',left_index=True,right_index=True)

	else:
		estimation_sample_a1 = estimation_sample[estimation_sample['a1_total']==1]
		drinking_status = estimation_sample_a1['a1_d'].values.reshape(len(estimation_sample_a1.index), 1)
		neighbor_drink = estimation_sample_a1['a1_d_d'].values.reshape(len(estimation_sample_a1.index), 1)
		regr = sklearn.linear_model.LinearRegression()
		regr.fit(drinking_status, neighbor_drink)

		#sample_delta = round(1+regr.coef_[0][0],3)
		#estimation_sample = estimation_sample.groupby(bc_mixing).sum().merge(estimation_cells,how='left',left_index=True,right_index=True)
		estimation_sample = estimation_sample.groupby(bc_mixing).sum()
		estimation_sample.loc[:,'a1_delta'] = round(1+regr.coef_[0][0],3)

	return estimation_sample[['a1_s','a1_d','a1_total','a1_delta','a2_ss','a2_sd','a2_dd','a2_total']]



def fit_model(analytic_sample, bac_threshold, bc_mixing, bsreps, mirep):
	#df_parameters = pandas.DataFrame()
	boot_results = numpy.zeros((bsreps,6))
	estimation_sample = get_estimation_sample(analytic_sample, bac_threshold, bc_mixing, mirep=mirep)
	estimation_sample_a1 = estimation_sample[estimation_sample['a1_total']==1]
	estimation_sample = estimation_sample.groupby(bc_mixing).sum()

	one_veh_ratio = estimation_sample['a1_d'].sum()/estimation_sample['a1_s'].sum() # save one vehicle crash ratio before elimination
	regr = sklearn.linear_model.LinearRegression() # regresssion of single vehicle and neighbor drinking status excess interaction parameters
	regr.fit(estimation_sample_a1['a1_d'].values.reshape(len(estimation_sample_a1.index), 1), estimation_sample_a1['a1_d_d'].values.reshape(len(estimation_sample_a1.index), 1))
	one_veh_alpha0 = round(regr.intercept_[0],3)
	one_veh_alpha = round(regr.coef_[0][0],3)

	for bsrep in range(bsreps): # bootstrap estimates
		bs_sample = estimation_sample.sample(frac=1,replace=True)

		# remove observations with 0 values, otherwise it will not converge    
		bs_sample = bs_sample.drop(bs_sample[bs_sample['a1_s'] == 0].index)
		bs_sample = bs_sample.drop(bs_sample[bs_sample['a1_d'] == 0].index)

		mod = DDP(bs_sample)
		results = mod.fit(start_params=[10,10])
		boot_results[bsrep,1], boot_results[bsrep,2] = results.params[-2], results.params[-1] #theta, lambda

		boot_results[bsrep,0] = (1/boot_results[bsrep,2])*one_veh_ratio # N
		boot_results[bsrep,3] = boot_results[bsrep,0]/(1+boot_results[bsrep,0]) # prevalance

		boot_results[bsrep,4] = one_veh_alpha0 # national alpha0 estimate
		boot_results[bsrep,5] = one_veh_alpha # national alpha estimate

	return boot_results


def _ll(A, thet, lamb):
	num_agg = numpy.size(A,axis=0)
	ll = numpy.zeros((num_agg)) # log-likelihood

	A_1 = A[:,:3]
	A_2 = A[:,-4:]
	N = (1/lamb)*(A_1[:,1]/A_1[:,0])

	p = numpy.zeros((num_agg,4))
	p[:,0] = [1]*num_agg # SS term
	p[:,1] = (thet+1)*N # SD term
	p[:,2] = thet*N**2 # DD term
	p[:,3] = p[:,0] + p[:,1] + p[:,2] # denominator

	for i in range(num_agg):
		ll[i] += math.lgamma(A_2[i,3]+1)
		for j in range(0,3):
			ll[i] -= math.lgamma(A_2[i,j]+1)

	ll += A_2[:,0]*numpy.log(p[:,0])
	ll += A_2[:,1]*numpy.log(p[:,1])
	ll += A_2[:,2]*numpy.log(p[:,2])

	ll -= A_2[:,3]*numpy.log(p[:,3])
	return ll


class DDP(GenericLikelihoodModel):
	def __init__(self, endog, exog=None, **kwds):
		super(DDP, self).__init__(endog, exog, **kwds)
        
	def nloglikeobs(self, params):
		thet = params[-2]
		lamb = params[-1]
		return -_ll(self.endog, thet, lamb)

	def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
		return super(DDP, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, disp=False, **kwds)


def bs_error(boot_results):
	return numpy.power(numpy.divide(numpy.power(numpy.subtract(boot_results,boot_results.sum(axis=0)/numpy.size(boot_results,0)),2).sum(axis=0),(numpy.size(boot_results,0))-1),0.5)


def mi_se(results, bs_error):
	return numpy.power(numpy.power(bs_error,2).sum(axis=0)/numpy.size(bs_error,0) + (1+(1/numpy.size(results,0)))*(1/(numpy.size(results,0)-1))*(numpy.power(numpy.subtract(results,results.sum(axis=0)/numpy.size(results,0)),2).sum(axis=0)),0.5)