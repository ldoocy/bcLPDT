import os, us, pandas, numpy, utils, csv, random

firstYear = 1983
lastYear = 2022

earliest_hour=20
latest_hour=5

bsreps = 10 # bootstrap replicates for replication
mireps = 10 # multiple imputation replicates for replication (FARS includes a total of 10), False if police report and BAC only
num_closest = 5 # define set of closest accidents for each accident
drinking_definition = 'bac_primary' # bac_primary or police_report_primary to check officer judgement first
bac_threshold = 0 # set BAC cutoff. 0 for any BAC, 8 for 0.08 legal limit, or 10 for 0.10 legal limit prior to 2004
st_yr_threshold = 0.13 # threshold for missing data in state-year pairs, any states falling above this threshold are eliminated LP threshold 0.13, 1 for missing data exclusion
yr_int = 5 # set year interv al
bc_mixing = ['year','state_abbr','hour','day_type'] # set observational unit parameters

df_accident = pandas.read_csv('summary_data/df_accident.csv', index_col='crash_id', low_memory=False)
df_vehicle = pandas.read_csv('summary_data/df_vehicle.csv', index_col='crash_id', low_memory=False)
df_person = pandas.read_csv('summary_data/df_person.csv', index_col='crash_id', low_memory=False)

if not os.path.exists('results'):
	os.makedirs('results') # generate results directory, if it doesn't exist

#print('[delta, N, theta, lambda, prevalance]')
final_results = pandas.DataFrame(index=list(range(firstYear,lastYear,yr_int)))
final_se = pandas.DataFrame(index=list(range(firstYear,lastYear,yr_int)))
for yr in range(firstYear, lastYear, yr_int):
	yr_end = yr+yr_int-1
	if yr_end > lastYear:
		yr_end = lastYear

	results = numpy.zeros((mireps,5))
	bs_error = numpy.zeros((mireps,5))
	summary_stats = pandas.DataFrame(index=list(range(1,mireps+1)))
	analytic_sample = utils.get_analytic_sample(df_accident,df_vehicle,df_person,bac_threshold,drinking_definition,st_yr_threshold,first_year=yr,last_year=yr_end,earliest_hour=earliest_hour,latest_hour=latest_hour)
	for mirep in range(1,mireps+1):
		boot_results = utils.fit_model(analytic_sample, bac_threshold, bc_mixing, bsreps, mirep)
		results[mirep-1,:] = numpy.mean(boot_results, axis=0)
		bs_error[mirep-1,:] = utils.bs_error(boot_results)
		#print(yr, 'mi', mirep, numpy.mean(boot_results, axis=0))

		estimation_sample = utils.get_estimation_sample(analytic_sample, bac_threshold, [], mirep=mirep)
		summary_stats.loc[mirep,['a1_s','a1_d','a1_total','a1_d_d','a2_ss','a2_sd','a2_dd','a2_total']] = estimation_sample.sum()


	totals = summary_stats.sum()
	print(yr, yr_end, 'total single car crash', int(totals['a1_total']/mireps), 'total two car', int(totals['a2_total']/mireps))
	print('share single car sober', totals['a1_s']/totals['a1_total'])
	print('share two drinking drivers', totals['a2_dd']/totals['a2_total'])
	print('share two sober drivers', totals['a2_ss']/totals['a2_total'])

	final_results.loc[yr,['delta', 'N', 'theta', 'lambda', 'prevalance']] = numpy.mean(results, axis=0)
	final_se.loc[yr,['delta', 'N', 'theta', 'lambda', 'prevalance']] = utils.mi_se(results, bs_error)
final_results.to_csv('bs_results.csv')
final_se.to_csv('bs_se.csv')