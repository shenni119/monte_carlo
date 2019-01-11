"""Monte Carlo Simulation for interest accrural that references prior simulated values
to simulate new values. Can accomodate both discrete and normally distrubted outcomes.
Also plots the simulated values and """

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

#Values to set
iteration_count=1000

#certain_val
certain_val=20

#discrete probable outcomes
discrete_elements = [0, 100, 300]
discrete_probabilities = [0.7, 0.2, 0.1]
np_data=np.random.choice(discrete_elements, 1000, p=discrete_probabilities)

#normally distributed outcomes
mean = .08
standard_deviation = .12
interest_norm_distribution = norm(loc = mean, scale = standard_deviation)

def simulation_based_on_prior(iterations,distribution_model): #distribution_model=output of norm()
    sub_simulations = iterations
    sub_results = distribution_model.rvs(sub_simulations)
    df = pd.DataFrame({
        "simulation_results": sub_results,
        "deposit":certain_val
        })
    df['year_end'] = (df.at[0,'simulation_results']+1)*20
    df.loc[0, 'compounding_results'] = df.loc[0, 'year_end']
    for i in range(1, len(df)):
        df.loc[i, 'compounding_results'] = (df.loc[i-1, 'compounding_results']+df.loc[i, 'deposit']) * \
            (1 + df.loc[i, 'simulation_results'])
    final_balance=df.loc[iterations-1, 'compounding_results']
    return final_balance

output_lst=[]
for i in range(iteration_count):
    output=simulation_based_on_prior(8,interest_norm_distribution)
    output_lst.append(output)

final_df = pd.DataFrame({
    "final_compounding_results": output_lst
    })

#Plotting the results
plt.hist(final_df["final_compounding_results"], bins=int(iteration_count/50))
plt.axvline(x = 300, c = "r")
plt.show()

#Calculating the confidence Interval
desired_confidence_intv=0.95
def mean_confidence_interval(data, confidence): #credit to stackoverflow
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

mean_val,conf_int_min,conf_int_max=mean_confidence_interval(final_df["final_compounding_results"]\
                                                            , confidence=desired_confidence_intv)
print ("the mean simulated value is {}, with a {} confidence interval between {} and {}".\
       format(mean_val,desired_confidence_intv,conf_int_min,conf_int_max))

final_df.to_csv("simulation_results.csv")
