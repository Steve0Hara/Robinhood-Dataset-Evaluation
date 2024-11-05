import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")  

# customize matplotlib
plt.rcParams['axes.facecolor'] = '#2e4b6b'  
plt.rcParams['figure.facecolor'] = '#2e4b6b'  
plt.rcParams['axes.edgecolor'] = 'white'  
plt.rcParams['axes.labelcolor'] = 'white'  
plt.rcParams['xtick.color'] = 'white'  
plt.rcParams['ytick.color'] = 'white'  
plt.rcParams['text.color'] = 'white' 
plt.rcParams['grid.color'] = '#073642'  
plt.rcParams['lines.color'] = 'white'  
plt.rcParams['patch.edgecolor'] = 'white'  
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['white'])
plt.rcParams['font.weight'] = 'bold'  
plt.rcParams['axes.labelweight'] = 'bold'  
plt.rcParams['axes.titleweight'] = 'bold'  

# class for plotting regression summaries
class RegressionSummaryPlotter:
    def __init__(self, model):
        self.model = model

    def plot_summary_table(self, save_path):
        summary_text = self.model.summary().as_text()
        fig = plt.figure(figsize=(12, 8))
        plt.axis('off') 
        plt.text(0, 0, summary_text, fontsize=10, family='monospace', color='white')  
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

# ------------------------------------- Assignment Starts Here ---------------------------------------------

# converting .dta file to pandas dataframe
df = pd.read_stata('rh_daily.dta')

"""
1)  Drop observations with negative or missing price. Drop also observations
    during Robinhood outages on March 2, March 3, and June 18, 2020. (Hint: you can use the command mdy
    in Stata to specify a date from month, day and year.) How many distinct stocks are left in the sample? 
    And how many days?
"""

# drop observations with negative or missing price
df = df[df['PRC'] > 0]
df = df.dropna(subset=['PRC'])  

# specify the dates to exclude
df['date'] = pd.to_datetime(df['date'])  
outage_dates = pd.to_datetime(['2020-03-02', '2020-03-03', '2020-06-18'])

# drop rows with the outage dates
df = df[~df['date'].isin(outage_dates)]

# number of distinct stocks and days
distinct_stocks = df['tic'].nunique()  
distinct_days = df['date'].nunique()
print(f"1) Number of distinct stocks = {distinct_stocks}")
print(f"1) Number of distinct days = {distinct_days}")

# ------------ Mean rh per day -----------------------
df_daily_stats = df.groupby('date')['rh'].agg(mean_rh='mean', median_rh='median').reset_index()
plt.figure(figsize=(10, 6))
plt.plot(df_daily_stats['date'], df_daily_stats['mean_rh'], color='white', label='Mean RH Investors')
plt.plot(df_daily_stats['date'], df_daily_stats['median_rh'], color='white', linestyle='--', label='Median RH Investors')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of RH Investors', fontsize=12)
plt.title('Daily Mean and Median of RH Investors Over Time', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Mean_Median_rh_investors_daily.png', bbox_inches='tight')
plt.show()



"""
2)  Generate a variable year. Then:

    (a) Present summary statistics by year of the number of RH investors
        holding a stock. For each year, the table should show the number
        of observations, mean, standard deviation and median number of
        investors holding a stock.
"""

# generate year variable
df['year'] = df['date'].dt.year

# summary statistics
summary_stats = df.groupby('year')['rh'].agg(observations='count', mean='mean', std_dev='std', median='median')

print(f"2) Summary statistics:")
print(summary_stats)

# ------------- Optional Plot ------------------
df_2020 = df[df['year'] == 2018]

# Cclculate the mean and median for 2018
mean_rh_2020 = df_2020['rh'].mean()
median_rh_2020 = df_2020['rh'].median()
plt.figure(figsize=(10, 6))
sns.histplot(df_2020['rh'], color='white', bins=30)
plt.axvline(mean_rh_2020, color='blue', linestyle='--', label=f'Mean: {mean_rh_2020:.2f}')
plt.axvline(median_rh_2020, color='orange', linestyle='-', label=f'Median: {median_rh_2020:.2f}')
plt.title('Distribution of RH Investors in 2018', fontsize=14)
plt.xlabel('Number of RH Investors', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('rh_distribution_2020_with_mean_median.png', bbox_inches='tight')
plt.show()



"""
3)  Robinhood holdings of Uber Inc. (ticker: UBER) only start being reported
    sometime in 2019. Explain why prior dates are missing.
"""

uber = df[df['tic'] == 'UBER']
first_uber_date = uber['date'].min()
print(f"3) First reported date for UBER: {first_uber_date}")



"""
4)  Generate a dummy variable year19 that is equal to 1 for observations in
    2019 and zero otherwise. Generate a similar dummy variable
    year20 to indicate holdings in 2020. Estimate the following model: ...
"""

# set dummy variables
df['year19'] = (df['year'] == 2019).astype(int)  
df['year20'] = (df['year'] == 2020).astype(int) 

# perform regression
X = df[['year19', 'year20']]
X = sm.add_constant(X)  # add beta0
y = df['rh']

model = sm.OLS(y, X).fit()
print('4) Regression of year19 and year20 on rh')
print(model.summary())

# print the p-values
print("P-values of the regression:")
print(model.pvalues)

# plot model summary table
plotter4 = RegressionSummaryPlotter(model)
plotter4.plot_summary_table('Exercise_4_Summary_Table.png')

# ------------- Apple Experiment -------------------
# filter the DataFrame for Apple stock (tic == 'AAPL')
df_apple = df[df['tic'] == 'AAPL']
df_apple['year19'] = (df_apple['year'] == 2019).astype(int)
df_apple['year20'] = (df_apple['year'] == 2020).astype(int)
X = df_apple[['year19', 'year20']]
X = sm.add_constant(X)  # add beta0
y = df_apple['rh']

model_apple = sm.OLS(y, X).fit()
print('4) Regression of year19 and year20 on rh for Apple (AAPL)')
print(model_apple.summary())

# print the p-values
print("P-values of the regression:")
print(model_apple.pvalues)

plotter4_apple = RegressionSummaryPlotter(model_apple)
plotter4_apple.plot_summary_table('Exercise_4_Summary_Table_Apple.png')



"""
5)  Create a dummy variable called COVID that equals 1 for dates on or
    after March 13, 2020, the date when COVID-19 was declared a national
    emergency in the United States. Regress rh on this dummy variable and
    interpret the coefficient of the dummy. Also, comment on its statistical
    significance.
"""

# set COVID variable
df['COVID'] = (df['date'] >= '2020-03-13').astype(int)
df.to_csv('ex_5.csv', index=False)

# perform regression
X_covid = df['COVID']
X_covid = sm.add_constant(X_covid) # add beta0
y_covid = df['rh']

model_covid = sm.OLS(y_covid, X_covid).fit()
print('5) Regression of rh on COVID dummy variable')
print(model_covid.summary())

# print the p-values
print("P-values of the regression:")
print(model_covid.pvalues)

# plot model summary table
plotter5 = RegressionSummaryPlotter(model_covid)
plotter5.plot_summary_table('Exercise_5_Summary_Table.png')



"""
6)  To what degree does the previous result offer evidence that the COVID lockdown
    causally contributed to an increase in the use of the RH plat- form? Formally, under which assumptions can we interpret the estimated
    coefficient on the COVID dummy as reflecting the causal effect of COVID on the number of holdings? What specific threats to the identification of
    this causal effect can you identify in this case? Provide a plausible non-causal explanation for what you observe in the data. 
    Additionally, provide a motivation for why the Covid-19 pandemic might have (causally) led to an increase in the number of holdings.
"""

"""
7)  While working on this assignment and trying to understand whether Covid did indeed contribute to the growth of retail investing platforms such
    as Robinhood, you happen to read the article below. Does the article spur any ideas for additional analysis you could perform (possibly after
    collecting more data) that would make the causal story more convincing?
    Explain. [Answers cannot exceed 300 words.]
"""

df['date'] = pd.to_datetime(df['date'])

# set dummy variables
df['February_2020'] = ((df['date'] >= '2020-02-01') & (df['date'] <= '2020-02-29')).astype(int)
df['March_2020'] = ((df['date'] >= '2020-03-01') & (df['date'] <= '2020-03-31')).astype(int)
df['April_2020'] = ((df['date'] >= '2020-04-01') & (df['date'] <= '2020-04-30')).astype(int)
df['May_2020'] = ((df['date'] >= '2020-05-01') & (df['date'] <= '2020-05-31')).astype(int)
df['June_2020'] = ((df['date'] >= '2020-06-01') & (df['date'] <= '2020-06-30')).astype(int)
df['July_2020'] = ((df['date'] >= '2020-07-01') & (df['date'] <= '2020-07-31')).astype(int)
df['After_July_2020'] = (df['date'] > '2020-07-31').astype(int)

# set COVID variable based on the cutoff date of March 13, 2020 (optional)
df['COVID'] = (df['date'] >= '2020-03-13').astype(int)

df.to_csv('ex_7_with_feb_july_and_covid.csv', index=False)

# perform regression
X_time_frames = df[['February_2020', 'March_2020', 'April_2020', 'May_2020', 'June_2020', 'July_2020', 'After_July_2020']]
X_time_frames = sm.add_constant(X_time_frames)  # add beta0
y_rh = df['rh']
model_time_frames = sm.OLS(y_rh, X_time_frames).fit()

print('Regression of rh on time frame dummy variables, including February 2020')
print(model_time_frames.summary())

# print the p-values
print("P-values of the regression:")
print(model_time_frames.pvalues)

plotter_time_frames = RegressionSummaryPlotter(model_time_frames)
plotter_time_frames.plot_summary_table('Exercise_Feb_July_Time_Frames_Summary_Table.png')



"""
8)  Compute the weight of stock i in the aggregate RH portfolio (ARH) on day t as ...
    (Notice that the ARH portfolio assumes that each investor holding represents an equal amount of dollars.)
"""

# total number of rh per day
df['total_rh_daily'] = df.groupby('date')['rh'].transform('sum')

# weight of stock i in ARH portfolio per day
df['arh_it'] = df['rh'] / df['total_rh_daily']

df.head(100).to_csv('ex_8.csv', index = False)



"""
9)  Compute the weights vwi,t of the market capitalization-weighted portfolio,
    which will serve as our benchmark portfolio:
"""

# calculating market cap for each stock for each day
df['mktcap'] = df['PRC'] * df['SHROUT']

# total market cap for each day
df['total_mktcap_daily'] = df.groupby('date')['mktcap'].transform('sum')

# weight of stock i in market cap weighted portfolio on day t
df['vw_it'] = df['mktcap'] / df['total_mktcap_daily']

df.head(100).to_csv('ex_9.csv', index=False)



"""
10) Which company had, on average, the largest weight in the ARH portfolio in 2018? What about 2019 and 2020? 
    Compare this with the largest positions in the value-weighted portfolio.
"""
df_ex_10 = df

# group df by ticker and year and calcualte average weights
average_weights = df_ex_10.groupby(['tic', 'year']).agg(avg_weight_arh_it=('arh_it', 'mean'), avg_weight_vw_it=('vw_it', 'mean')).reset_index()

print('10)')

# find stocks with highest average weights
for i in [2018, 2019, 2020]:
    max_arh_tic = average_weights[average_weights['year'] == i].nlargest(1, 'avg_weight_arh_it')
    max_vw_tic = average_weights[average_weights['year'] == i].nlargest(1, 'avg_weight_vw_it')
    print(f"In year {i} the stock with highest arh_it weight was: {max_arh_tic['tic'].values[0]} with an average weight of {max_arh_tic['avg_weight_arh_it'].values[0]}")
    print(f"In year {i} the stock with highest vw_it weight was: {max_vw_tic['tic'].values[0]} with an average weight of {max_vw_tic['avg_weight_vw_it'].values[0]}")



"""
11) Keep only observations in 2019. For each stock, calculate the average arh, vw and mktcap during that year. 
    Generate a variable rank that ranks stocks based on their average market cap (i.e., rank = 1 for the company with the highest average market cap).
"""

print(df.head())
print ('11)')

# only keeping 2019
df_2019 = df[df['year'] == 2019]

# calculate average arh, vw and mktcap in new df
df_2019_avg = df_2019.groupby('tic').agg(avg_arh_it=('arh_it', 'mean'), avg_vw_it=('vw_it', 'mean'), avg_mktcap_it=('mktcap', 'mean')).reset_index()

# generating a rank column
df_2019_avg['rank'] = df_2019_avg['avg_mktcap_it'].rank(ascending=False)

print(df_2019_avg.head(100))
df_2019_avg.to_csv('ex_11.csv', index=False)

# plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='rank', y='avg_mktcap_it', data=df_2019_avg, color='white', s=20)
plt.xlabel('Rank (1 = highest market cap)', fontsize=12, fontweight='bold')
plt.ylabel('Average Market Cap (mktcap)', fontsize=12, fontweight='bold')
plt.title('Stock Rank & Average Market Cap in 2019', fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('Exercise_11.png', bbox_inches='tight')
plt.show()



"""
12) Regress (average) arh on log(rank). Interpret the slope coefficient (both magnitude and sign!). 
    Do Robinhood investor tend to allocate a larger proportion of their portfolios to large or small companies?
"""

# log-transforming the rank variable
df_2019_avg['log(rank)'] = np.log(df_2019_avg['rank'])

# perform regression of avg_arh_it on log(rank)
X = df_2019_avg['log(rank)']
y = df_2019_avg['avg_arh_it']
X = sm.add_constant(X) # add beta0

model_rank = sm.OLS(y, X).fit()

print('12) Regression of average arh_it on log(rank)')
print(model_rank.summary())

# print the p-values
print("P-values of the regression:")
print(model_rank.pvalues)

# plot model summary table
plotter12 = RegressionSummaryPlotter(model_rank)
plotter12.plot_summary_table('Exercise_12_Summary_Table.png')

# plot arh and rank
plt.figure(figsize=(10, 6))
plt.scatter(df_2019_avg['rank'], df_2019_avg['avg_arh_it'], color='white')
plt.title('arh vs. rank (no log transformation)', fontsize=14)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Average arh', fontsize=12)
plt.tight_layout()
plt.savefig('arh_and_rank_nolog.png', bbox_inches='tight')
plt.show()

# plot arh and log(rank)
plt.figure(figsize=(10, 6))
plt.scatter(df_2019_avg['log(rank)'], df_2019_avg['avg_arh_it'], color='white')
plt.title('arh vs. log(rank)', fontsize=14)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Average arh', fontsize=12)
plt.tight_layout()
plt.savefig('arh_and_rank_log.png', bbox_inches='tight')
plt.show()

# plot arh and log(rank) and regression line
plt.figure(figsize=(10, 6))
plt.scatter(df_2019_avg['log(rank)'], df_2019_avg['avg_arh_it'], color='white', label='Data Points')
predicted_values = model_rank.predict(X)
plt.plot(df_2019_avg['log(rank)'], predicted_values, color='orange', label='Regression Line')
plt.title('arh vs. log(rank)', fontsize=14)
plt.xlabel('Log(Rank)', fontsize=12)
plt.ylabel('Average arh', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('arh_and_rank_log_with_regression.png', bbox_inches='tight')
plt.show()



"""
13) What is the predicted change in (average) arh for a company moving from rank 10 to rank 9? And what about a company moving from rank 2 to rank 1?
"""

# regression coefficients from model
beta0 = model_rank.params['const'] # intercept
beta1 = model_rank.params['log(rank)'] # slope

# predicted change
predicted_change_10_9 = (beta0 + beta1 * np.log(10)) - (beta0 + beta1 * np.log(9))
predicted_change_2_1 = (beta0 + beta1 * np.log(2)) - (beta0 + beta1 * np.log(1))

print('13)')
print(f"Predicted change in arh from rank 10 to rank 9: {predicted_change_10_9}")
print(f"Predicted change in arh from rank 2 to rank 1: {predicted_change_2_1}")



"""
14) Comment on the following statement: “arh larger than the fitted value arh d indicates that RH investors overweight the stock compared to the value-weighted benchmark portfolio.”
"""

# experiment for ex 14
df_2019_avg['fitted_arh'] = model_rank.predict(X)

# condition 1: arh_i > fitted_arh_i (actual arh > predicted/fitted arh)
df_2019_avg['cond1_arh_gt_fitted'] = df_2019_avg['avg_arh_it'] > df_2019_avg['fitted_arh']

# condition 2: arh_i > vw_i (actual arh > value-weighted portfolio weight)
df_2019_avg['cond2_arh_gt_vw'] = df_2019_avg['avg_arh_it'] > df_2019_avg['avg_vw_it']

df_2019_avg['both_conditions'] = (df_2019_avg['cond1_arh_gt_fitted']) & (df_2019_avg['cond2_arh_gt_vw'])

total_stocks = len(df_2019_avg)
stocks_meeting_both_conditions = df_2019_avg['both_conditions'].sum()
proportion_meeting_both_conditions = stocks_meeting_both_conditions / total_stocks

print(f"Proportion of stocks where arh_i > fitted_arh_i and arh_i > vw_i: {proportion_meeting_both_conditions:.4f}")

# check conditions
all_conditions_true = df_2019_avg['both_conditions'].all()

if all_conditions_true:
    print("The statement holds true for all data points.")
else:
    print(f"The statement does not hold true for {total_stocks - stocks_meeting_both_conditions} data points.")

# find the instances where c1 is True but c2 is False
df_cond1_true_cond2_false = df_2019_avg[(df_2019_avg['cond1_arh_gt_fitted'] == True) & (df_2019_avg['cond2_arh_gt_vw'] == False)]

# find the instances where c1 is True but c2 is False
df_cond1_true_cond2_true = df_2019_avg[(df_2019_avg['cond1_arh_gt_fitted'] == True) & (df_2019_avg['cond2_arh_gt_vw'] == True)]

print(f"Number of stocks where arh_i > fitted_arh_i but arh_i <= vw_i: {len(df_cond1_true_cond2_false)}")
print(f"Number of stocks where arh_i > fitted_arh_i and arh_i >= vw_i: {len(df_cond1_true_cond2_true)}")

df_cond1_true_cond2_false.to_csv('ex_14_cond1_true_cond2_false.csv', index=False)
