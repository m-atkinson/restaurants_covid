#%% [markdown]
# # Overview 
# - Load restaurant survey data
# - weight results
# - Create graph showing changes
#   - Bar charts of question responses
#   - Combine frequency eat out questions into binary
#   - Chart the change from wave to wave
# - Cluster
# - Run another round of surveys

#%%
import pandas as pd
import numpy as np
from ipfn.ipfn import ipfn
import pytest
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
# %%
df = pd.read_csv('C:/Users/matth/environments/dsci1_env/restaurants_covid/Dining Out Survey (Responses) - Form Responses 1.csv')

# dropping duplicate survey takers and columns don't need
df.drop_duplicates('What is your Amazon Worker ID?', inplace=True)
df.drop(['What is your Amazon Worker ID?', 'Unnamed: 9', 'Timestamp', 'In what zip code do you live?'], axis=1, inplace=True)

# making human readable names
df.rename(
    {
        'Before the coronavirus crisis, how often did you eat out at a sit-down restaurant?':'precovid',
        'Currently how often do you eat out at a sit-down restaurant?': 'current',
        "Next year how often do you think you'll eat out at a sit-down restaurant?": 'postcovid',
        'What is your gender?': 'gender',
        'What is the highest level of education you have completed?': 'educ'
    },
    axis = 1,
    inplace = True
)
# %% Adding column for age in years
df['age_years'] = (2020-df['In what year were you born?'])

# %% remapping answers for current eat out to continuous variable
# Min number of times eat out per category per month
df['current_freq'] = df.current
df['current_freq'] = df.current_freq.map({'Less than once per month' : 0, 'Once a month': 1, 'Multiple time per month' : 3, 'Once a week': 4, 'Multiple times per week': 8 })

#%%

# get age, bin, and drop year column
df['age'] = pd.cut(2020 - df['In what year were you born?'], bins=[-1, 24, 34, 44, 54, 64, 1000],
       labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
df.drop(['In what year were you born?'], axis=1, inplace=True)



# too few responses to estimate
df = df.loc[df.gender != 'Other', :]

# grouping educ into college, not college
df['educ'] = df.educ.map({'College graduate': 'college', 'Post-graduate (e.g. MS, MBA, PhD, etc.)': 'college',
                    'High school' : 'not_college', 'Did not finish high school': 'not_college'})


# fixing indexing after dropping some rows
df.reset_index(drop=True, inplace=True)

print(df.shape)
print(df.columns)
df.head()
# %%
# Weighting
np.random.seed(1)

bias_vars = ['gender', 'educ', 'age']
samp_aggs = [df[col].value_counts().sort_index().values for col in bias_vars]   # marginal counts
dims = [[i] for i in range(len(samp_aggs))]  # i'm using 1-d marginals
# print(aggs, dims)

# random initial values, using poisson to get count-like values
samp_init = np.random.poisson(np.mean([np.mean(x) for x in samp_aggs]), (2,2,6)) + 1  

ipf = ipfn(samp_init, samp_aggs, dims)
samp_dist = ipf.iteration()
samp_dist = samp_dist / np.sum(samp_dist)  # normalizing it to a proper distribution

# %%
# repeating for population marginals
np.random.seed(1)

from collections import OrderedDict

# ['gender', 'educ', 'age']
refdist = OrderedDict({
    'gender': {'Female': .52, 'Male': .48},  
    'educ':  {'college': .35, 'not_college': .65}, 
    # i added .001 to '65+' to make it exactly sum to 1
    'age': {'18-24': .141, '25-34': .194, '35-44': .187, '45-54': .194, '55-64': .160, '65+': .124}
})

## checking that everything looks ok
for i, k in enumerate(refdist.keys()):
    assert bias_vars[i] == k  # names in right order?
for k,v in refdist.items():
    assert set(v.keys()) == set(df[k])  # unique values all correct?
    assert sum(v.values()) == 1.  # each a proper distribution?

ref_aggs = [pd.Series(v).sort_index().values for v in refdist.values()]


# random initial values, using unif(.25,.75) to get probability-like values
ref_init = np.random.uniform(.25,.75, (2,2,6)) 

ipf = ipfn(ref_init, ref_aggs, dims)  # reusing same dims
pop_dist = ipf.iteration()

# %%
# creating weights table
wt_arr = pop_dist / samp_dist
print(wt_arr)

dimnames = [df[col].value_counts().sort_index().index for col in bias_vars] 
wt_df = pd.DataFrame(np.zeros((np.prod(wt_arr.shape), len(refdist) + 1))) # +1 for wt column
wt_df.columns = list(refdist.keys()) + ['wt']

l = 0
for i, f in enumerate(refdist['gender'].keys()):
    for j, e in enumerate(refdist['educ'].keys()):
        for k, a in enumerate(refdist['age'].keys()):
            wt_df.iloc[l,:len(refdist)] = [f,e,a]
            wt_df.iloc[l,len(refdist)] = wt_arr[i,j,k]
            l += 1
            
wt_df
#%%
#Adding the weights back to the data frame
df_wv1 = pd.merge(df, wt_df, on =['gender','educ','age'])
df_wv1['wt'] = df_wv1.wt / df_wv1.wt.mean()
#%%[markdown]







# # Graphing Wave 1








# %%
# picking question to graph
pcvd_wv1 = df_wv1.groupby(df_wv1['precovid'], dropna=False)['wt'].sum().reset_index()
# turning counts into percentages
pcvd_wv1['wt_percent'] = pcvd_wv1['wt']/pcvd_wv1['wt'].sum()
#%%
# graph
plt.barh(pcvd_wv1.precovid, pcvd_wv1.wt_percent)
plt.xlabel('Percent')
plt.title('How often did you eat out pre-covid?')

#%% [markdown]


# # Wave 2



# %%
df = pd.read_csv('C:/Users/matth/environments/dsci1_env/restaurants_covid/Dining Out Survey Batch 3 (Responses) - Form Responses 1.csv')

# dropping duplicate survey takers and columns don't need
df.drop_duplicates('What is your Amazon Worker ID?', inplace=True)
# this is different than wave 1, "unamed 9" not here but is in wave 1??
df.drop(['What is your Amazon Worker ID?', 'Timestamp', 'In what zip code do you live?'], axis=1, inplace=True)

# making human readable names
df.rename(
    {
        'Before the coronavirus crisis, how often did you eat out at a sit-down restaurant?':'precovid',
        'Currently how often do you eat out at a sit-down restaurant?': 'current',
        "Next year how often do you think you'll eat out at a sit-down restaurant?": 'postcovid',
        'What is your gender?': 'gender',
        'What is the highest level of education you have completed?': 'educ'
    },
    axis = 1,
    inplace = True
)
#%% New Question for wave2, renaming
df.rename({'Currently, about how often do you order takeout?':'how_often_takeout'}, axis = 1, inplace = True)
# %%
# get age, bin, and drop year column
df['age'] = pd.cut(2020 - df['In what year were you born?'], bins=[-1, 24, 34, 44, 54, 64, 1000],
       labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
df.drop(['In what year were you born?'], axis=1, inplace=True)



# too few responses to estimate
df = df.loc[df.gender != 'Other', :]
df['educ'] = df.educ.map({'College graduate': 'college', 'Post-graduate (e.g. MS, MBA, PhD, etc.)': 'college',
                    'High school' : 'not_college', 'Did not finish high school': 'not_college'})


# fixing indexing after dropping some rows
df.reset_index(drop=True, inplace=True)

print(df.shape)
print(df.columns)
df.head()
# %%
# Weighting
np.random.seed(1)

bias_vars = ['gender', 'educ', 'age']
samp_aggs = [df[col].value_counts().sort_index().values for col in bias_vars]   # marginal counts
dims = [[i] for i in range(len(samp_aggs))]  # i'm using 1-d marginals
# print(aggs, dims)

# random initial values, using poisson to get count-like values
samp_init = np.random.poisson(np.mean([np.mean(x) for x in samp_aggs]), (2,2,6)) + 1  

ipf = ipfn(samp_init, samp_aggs, dims)
samp_dist = ipf.iteration()
samp_dist = samp_dist / np.sum(samp_dist)  # normalizing it to a proper distribution

# %%
# repeating for population marginals
np.random.seed(1)

from collections import OrderedDict

# ['gender', 'educ', 'age']
refdist = OrderedDict({
    'gender': {'Female': .52, 'Male': .48},  
    'educ':  {'college': .35, 'not_college': .65}, 
    # i added .001 to '65+' to make it exactly sum to 1
    'age': {'18-24': .141, '25-34': .194, '35-44': .187, '45-54': .194, '55-64': .160, '65+': .124}
})

## checking that everything looks ok
for i, k in enumerate(refdist.keys()):
    assert bias_vars[i] == k  # names in right order?
for k,v in refdist.items():
    assert set(v.keys()) == set(df[k])  # unique values all correct?
    assert sum(v.values()) == 1.  # each a proper distribution?

ref_aggs = [pd.Series(v).sort_index().values for v in refdist.values()]


# random initial values, using unif(.25,.75) to get probability-like values
ref_init = np.random.uniform(.25,.75, (2,2,6)) 

ipf = ipfn(ref_init, ref_aggs, dims)  # reusing same dims
pop_dist = ipf.iteration()

# %%
# creating weights table
wt_arr = pop_dist / samp_dist
print(wt_arr)

dimnames = [df[col].value_counts().sort_index().index for col in bias_vars] 
wt_df = pd.DataFrame(np.zeros((np.prod(wt_arr.shape), len(refdist) + 1))) # +1 for wt column
wt_df.columns = list(refdist.keys()) + ['wt']

l = 0
for i, f in enumerate(refdist['gender'].keys()):
    for j, e in enumerate(refdist['educ'].keys()):
        for k, a in enumerate(refdist['age'].keys()):
            wt_df.iloc[l,:len(refdist)] = [f,e,a]
            wt_df.iloc[l,len(refdist)] = wt_arr[i,j,k]
            l += 1
            
wt_df
#%%
#Adding the weights back to the data frame
df_wv2 = pd.merge(df, wt_df, on =['gender','educ','age'])
df_wv2['wt'] = df_wv2.wt / df_wv2.wt.mean()

# %% [markdown]



# # Comparing Waves



#%% Picking question to graph
question = 'current'
#%% Grouping data for graph
q_wv1 = df_wv1.groupby(df_wv1[question], dropna=False)['wt'].sum().reset_index()
q_wv2 = df_wv2.groupby(df_wv2[question], dropna=False)['wt'].sum().reset_index()

q_wv1['wt_percent'] = q_wv1['wt']/q_wv1['wt'].sum()
q_wv2['wt_percent'] = q_wv2['wt']/q_wv2['wt'].sum()
# %% turning the above dfs into single df for graphing

q_wv1_2 = q_wv1[[question, 'wt_percent']].copy()
q_wv1_2['wt_percent_wv2'] = q_wv2['wt_percent']
q_wv1_2

df = pd.melt(q_wv1_2, id_vars=question, var_name='wave' , value_name='percent') #making wave a var

df
#%% Creating graph comparing two waves
sns.catplot(x = 'percent', y = question, hue='wave', data=df, kind='bar' )
plt.xlabel('Percent')
plt.title('How often eat out ' + question + '?')

# #%% Clustering on Age and Frequency dine out
# # NEED TO INCORPORATE SURVEY WEIGHTS INTO CLUSTERS 
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4, random_state=0)



# df2 = df_wv1[['age_years', 'current_freq']].copy()
# kmeans.fit(df2)
# df2['cluster'] = kmeans.labels_
# kmeans.predict([[40, 4]])
# kmeans.cluster_centers_

# # SOMETHING WRONG HERE WITH CLUSTERS. ONLY CLUSTERING ON AGE
# sns.scatterplot(data=df2, x='age_years', y='current_freq', hue='cluster')

#%%
df2 = df_wv1[['age_years', 'current_freq']].copy()
df2.current_freq = df2.current_freq / df2.current_freq.max()
df2.age_years= df2.age_years / df2.age_years.max()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
#df2 = df_wv1[['age_years', 'current_freq']].copy()
#df2 = df_wv1[['current_freq']].copy()

x = np.array(df2)
#kmeans.fit(x)
kmeans.fit(df2)
df2['cluster'] = kmeans.labels_
df2['age_years2'] = df_wv1['age_years']
df2['current_freq2'] = df_wv1.current_freq
#%%
fig = sns.scatterplot(data=df2, x='age_years2', y='current_freq2', hue='cluster')
plt.ylabel("freq eatout per month")
plt.show(fig)

#%%
def cluster_age_chart(clus):
    x_plot = df2.loc[(df2['cluster'] == clus)]
    plt.hist(x_plot.age_years2, bins=30)
    title = ('Cluster', clus)
    plt.title(title)
    plt.show()
    print('median age', x_plot.age_years2.median())
#%%
cluster_age_chart(2)
#%%

plt.hist(df2.age_years2, bins=30)
plt.title('all')
plt.show()

#(plt.hist, 'Age', bins=20)


#%% [markdown]
# ### To-Do
# * Turn weighting into function or just turn the strings into variables. 
# goal here is to easily reuse code for survye weighting
# * 
# See what predicts eating out? Income, age, 
# %% [markdown] <a class="anchor" id="insights"></a>
# ## Insights 
# ### Overview
# As we all know the COVID crisis has been hard on the restaurant industry, especially with 
# the sit-down restaurants. What I
# wanted to know, is it getting better or worse? In June 2020 I ran a survey asking respondents
# how often they eat out at a sit down restaurant. Then I followed up with a new survey in 
# October 2020 to see if it had changed. The survey found that diners are choosing to eat out 
# more often. The number of responents that said they eat out less than once per month declined
# by approximately 7%. 
# ### Methods
# Survey respondents were recruited from Amazon's Mehcanical Turk where they were sent to a 
# simple online survey hosted with Google Forms. Respondents were weighted to match the internet 
# population. 
# #### Weighting Variables
# * Age
# * Gender
# * Education
# * Zip code data is available for each wave and should used for weighting
#
#
# 
# * Wave 1 70% of people ate out less than once per month. 
# * Wave 2 65%, so people are eating out more





# %%
