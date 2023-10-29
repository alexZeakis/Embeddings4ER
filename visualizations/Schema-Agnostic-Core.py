#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib

from pareto_functions import *
from utils import box_plot, filter_df, vectorizers_order2, line_plot, heatmap_plot, corr_plot
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True

import plotly.express as px
import plotly.io as pio

input_dir_sa = '../logs/schema_agnostic/core/'
input_dir_bl = '../logs/baseline/'
case_order = [f'D{i}' for i in range(1, 11)]
case_order2 = [f'DSM{i}' for i in range(1, 6)]

case_map = {'rest1_rest2': 'D1', 'abt_buy': 'D2', 'amazon_gp': 'D3',
            'dblp_acm': 'D4', 'imdb_tvdb': 'D5', 'tmdb_tvdb': 'D6', 'imdb_tmdb': 'D7',
            'walmart_amazon': 'D8', 'dblp_scholar': 'D9', 'imdb_dbpedia': 'D10' }

sizes = {'D1': 339*2256, 'D2': 1076*1076, 'D3': 1354*3039, 'D4': 2616*2294, 'D5': 5118*6056,
         'D6': 5118*7810, 'D7': 6056*7810, 'D8': 2554*22074, 'D9': 2516*61353, 'D10': 27615*23182}

categories = {'AT':'BERT', 'BT':'BERT', 'DT':'BERT', 'RA':'BERT', 'XT':'BERT', 
              'S5':'SBERT', 'SA':'SBERT', 'SM':'SBERT', 'ST':'SBERT',
              'FT':'Static', 'GE':'Static', 'WC':'Static', }

f_sch = ('Columns', 'ne', 2)
f_agn = ('Columns', 'eq', 2)
f_q2i = ('Direction', 'eq', 'q2i')
f_i2q = ('Direction', 'eq', 'i2q')
f_exa = ('Exact', 'eq', 'exact')
f_app = ('Exact', 'eq', 'approx')
f_k = ('k', 'eq', 10)


# # Real Data

# ## Vectorization

# In[2]:


with open(input_dir_sa + 'vectorization_real.txt') as f:
    lines = []
    for line in f.readlines():
        line = json.loads(line)
        
        lines.append((line['dir'], line['vectorizer'], line['init_time'], line['time'], line['file'],
                      line['column']['name'], line['memory']['process']['rss'],
                      line['memory']['total']['used']))
    
vec_df = pd.DataFrame(lines, columns=['Case', 'Vectorizer', 'Init Time', 'Total Time', 'File',
                                        'Column', 'Memory Process', 'Memory Total'])

vec_df['Case'] = vec_df['Case'].apply(lambda x: x.split('(')[0])
vec_df['Vectorizer'] = vec_df['Vectorizer'].apply(lambda x: x[0]+x[-1]).str.upper()


# # Load Data

# ## Blocking

# In[3]:


block_df = pd.read_csv(input_dir_sa + 'blocking_euclidean_real.csv')

block_df['Vectorizer'] = block_df['Vectorizer'].apply(lambda x: x[0]+x[-1]).str.upper()
block_df['F1'] = block_df.apply(lambda x: 2*x['Recall']*x['Precision'] / (x['Recall']+x['Precision']) if x['Recall']+x['Precision'] > 0 else 0, axis=1)
print(block_df.shape)

tuples = [('D1', 0), ('D2', 0), ('D3', 0), ('D4', 0), ('D5', 1), ('D6', 0), ('D7', 0), ('D8', 0), ('D9', 0), ('D10', 0), 
          ('D1', 2), ('D2', 2), ('D3', 2), ('D4', 2), ('D5', 2), ('D6', 2), ('D7', 2), ('D8', 2), ('D9', 2), ('D10', 2)]

block_df = block_df.loc[block_df.apply(lambda x: (x['Case'], x['Columns']) in tuples, axis=1)]
print(block_df.shape)
block_df = filter_df(block_df, [f_i2q, f_exa])
print(block_df.shape)
block_df


# In[4]:


import json
import math
import statistics
import pandas as pd

base_stats = []
with open(input_dir_bl + 'TokenJoin.txt') as f:
    for line in f:
        j = json.loads(line)
        j['vec'] = 'TokenJoin'
        base_stats.append(j)
with open(input_dir_bl + 'JedAI.txt') as f:
    for line in f:
        j = json.loads(line)
        j['vec'] = 'kNN-Join'
        base_stats.append(j)
with open(input_dir_bl + 'Sparkly.txt') as f:
    for line in f:
        j = json.loads(line)
        j['vec'] = 'Sparkly'
        base_stats.append(j)        

temp_block_df = block_df.loc[block_df.Vectorizer =='st5']
temp_block_df = temp_block_df[['Case', 'k', 'Time', 'Precision', 'Recall', 'Vectorizer']]
temp_block_df.columns = ['case', 'k', 'time', 'prec', 'rec', 'vec']
temp_block_df.vec = 'S-GTR-T5'

        
deepblocker_df = []
with open(input_dir_bl + 'DeepBlocker.txt') as f:
    for line in f:
        deepblocker_df.append(json.loads(line))
deepblocker_df = pd.DataFrame(deepblocker_df)
deepblocker_df.columns = ['k', 'time', 'rec', 'prec', 'cands', 'col', 'case']
deepblocker_df['vec'] = 'DeepBlocker'
deepblocker_df


base_stats = pd.DataFrame(base_stats)
base_stats = pd.concat([base_stats, temp_block_df, deepblocker_df])
base_stats


# ## Unsup Matching

# In[5]:


match_df = pd.read_csv(input_dir_sa + 'matching_unsupervised_euclidean.csv')
match_df['Vectorizer'] = match_df['Vectorizer'].apply(lambda x: x[0]+x[-1]).str.upper()
match_df_best = match_df.loc[match_df.groupby(['Case', 'Vectorizer'])['F1'].idxmax().values].reset_index(drop=True)
match_df_last = match_df.loc[match_df.groupby(['Case', 'Vectorizer'])['Delta'].idxmin().values].reset_index(drop=True)
match_df


# In[6]:


match_block_df = pd.read_csv(input_dir_sa + 'matching_unsupervised_euclidean_block.csv')
match_block_df['Vectorizer'] = match_block_df['Vectorizer'].apply(lambda x: x[0]+x[-1]).str.upper()
#match_block_df = match_block_df.loc[match_block_df.groupby(['Case', 'Vectorizer'])['F1'].idxmax().values].reset_index(drop=True)
match_block_df = match_block_df.loc[match_block_df.groupby(['Case', 'Vectorizer'])['Delta'].idxmin().values].reset_index(drop=True)
match_block_df


# In[7]:


zeroer_df = []
with open(input_dir_bl + 'ZeroER.txt') as f:
    for line in f:
        zeroer_df.append(json.loads(line))
zeroer_df = pd.DataFrame(zeroer_df)
zeroer_df.dataset = zeroer_df.dataset.map(lambda x: case_map[x])
zeroer_df = zeroer_df.set_index('dataset')
#zeroer_df = zeroer_df.loc[case_order]
zeroer_df


# ## Supervised Matching

# In[8]:


plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})
lines = []
with open(input_dir_sa + 'matching_supervised_dynamic.txt') as f:
    for line in f:
        lines.append(json.loads(line))
with open(input_dir_sa + 'matching_supervised_static.txt') as f:
    for line in f:
        lines.append(json.loads(line))
        
match_df_sup = pd.DataFrame(lines)
dirs = {'abt_buy' : 'DSM1', 'dirty_amazon_itunes' : 'DSM2', 'dirty_dblp_acm' : 'DSM3',
        'dirty_dblp_scholar' : 'DSM4', 'dirty_walmart_amazon' : 'DSM5'}

match_df_sup['data_name'] = match_df_sup['data_name'].apply(lambda x: dirs[x])
match_df_sup['model_type'] = match_df_sup['model_type'].apply(lambda x: x[0]+x[-1]).str.upper()
match_df_sup['f1'] = match_df_sup['f1'].apply(lambda x: x if x <= 1.0 else x/100)

match_df_sup


# In[9]:


sup_sota_df = []
with open(input_dir_bl + 'supervised_sota.txt') as f:
    for line in f:
        sup_sota_df.append(json.loads(line))
sup_sota_df = pd.DataFrame(sup_sota_df)
sup_sota_df['model_type'] = sup_sota_df['model_type'].apply(lambda x: x[0]+x[-1]).str.upper()
sup_sota_df


# # Effectiveness

# ## Blocking

# ### Blocking.Spider Plot

# In[10]:


import plotly.graph_objects as go
import plotly.express as px

plt.rcParams.update({'font.size': 20, 'font.weight': 'normal'})

#for k in (1, 5, 10):
for k in [10]:    
    
    block_df2 = filter_df(block_df, [('k', 'eq', k), f_agn])
    block_df2 = block_df2.pivot(index='Case', columns='Vectorizer', values='Recall')

    total_cols = {'static': ['WC', 'FT', 'GE'],
                  'BERT': ['BT', 'AT', 'RA', 'DT', 'XT'],
                  'SBERT': ['ST', 'S5', 'SA', 'SM']}

    theta_case_order = case_order + [case_order[0]]
    for grp, cols in total_cols.items():
        fig = go.Figure()
        fig.update_layout(font=dict(size=25,), 
                          polar=dict(radialaxis=dict(dtick=0.2), angularaxis=dict(direction="clockwise", rotation=90)),
                          legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
                         )

        for col in cols:
            r = list(block_df2[col][case_order].values)
            fig.add_trace(go.Scatterpolar(r=r+ [r[0]], theta=theta_case_order, name=col))
        
        #if k != 1:
        #    fig.update_layout(showlegend=False)
        fig.show()
        path = f"../plots_2/effectiveness/blocking/blocking_{k}_{grp}.pdf"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.write_image(path)
        
    block_df3 = pd.DataFrame()
    block_df3['S5'] = block_df2['S5'].loc[case_order]
    block_df3['DR'] = deepblocker_df.loc[deepblocker_df.k==10].set_index('case')['rec'].loc[case_order]

    theta_case_order = case_order + [case_order[0]]
    fig = go.Figure()
    fig.update_layout(font=dict(size=25,), 
                      polar=dict(radialaxis=dict(dtick=0.2), angularaxis=dict(direction="clockwise", rotation=90)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
                     )

    for col in block_df3.columns:
        r = list(block_df3[col][case_order].values)
        fig.add_trace(go.Scatterpolar(r=r+ [r[0]], theta=theta_case_order, name=col))

    #if k != 1:
    #    fig.update_layout(showlegend=False)        
    fig.show()
    fig.write_image(f"../plots_2/effectiveness/blocking/blocking_{k}_sota.pdf")           
        


# ### Blocking.Heatmap

# In[11]:


plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})
block_df2 = filter_df(block_df, [f_k])
print(block_df2.shape)

block_df22 = filter_df(block_df2, [f_agn])
heatmap_plot(block_df22, 'Case', 'Recall', 'Vectorizer', order=case_order, legend=True, reverse_color=True)
plt.savefig(f'../plots_2/effectiveness/blocking/blocking_real_vec_exact_agn_heat.pdf', bbox_inches='tight')

block_df2 = filter_df(block_df, [f_agn, f_k])
print(block_df2.shape)

corr_plot(block_df2, 'Case', 'Recall', 'Vectorizer', order=vectorizers_order2, figsize=(10, 6), reverse_color=True)
plt.savefig(f'../plots_2/effectiveness/blocking/blocking_real_vec_exact_agn_corr.pdf', bbox_inches='tight')


# ## Matching

# ### Unsupervised

# ### Unsupervised.Spider Plot

# In[12]:


import plotly.graph_objects as go

plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})

for metric in ['Precision', 'Recall', 'F1']:
    match_df2 = match_df_best.pivot(index='Case', columns='Vectorizer', values=metric)

    total_cols = {'static': ['WC', 'FT', 'GE'],
                  'BERT': ['BT', 'AT', 'RA', 'DT', 'XT'],
                  'SBERT': ['ST', 'S5', 'SA', 'SM']}

    theta_case_order = case_order + [case_order[0]]
    for grp, cols in total_cols.items():
        fig = go.Figure()
        fig.update_layout(font=dict(size=25,),
                          polar=dict(radialaxis=dict(dtick=0.2), angularaxis=dict(direction="clockwise", rotation=90)),
                          legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
                         )
        for col in cols:
            r = list(match_df2[col][case_order].values)
            fig.add_trace(go.Scatterpolar(r=r+[r[0]], theta=theta_case_order, name=col))

        #if metric != 'Precision':
        #    fig.update_layout(showlegend=False)
        fig.show()
        path = f"../plots_2/effectiveness/unsup_matching/unsup_matching_{metric}_{grp}.pdf"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.write_image(path)
        
    match_block_df2 = match_block_df.pivot(index='Case', columns='Vectorizer', values=metric)
        
    match_df3 = pd.DataFrame()
    match_df3['S5'] = match_block_df2['S5']
    match_df3['ZR'] = zeroer_df[metric.lower()]
    match_df3 = match_df3.fillna(0) # what about missing values?

    theta_case_order = case_order + [case_order[0]]
    fig = go.Figure()
    fig.update_layout(font=dict(size=25,), 
                      polar=dict(radialaxis=dict(dtick=0.2), angularaxis=dict(direction="clockwise", rotation=90)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
                     )

    for col in match_df3.columns:
        r = list(match_df3[col][case_order].values)
        fig.add_trace(go.Scatterpolar(r=r+ [r[0]], theta=theta_case_order, name=col))

    #if metric != 'Precision':
    #    fig.update_layout(showlegend=False)
    fig.show()
    fig.write_image(f"../plots_2/effectiveness/unsup_matching/unsup_matching_{metric}_sota.pdf")     


# ### Unsupervised.Heatmap

# In[13]:


plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})
heatmap_plot(match_df_best, 'Case', 'F1', 'Vectorizer', order=case_order, reverse_color=True)
plt.savefig(f'../plots_2/effectiveness/unsup_matching/matching_unsupervised_heat.pdf', bbox_inches='tight')

plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})

corr_plot(match_df_best, 'Case', 'F1', 'Vectorizer', order=vectorizers_order2, figsize=(10, 6), reverse_color=True)
plt.savefig(f'../plots_2/effectiveness/unsup_matching/matching_unsupervised_corr.pdf', bbox_inches='tight')
plt.show()


# ## Supervised

# ### Supervised.Spider plots

# In[14]:


import plotly.graph_objects as go

plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})

match_df_sup2 = pd.concat([match_df_sup[['f1', 'model_type', 'data_name']], sup_sota_df]).reset_index(drop=True)
match_df_sup2 = match_df_sup2.pivot(index='data_name', columns='model_type', values='f1')

total_cols = {'static': ['FT', 'GE'],
              'BERT': ['BT', 'AT', 'RA', 'DT', 'XT'],
              'SBERT': ['ST', 'SA', 'SM'],
              'sota': ['RA', 'DO', 'D+'],}

theta_case_order = case_order2 + [case_order2[0]]
for grp, cols in total_cols.items():
    fig = go.Figure()
    fig.update_layout(font=dict(size=25,), 
                      polar=dict(radialaxis=dict(dtick=0.2), angularaxis=dict(direction="clockwise", rotation=90)),
                     legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
                     )

    for col in cols:
        r = list(match_df_sup2[col][case_order2].values)
        fig.add_trace(go.Scatterpolar(r=r+[r[0]], theta=theta_case_order, name=col))

    fig.show()
    path = f"../plots_2/effectiveness/sup_matching/sup_matching_{grp}.pdf"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_image(path)


# # Efficiency

# ## Vectorization

# In[15]:


vec_df2 = vec_df.loc[vec_df.Column == 'aggregate value'].reset_index(drop=True)
vec_df2.File = vec_df2.File.apply(lambda x: x[:-4])

vec_df21 = vec_df2.loc[vec_df2['Case'] != 'D5_D6_D7']
vec_df22 = vec_df2.loc[vec_df2['Case'] == 'D5_D6_D7']

cases = {'D5': ('imdb', 'tvdb'), 'D6': ('tmdb', 'tvdb'), 'D7': ('imdb', 'tmdb'),}

for k, v in cases.items():
    for vv in v:
        temp = vec_df22.loc[vec_df22.File == vv].copy()
        temp['Case'] = k
        vec_df21 = pd.concat([vec_df21, temp])
vec_df2 = vec_df21.groupby(['Case', 'Vectorizer'])['Total Time'].sum()
vec_df2 = vec_df2.unstack()
vec_df2 = vec_df2.reset_index(drop=False)
vec_df2['Time'] = 'Vectorization'
#vec_df2 = vec_df2.set_index(['Case', 'Time'])

#tables_times = pd.concat([vec_df2, block_df2, match_df2])
tables_times = vec_df2
tables_times = tables_times.set_index(['Case', 'Time'])
#tables_times.sort_index()
tables_times = tables_times.loc[case_order][vectorizers_order2]
tables_times = tables_times.round(1)

tables_times = tables_times.droplevel(1)

print(tables_times.to_latex())

tables_times


# In[16]:


init_times = vec_df21.groupby('Vectorizer')['Init Time'].mean()
init_times = pd.DataFrame(init_times).T
init_times = init_times[vectorizers_order2]
init_times = init_times.round(2)

print(init_times.to_latex())
init_times


# In[17]:


vect_total_times = pd.DataFrame()
for col in tables_times:
     vect_total_times[col] = tables_times[col]+init_times[col]['Init Time']
vect_total_times


# ## Blocking

# ### Blocking.Pareto

# In[18]:


plt.rcParams.update({'font.size': 18, 'font.weight': 'normal'})

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

block_pareto = pd.DataFrame()

block_df2 = filter_df(block_df, [f_k, f_agn])
block_df2 = block_df2.pivot(index='Case', columns='Vectorizer', values='Time')
block_df2 = block_df2+vect_total_times
block_df2 = block_df2.apply(lambda x: x/x.min(), axis=1) #normalize per dataset
block_pareto['Normalized Time'] = block_df2.mean()

block_df2 = filter_df(block_df, [f_k, f_agn])
block_df2 = block_df2.pivot(index='Case', columns='Vectorizer', values='Recall')
block_pareto['(a) - Recall'] = block_df2.mean()

block_pareto['Size'] = 20
block_pareto['Group'] = pd.Series(categories)

block_pareto = block_pareto.sort_values(by=['Group'], key=lambda x: x.map({v: i for i, v in enumerate(['Static', 'BERT', 'SBERT'])}))

fig1 = px.scatter(block_pareto, y='Normalized Time', x='(a) - Recall', hover_name=block_pareto.index,
                  size='Size', color='Group', text=block_pareto.index)
fig1.update_layout(yaxis_range=[0,25], xaxis_range=[0,1], font=dict(size=20,), 
                  legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="center", x=0.5, title=''))
fig1.show()

path = f"../plots_2/efficiency/blocking_pareto.pdf"
os.makedirs(os.path.dirname(path), exist_ok=True)
fig1.write_image(path)


# In[19]:


df1 = deepblocker_df[['time', 'k', 'case']].pivot(values='time', columns='k', index='case')
df2 = block_df.loc[block_df.Vectorizer=='S5'].pivot(index='Case', columns='k', values='Time')
df2 = df2.apply(lambda x: x.add(vect_total_times['S5']))
sota_blocking_table = pd.concat([df1, df2], axis=1)

sota_blocking_table.columns = pd.MultiIndex.from_product([['DeepBlocker', 'S-T5'], [1,5,10]])
sota_blocking_table = sota_blocking_table.round(1)

print(sota_blocking_table.to_latex())


# ## Unsup Matching

# In[20]:


match_df_bars = match_df_best.pivot(index='Case', columns='Vectorizer', values='Matching Time').loc[case_order, vectorizers_order2]
match_df_bars = match_df_bars.round(1)
print(match_df_bars.to_latex())

match_df_bars


# ### UnsupMatching.Pareto

# In[21]:


plt.rcParams.update({'font.size': 18, 'font.weight': 'normal'})

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

unsup_match_pareto = pd.DataFrame()

match_df_2 = match_df_best.pivot(index='Case', columns='Vectorizer', values='Matching Time')
match_df_2 = match_df_2+vect_total_times
match_df_2 = match_df_2.apply(lambda x: x/x.min(), axis=1) #normalize per dataset
unsup_match_pareto['Normalized Time'] = match_df_2.mean()

match_df_2 = match_df_best.pivot(index='Case', columns='Vectorizer', values='F1')
unsup_match_pareto['(b) - F1'] = match_df_2.mean()

unsup_match_pareto['Size'] = 20
unsup_match_pareto['Group'] = pd.Series(categories)
unsup_match_pareto = unsup_match_pareto.sort_values(by=['Group'], key=lambda x: x.map({v: i for i, v in enumerate(['Static', 'BERT', 'SBERT'])}))

fig1 = px.scatter(unsup_match_pareto, y='Normalized Time', x='(b) - F1', hover_name=unsup_match_pareto.index,
                  size='Size', color='Group', text=unsup_match_pareto.index)
fig1.update_layout(yaxis_range=[0,25], xaxis_range=[0,1], font=dict(size=20,), 
                  legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="center", x=0.5, title=''))
fig1.show()
fig1.write_image(f"../plots_2/efficiency/unsup_matching_pareto.pdf")


# ### UnsupMatching.SotA

# In[22]:


df1 = zeroer_df[['features', 'zeroer']]
df2 = match_block_df.set_index('Case')[['Blocking Time', 'Matching Time']]
df2['Blocking Time'] = df2['Blocking Time'] + vect_total_times['S5']

match_df3 = pd.concat([df1, df2], axis=1)
match_df3 = match_df3.round(3)
match_df3 = match_df3.fillna('-')
match_df3 = match_df3.loc[case_order]
match_df3.columns = pd.MultiIndex.from_product([['ZeroER', 'S-T5'], ['Preprocessing', 'Matching']])

print(match_df3.to_latex())

match_df3


# ## Supervised

# ### Table

# In[23]:


order = [[f'DSM{i}', j] for i in range(1, 6) for j in ['training_time', 'testing_time']]
match_df_sup_tab = match_df_sup.pivot(index='model_type', columns='data_name', values=['training_time', 'testing_time'])
match_df_sup_tab = match_df_sup_tab.round(1)
c = [v for v in vectorizers_order2 if v not in ['WC', 'S5']]
match_df_sup_tab = match_df_sup_tab.swaplevel(axis=1)
match_df_sup_tab = match_df_sup_tab.loc[c, order]
match_df_sup_tab.columns = pd.MultiIndex.from_product([[f'DSM{i}' for i in range(1, 6)], ['$t_t$', '$t_e$']])

print(match_df_sup_tab.to_latex())
match_df_sup_tab


# ### SupMatching.Pareto

# In[24]:


plt.rcParams.update({'font.size': 18, 'font.weight': 'normal'})

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

sup_match_pareto = pd.DataFrame()

match_df_sup['total_time'] = match_df_sup['training_time'] + match_df_sup['testing_time']

match_df_2 = match_df_sup.pivot(index='data_name', columns='model_type', values='testing_time')
match_df_2 = match_df_2.apply(lambda x: x/x.min(), axis=1) #normalize per dataset
sup_match_pareto['Normalized Time'] = match_df_2.mean()

match_df_2 = match_df_sup.pivot(index='data_name', columns='model_type', values='f1')
sup_match_pareto['(c) - F1'] = match_df_2.mean()

sup_match_pareto['Size'] = 20
sup_match_pareto['Group'] = pd.Series(categories)
sup_match_pareto = sup_match_pareto.sort_values(by=['Group'], key=lambda x: x.map({v: i for i, v in enumerate(['Static', 'BERT', 'SBERT'])}))

fig1 = px.scatter(sup_match_pareto, y='Normalized Time', x='(c) - F1', hover_name=sup_match_pareto.index,
                  size='Size', color='Group', text=sup_match_pareto.index)
fig1.update_layout(yaxis_range=[0,10], xaxis_range=[0,1], font=dict(size=20,), 
                  legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="center", x=0.5, title=''))
fig1.show()
fig1.write_image(f"../plots_2/efficiency/sup_matching_pareto.pdf")


# ## Scalability

# ## Vectorization

# In[25]:


files = ['10K', '50K', '100K', '200K', '300K', '1M', '2M']
markers = ['*', 'o', 'o', 'o', '*', '*', '^', '^', 'o', '*', '^', '*']


plt.rcParams.update({'font.size': 15, 'font.weight': 'normal'})
with open(input_dir_sa +'vectorization_synthetic.txt') as f:
    lines = []
    for line in f.readlines():
        line = json.loads(line)
        lines.append((line['vectorizer'], line['time'], line['file'], line['memory']['process']['rss']))
    
vec_df2 = pd.DataFrame(lines, columns=['Vectorizer', 'Total Time', 'File', 'Memory Process'])
vec_df2['File'] = vec_df2['File'].apply(lambda x: x.split('.')[0])
vec_df2['File'] = vec_df2['File'].apply(lambda x: int(x[:-1])* (1000 if x[-1] == 'K' else 1000000) )
vec_df2['Vectorizer'] = vec_df2['Vectorizer'].apply(lambda x: x[0]+x[-1]).str.upper()

fig, axes = plt.subplots(nrows=1, ncols=1)
line_plot(vec_df2, 'Vectorizer', 'Total Time', 'File', legend=False,
          ax=axes, markers=markers, yscale='log', xlabel='', order=vectorizers_order2,
              ylabel='', ylim=(0.1, pow(10, 4))
             )

path = f'../plots_2/scalability/vectorization_synthetic_time.pdf'
os.makedirs(os.path.dirname(path), exist_ok=True)
plt.savefig(path, bbox_inches='tight')


# ## Blocking

# In[26]:


files = ['10K.csv', '50K.csv', '100K.csv', '200K.csv', '300K.csv', '1M.csv', '2M.csv']
markers = ['*', 'o', 'o', 'o', '*', '*', '^', '^', 'o', '*', '^', '*']

block_df2 = pd.read_csv(input_dir_sa + 'blocking_euclidean_synthetic.csv')
#results.Case = results.Case.apply(lambda x: files[x].split('.')[0])
block_df2.Case = block_df2.Case.apply(lambda x: files[x])
block_df2['Case'] = block_df2['Case'].apply(lambda x: x.split('.')[0])
block_df2['Case'] = block_df2['Case'].apply(lambda x: int(x[:-1])* (1000 if x[-1] == 'K' else 1000000) )
block_df2['Vectorizer'] = block_df2['Vectorizer'].apply(lambda x: x[0]+x[-1]).str.upper()
print(block_df2.shape)
block_df2 = filter_df(block_df2, [('k', 'eq', 10), f_i2q])
print(block_df2.shape)
block_df2


# In[27]:


block_df3 = filter_df(block_df2, [f_app])
fig, axes = plt.subplots(nrows=1, ncols=1)
line_plot(block_df3, 'Vectorizer', 'Recall', 'Case', legend=False, markers=markers, ax=axes, title='', ylabel='', order=vectorizers_order2)
plt.savefig(f'../plots_2/scalability/blocking_synthetic_recall_approx.pdf', bbox_inches='tight')

fig, axes = plt.subplots(nrows=1, ncols=1)
line_plot(block_df3, 'Vectorizer', 'Precision', 'Case', legend=False, markers=markers, ax=axes, title='', ylabel='', order=vectorizers_order2)
plt.savefig(f'../plots_2/scalability/blocking_synthetic_precision_approx.pdf', bbox_inches='tight')

fig, axes = plt.subplots(nrows=1, ncols=1)
leg = line_plot(block_df3, 'Vectorizer', 'Time', 'Case', legend=True, markers=markers, ax=axes, title='', yscale='log', ylabel='', order=vectorizers_order2, ylim=(0.1, pow(10, 4)))
leg.set_visible(False)
plt.savefig(f'../plots_2/scalability/blocking_synthetic_time_approx.pdf', bbox_inches='tight')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig = plt.figure()
fig.add_subplot(111)
    
#leg = [Patch(facecolor=c, alpha=1.0, label=v, edgecolor='#000') for c, v in zip(2*colors, vectorizers_order2)]

leg = [ Line2D([], [], color=c, marker=m, markersize=10, label=v)
         for c, v, m in zip(2*colors, vectorizers_order2, 2*markers)]
#leg = fig.legend(handles=leg, ncol=len(leg), bbox_to_anchor=(1.5, 1.5))    
leg = fig.legend(handles=leg, ncol=len(leg)//2, bbox_to_anchor=(1.5, 1.5))    
#leg = fig.legend(handles=leg, bbox_to_anchor=(1.5, 1.5))    

leg.set_visible(True)

expand=[-5,-5,5,5]
fig  = leg.figure
fig.canvas.draw()
bbox  = leg.get_window_extent()
bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
fig.savefig('../plots_2/scalability/blocking_synthetic_legend.pdf', dpi="figure", bbox_inches=bbox)    

plt.show()

