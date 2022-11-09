import pandas as pd
from numpy import ones, triu
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def box_plot(df, col_columns, col_values, col_index, angle_x=0, ax=None, title="", order=None, yscale='linear',
            ylabel=None, xlabel=""):
    df2 = df.pivot(columns=col_columns, values=col_values, index=col_index)
    if order is not None:
        df2 = df2[order]
    
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    df2.plot.box(ax=ax, title=title, showmeans=True)
    ax.set_xlabel(col_columns)
    if ylabel is None:
        ylabel = col_values
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = angle_x)
    
    ax.set_yscale(yscale)
    
def filter_df(df, filters):
    df2 = df.copy()
    for filter in filters:
        if filter[1] == 'eq':
            df2 = df2[df2[filter[0]] == filter[2]]
        else:
            df2 = df2[df2[filter[0]] != filter[2]]
    return df2    

def line_plot(df, col_columns, col_values, col_index, legend=False, ax=None, title="", order=None, markers=None,
             yscale='linear', ylabel=None, xlabel=""):
    df2 = df.pivot(columns=col_columns, values=col_values, index=col_index)
    if order is not None:
        df2 = df2[order]
    
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    
    #ax = df2.plot(legend=legend)
    df2.plot(legend=legend, ax=ax, title=title)
    if ylabel is None:
        ylabel = col_values
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    if markers is not None:
        for i, line in enumerate(ax.get_lines()):
            line.set_marker(markers[i])
    
    ax.set_yscale(yscale)
    
    if legend:
        return plt.legend(bbox_to_anchor=(1.05, 1.05))
        
def heatmap_plot(df, col_columns, col_values, col_index, legend=False, order=None, ylabel=None, xlabel=""):
    df2 = df.pivot(columns=col_columns, values=col_values, index=col_index)
    if order is not None:
        df2 = df2[order]
    vorder = [vec for vec in vectorizers_order2 if vec in df2.index]
        
    df2 = df2.loc[vorder]
    df2 ['Total'] = df2.apply(lambda x: x.mean(), axis=1)
    df2 = df2.rank(ascending=False, method='min')
    ax = sns.heatmap(df2, annot=True, cbar=legend)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    #plt.show()
    
def corr_plot(df, col_columns, col_values, col_index, order=None, figsize=(10,10)):
    df2 = df[[col_columns, col_values, col_index]]
    
    df2['Total'] = df2.apply(lambda x: (x[col_columns], x[col_values]), axis=1)
    df2 = df2.groupby(col_index)['Total'].apply(list)
    df2 = df2.apply(lambda x: sorted(x, key=lambda x: x[0]))
    df2 = df2.apply(lambda x: [y[1] for y in x])

    if order is not None:
        df2 = df2.loc[order]
        
    no = df2.shape[0]
    pearsonr = ones((no, no))

    for i in range(no):
        for j in range(i+1,no):
            pearsonr[i,j] = pearsonr[j,i] = scipy.stats.pearsonr(df2[df2.index[i]], df2[df2.index[j]])[0]

    matrix = triu(pearsonr)
    fig, ax = plt.subplots(figsize=figsize)            
    ax = sns.heatmap(pearsonr, annot=True, mask=matrix, ax=ax)
    ax.set_xticks(range(no), df2.index, rotation=90)
    ax.set_yticks(range(no), df2.index, rotation=0)    
    
vectorizers_order = ['word2vec', 'fasttext', 'glove', 
                     'bert', 'albert', 'roberta', 'distilbert', 'xlnet', 
                       'smpnet', 'st5', 'sdistilroberta', 'sminilm',
                      ]

vectorizers_order2 = ['WC', 'FT', 'GE', 'BT', 'AT', 'RA', 'DT', 'XT', 'ST', 'S5', 'SA', 'SM']