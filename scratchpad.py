# -*- coding: utf-8 -*-
"""scratchpad

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/notebooks/empty.ipynb
"""

import pandas as pd
import numpy as np

df=pd.read_csv('2nd Innings T20.csv')

first=pd.read_csv('1st Innings T20.csv')

first

df.info()

df['matchId']=df['matchId'].str.split('_').str[0].astype(int)

df=df.dropna()

df.info()

df['runs'] = pd.to_numeric(df['runs'], errors='coerce')

df['Score'] = df.groupby('matchId')['runs'].cumsum()

df

df.drop(columns=[
     'tossWinner'
],inplace=True)

df['over']=df['balls'].astype(str)

df['overs']=df['over'].str.split('.').str.get(0).astype(int)
df['ball']=df['over'].str.split('.').str.get(1).astype(int)

df= df.groupby('matchId').sum()['runs'].reset_index().merge(df,on='matchId')
df['balls_left']=120-(6*df['overs'])-df['ball']

df['crr']=(df['Score']*6)/(120-df['balls_left'])

df.info()

df['runs_y'] = pd.to_numeric(df['runs_y'], errors='coerce')

groups = df.groupby('matchId')

match_ids = df['matchId'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=24)['runs_y'].sum().values.tolist())

df['last_five_runs']=last_five

first= first.groupby('matchId').sum()['runs'].reset_index().merge(first,on='matchId')
first

first['inning']=1
first['id']=first['matchId'].str.split('_').str.get(0).astype(int)

match_df=df.merge(first,left_on='matchId',right_on='id')

match_df.info()

match_df['inning'].unique()

match_df['rrr']=((match_df['runs_x_y']-match_df['Score'])*6)/match_df['balls_left']

match_df['runs_left']=match_df['runs_x_y']-match_df['Score']
match_df=match_df[match_df['runs_left']>=0]
match_df

match_df['x1']=match_df['runs_x_x']-match_df['runs_x_y']

match_df['winner']=match_df['x1'].apply(lambda x:1 if x >= 0 else 0)

df1=match_df[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','player_out_x','runs_x_y','crr','rrr','winner','last_five_runs']]

match_df.info()

x = df1.drop('winner', axis=1)
y = df1['winner']

df1.info()

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['battingTeam_x','bowlingTeam_x','city_y'])],remainder='passthrough')

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

pipe=Pipeline(
    steps=[
        ('step1',trf),
        ('step2',LogisticRegression())
    ])

pipe.fit(xtrain,ytrain)

