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
def match_progression(x_df,Id,pipe):
    match = x_df[x_df['match_id'] ==Id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['battingTeam_x','bowlingTeam_x','city_y','runs_left','balls_left','wickets','runs_x_y','crr','rrr','winner','last_five_runs']].fillna(0)
    temp_df = temp_df[temp_df['balls_left'] != 0]
    if temp_df.empty:
        print("Error: Match is not Existed")
        a=1
        return None, None
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['runs_x_y'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
temp_df,target = match_progression(match_df,'1000',pipe)
temp_df
import streamlit as st
st.write(temp_df)

