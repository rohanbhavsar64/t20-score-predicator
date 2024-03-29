import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
final_df = pd.read_csv('match.csv')
final_df=final_df.drop(columns=['Unnamed: 0'])
# st.write(df)
X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['battingTeam','bowlingTeam','city'])],remainder='passthrough')
scaler=StandardScaler()
pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',StandardScaler()),
    ('step3', DecisionTreeRegressor())
])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
batting=['West Indies', 'India', 'England', 'South Africa', 'Netherlands',
       'Pakistan', 'New Zealand', 'Australia', 'Afghanistan', 'Ireland',
       'Zimbabwe', 'Bangladesh', 'Sri Lanka']

shar=['Lauderhill', 'St Lucia', 'Bangalore', 'Nottingham', 'Cape Town',
       'Dubai', 'Johannesburg', 'Wellington', 'Dhaka', 'Harare',
       'Hamilton', 'Chandigarh', 'Sharjah', 'Colombo', 'Pallekele',
       'Southampton', 'Abu Dhabi', "St George's", 'Melbourne', 'London',
       'Rawalpindi', 'Trinidad', 'Durban', 'Sylhet', 'Rajkot', 'Auckland',
       'Pune', 'Kolkata', 'Mirpur', 'Nagpur', 'Bristol', 'Kanpur',
       'Sydney', 'Rotterdam', 'Chittagong', 'Bready', 'Dehradun',
       'Potchefstroom', 'Al Amarat', 'The Hague', 'Hyderabad', 'Barbados',
       'Mumbai', 'Providence', 'Lahore', 'Karachi', 'Mount Maunganui',
       'Cardiff', 'Dublin', 'Christchurch', 'St Vincent', 'Bridgetown',
       'Chennai', 'Guyana', 'Manchester', 'Ranchi', 'Thiruvananthapuram',
       'Basseterre', 'Khulna', 'Adelaide', 'Bloemfontein', 'Delhi',
       'Coolidge', 'Greater Noida', 'Hobart', 'Chester-le-Street',
       'St Kitts', 'Hambantota', 'Napier', 'Port Elizabeth', 'Dharamsala',
       'Centurion', 'Perth', 'Bengaluru', 'Cuttack', 'Ahmedabad',
       'Gros Islet', 'Brisbane', 'Bulawayo', 'Birmingham', 'Kandy',
       'Lucknow', 'Visakhapatnam', 'Antigua', 'Leeds', 'Jamaica',
       'Carrara', 'East London', 'Canberra', 'Paarl', 'Victoria',
       'Belfast', 'Dehra Dun', 'Chattogram', 'Dharmasala', 'Derry',
       'Dunedin', 'Dominica', 'Indore', 'Nairobi', 'King City', 'Nelson',
       'Guwahati', 'Kimberley', 'Taunton', 'Edinburgh']
wic=['0','1','2','3','4','5','6','7','8','9']
st.title('T20I Match Score Predictor')
col1,col2,col3=st.columns(3)
with col1:
    a = st.selectbox('batting_team',sorted(batting))
with col2:
    b = st.selectbox('bowling',sorted(batting))
with col3:
    c= st.selectbox('city',sorted(shar))
col1,col2,col3=st.columns(3)
with col1:
    d= int(st.number_input('score'))
with col2:
    f=st.selectbox('wickets',sorted(wic))
with col3:
    g=st.number_input('crr')
col1,col2,col3=st.columns(3)
with col1:
    h=st.number_input('Runs in last three overs')
with col2:
    i=st.selectbox('Wickets in last three overs',sorted(wic))
with col3:
    e= st.number_input('balls left in Inning')
n = pipe.predict(pd.DataFrame(columns=['battingTeam','bowlingTeam','city','Score','ball_left','wickets','crr','last_five','last_five_wickets'],
                              data=np.array([a,b,c,d,e,f,g,h,i]).reshape(1, 9)))
import plotly.express as px
if st.button('Prediction & Analysis') and a!=b:
    st.subheader('Predicted Score is:'+str(int(n[0])))
    m=final_df[final_df['city']==c]['runs_x'].mean()
    data1=np.array([(int(n[0]),'predicted score'),(int(m),'Average score')])
    df=pd.DataFrame(data1,columns=['runs',' '])
    a1=final_df[final_df['battingTeam']==a][final_df[final_df['battingTeam']==a]['ball_left']==e]['Score'].mean()
    b1=final_df[final_df['battingTeam']==a][final_df[final_df['battingTeam']==a]['ball_left']==0]['Score'].mean()
    s1 = final_df[final_df['city'] == c][final_df[final_df['city'] == c]['ball_left'] == e]['Score'].mean()
    s2 = final_df[final_df['city'] == c][final_df[final_df['city'] == c]['ball_left'] == 0]['Score'].mean()
    x1=final_df[final_df['bowlingTeam']==b][final_df[final_df['bowlingTeam']==b]['ball_left']==e]['Score'].mean()
    x2=final_df[final_df['bowlingTeam'] == b][final_df[final_df['bowlingTeam'] == b]['ball_left'] == 0]['Score'].mean()
    fig=px.bar(df,x=' ',y='runs',color_discrete_sequence=['purple','Red'],title='Avg.Total & Predicted Total ')
    st.write(fig)
    df1 = final_df[final_df['battingTeam'] == a][final_df[final_df['battingTeam'] == a]['ball_left'] == 0]
    fig1=px.scatter(df1[df1['bowlingTeam']==b].tail(10),y='Score',color_discrete_sequence=['purple'],title='Totals Against '+b)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title='matches')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig1.update_traces(marker=dict(size=10,
                                   line=dict(width=2,
                                             color='rgba(135, 206, 250, 0.5)')),
                       selector=dict(mode='markers'))
    fig1.update_xaxes(showticklabels=False)
    st.write(fig1)
    df2=final_df[final_df['city'] == c][final_df[final_df['city'] == c]['ball_left'] == 0]
    fig2 = px.line(df2,y='Score', color_discrete_sequence=['Purple'],title='Satuts of Ground in Previous match')
    st.write(fig2)
    st.subheader('facts')
    st.write('Avg. Runs Scored by '+a+' in last '+str(e/6)+' overs is: '+str(int(b1-a1)))
    st.write('Avg. Runs given by '+b+' in last '+str(e/6)+' overs is: '+str(int(x2-x1)))
    st.write('Avg. Runs Scored in ' + c + ' in last ' + str(e / 6) + ' overs is: ' + str(int(s2 - s1)))




