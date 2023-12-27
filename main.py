import streamlit as st
import pickle
import pandas as pd
import numpy as np

final_df = pd.read_csv('C:/Users/SMART COMPUTER/Untitled Folder 5/match.csv')
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
    ('step3', LinearRegression())
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
#input_df = pd.DataFrame(
  #  {'name': [name], 'company': [company], 'year': [year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})
n = pipe.predict(pd.DataFrame(columns=['battingTeam','bowlingTeam','city','Score','ball_left','wickets','crr','last_five','last_five_wickets'],
                              data=np.array([a,b,c,d,e,f,g,h,i]).reshape(1, 9)))
# result = pipe.predict(input_df)

if st.button('Predict'):
    st.subheader('Predicted Score is:'+str(int(n[0])))
