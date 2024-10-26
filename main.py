import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
sf=pd.read_csv('flags_iso (1).csv')
o=st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 20
h = st.text_input('URL( ESPN CRICINFO >Select Match > Click On Overs )') or 'https://www.espncricinfo.com/series/india-in-west-indies-2023-1381201/west-indies-vs-india-1st-t20i-1381217/match-overs-comparison'
if (h=='https://www.espncricinfo.com/series/india-in-west-indies-2023-1381201/west-indies-vs-india-1st-t20i-1381217/match-overs-comparison'):
    st.write('Enter Your URL')
r = requests.get(h)
#r1=requests.get('https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/india-vs-new-zealand-1st-semi-final-1384437/full-scorecard')
b=BeautifulSoup(r.text,'html')
venue=b.find(class_='ds-flex ds-items-center').text.split(',')[1]
list=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
list7=[]
list8=[]
list9=[]
list10=[]
#print(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[49].text)
elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
for i, element in enumerate(elements):
    if not element.text.split('/'):
        print(' ')
    else:
        if i % 2 != 0:
            list.append(element.text.split('/')[0])
            list1.append(element.text.split('/')[1].split('(')[0])
for i, element in enumerate(elements):
    if element.text.split('/') is None:
        print(' ')
    else:
        if i % 2 == 0:
            list8.append(element.text.split('/')[0])
            list9.append(i/2+1)
            list10.append(element.text.split('/')[1].split('(')[0])
            
dict1={'inng1':list8,'over':list9,'wickets':list10}
df1=pd.DataFrame(dict1)
for i in range(len(list)):
    list2.append(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[i].text)
    list3.append(b.find(class_='ds-text-compact-m ds-text-typo ds-text-right ds-whitespace-nowrap').text.split('/')[0])
    list4.append(b.find_all('th',class_='ds-min-w-max')[1].text)
    list5.append(b.find_all('th',class_='ds-min-w-max')[2].text)
    list6.append(b.find(class_='ds-flex ds-items-center').text.split(',')[1])
    if o==20:
        list7.append(b.find(class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo').text.split(' ')[0])
if o==20:
    dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3,'winner':list7} 
else:
    dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3} 
df=pd.DataFrame(dict)

df['score']=df['score'].astype('int')
df1['inng1']=df1['inng1'].astype('int')
df1['over']=df1['over'].astype('int')
df['over']=df['over'].astype('int')
df['wickets']=df['wickets'].astype('int')
df['target']=df['target'].astype('int')
df['runs_left']=df['target']-df['score']
df=df[df['score']<df['target']]
df['crr']=(df['score']/df['over'])
df['rrr']=((df['target']-df['score'])/(20-df['over']))
df['balls_left']=120-(df['over']*6)
df['runs'] = df['score'].diff()
df['last_10']=df['runs'].rolling(window=10).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_10_wicket']=df['wickets_in_over'].rolling(window=10).sum()
df=df.fillna(20)
#st.write(df)
df['match_id']=100001
neg_idx = df1[df1['inng1']<0].diff().index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]
lf=df
lf=lf[:int(o)]
st.subheader('Scorecard')
o=int(o)
if o != 50:
    # Create a single row with two columns
    col1, col2 = st.columns([1, 1])  # Equal width columns

    with col1:
        bowling_team = df['bowling_team'].unique()[0]
        batting_team = df['batting_team'].unique()[0]

        # Get the URL for the bowling team
        bowling_team_url = sf[sf['Country'] == bowling_team]['URL']
        if not bowling_team_url.empty:
            # Display the bowling team flag and name in the same line
            col_bowling, col_bowling_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_bowling:
                st.image(bowling_team_url.values[0], width=50)  # Adjust width as needed
            with col_bowling_name:
                st.write(f"**{bowling_team}**")

        # Get the URL for the batting team
        batting_team_url = sf[sf['Country'] == batting_team]['URL']
        if not batting_team_url.empty:
            # Display the batting team flag and name in the same line
            col_batting, col_batting_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_batting:
                st.image(batting_team_url.values[0], width=50)  # Adjust width as needed
            with col_batting_name:
                st.write(f"**{batting_team}**")

    with col2:
        # Adjust the layout of col2 to be left-aligned
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)  # Ensure left alignment
        st.write(str(df['target'].unique()[0]) + '/' + str(df1.iloc[-1, 2]))
        st.write('(' + str(df.iloc[o - 1, 5]) + '/' + '20)' + '    ' + str(df.iloc[o - 1, 3]) + '/' + str(df.iloc[o - 1, 4]))
        st.text('crr : ' + str(df.iloc[o - 1, 8].round(2)) + '  rrr : ' + str(df.iloc[o - 1, 9].round(2)))
        st.write(batting_team + ' Required ' + str(df.iloc[o - 1, 7]) + ' runs in ' + str(df.iloc[o - 1, 10]) + ' balls')
        st.markdown("</div>", unsafe_allow_html=True)  # Close the div for left alignment

    # Display teams and results
else:
  col1, col2 = st.columns(2)
  with col1:
    st.write(f"**{df['bowling_team'].unique()[0]}**")
    st.write(f"**{df['batting_team'].unique()[0]}**")
  with col2:
    st.write(str(df['target'].unique()[0]))
    st.write('(' + str(df.iloc[-1, 5]) + '/' + '20)   ' + str(df.iloc[-1, 3]) + '/' + str(df.iloc[-1, 4]))

  if 'winner' in df.columns and not df['winner'].empty:
    winner = df['winner'].unique()
    if len(winner) > 0:
      st.write(winner[0] + ' Won')
    else:
      st.write("Winner information not available.")
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Scatter(x=df1['over'], y=df1['inng1'], mode='lines', line=dict(width=3, color='red'), name=df['bowling_team'].unique()[0]),
    go.Scatter(x=lf['over'], y=lf['score'], mode='lines', line=dict(width=3, color='green'), name=df['batting_team'].unique()[0])
])

fig.update_layout(title='Score Comparison',
                  xaxis_title='Over',  # This is where we specify the x-axis title
                  yaxis_title='Score')
st.write(fig)
gf=df
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
    a = st.selectbox('Batting Team',sorted(batting))
with col2:
    b = st.selectbox('Bowling Team',sorted(batting))
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
    st.subheader('facts')
    st.write('Avg. Runs Scored by '+a+' in last '+str(e/6)+' overs is: '+str(int(b1-a1)))
    st.write('Avg. Runs given by '+b+' in last '+str(e/6)+' overs is: '+str(int(x2-x1)))
    st.write('Avg. Runs Scored in ' + c + ' in last ' + str(e / 6) + ' overs is: ' + str(int(s2 - s1)))
    t=(120-e)/6
    x=[0,d-h,d,n[0]]
    y=[0,t-3,t,20]
    i=px.line(x)
    st.write(i)




