import pandas as pd
import streamlit as st
df=pd.read_csv('pornstars_0.csv')
st.image(df['1'][0])
