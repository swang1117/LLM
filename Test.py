import streamlit as st
option = st.selectbox('Select a country', ('USA', 'Canada', 'Mexico'))
st.write('You selected', option)



import streamlit as st
user_input = st.text_input("Enter text")
if st.button("Submit"): 
    st.write('You entered', user_input)

placeholder = st.empty()
with placeholder.container():
    with st.expander("Show me"):
        st.success("You got it. Now you can see more")
    if st.button("Clear", type="primary"):
        placeholder.empty()

import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Upload a CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f'Uploaded CSV file has'+ {df.shape[0]} rows and {df.shape[1]} columns')



import plotly.express as px
import plotly.graph_objects as go
fig = px.histogram(df, x='survived', color='pclass', barmode="group")
st.plotly_chart(fig)
