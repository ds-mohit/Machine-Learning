# streamlit is used to create a web application
#there is no need ofr css and html
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

#page configuration
st.set_page_config(

    page_title="Streamlit page configuration",
    page_icon="😂",
    layout="wide"
)

#page title
st.title('Hello world')
# 
st.header('Streamlit App')
#to write a subheader
st.subheader('Streamlit is a framwork for Machine Learning and Data Science')
# to write a text
st.write('Hello world')
#to change font style
st.markdown("<h1 style='text-align: center; color: red;'>Hello world</h1>", unsafe_allow_html=True)
# to write a markdown
st.markdown("```Hello world```")
# to write a code
st.code('print("Hello world")')
# to write a latex(formula)
st.latex(r"a^2 + b^2 = c^2")
# to create a line on page
st.divider()
# metrices and message
st.header(" Metrices and Message ")

st.metric(label="Revenue",value=1234,delta="+10%",delta_color="inverse")

st.error("This is an error message")
st.warning("This is a warning message")
st.info("This is an info message")
st.success("This is a success message")
st.exception("This is an exception message")    

st.divider()

st.header(" Data Display")
# creating data
df = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
st.dataframe(df)
st.table(df)
st.json(df.to_dict())

st.divider()

st.header("Charts")
st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)
# st.altair_chart(df)

chart=alt.Chart(df.reset_index()).mark_line().encode(x='index',y='a')
st.altair_chart(chart,use_container_width=True)
fig, ax = plt.subplots()
ax.plot(df.index,df.a)
st.pyplot(fig)