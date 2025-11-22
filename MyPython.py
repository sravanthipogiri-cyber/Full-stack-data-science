import streamlit as st
st.title("My First Streamlit App created by SRAVANTHI")
st.write("Welcome!This app calculates the cube of a number")
st.header("Select a Number")
number=st.slider("Pick a number",0,100,25)
st.subheader("Result")
cube_number=number*number*number
st.write(f"The cube of **{number}** is **{cube_number}**.")
