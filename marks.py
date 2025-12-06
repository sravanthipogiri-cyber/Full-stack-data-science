import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Marks Dashboard")

st.title("📚 Student Marks Dashboard")

file = st.file_uploader("Upload marks CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📌 Marks Table")
    st.dataframe(df)

    df["Total"] = df.iloc[:, 1:].sum(axis=1)
    df["Percentage"] = df["Total"] / (len(df.columns) - 1)  # Subjects count
    df["Percentage"] = df["Percentage"].round(2)

    st.subheader("📌 Student Results")
    st.dataframe(df[["Name", "Total", "Percentage"]])

    st.subheader("📌 Highest & Lowest Scorers")
    highest = df.loc[df["Total"].idxmax()]
    lowest = df.loc[df["Total"].idxmin()]
    st.write("🏆 Highest Scorer:", highest["Name"], "-", highest["Total"])
    st.write("🔻 Lowest Scorer:", lowest["Name"], "-", lowest["Total"])

    st.subheader("📌 Subject Wise Chart")
    subject = st.selectbox("Choose a subject", df.columns[1:-2])
    fig, ax = plt.subplots()
    ax.bar(df["Name"], df[subject])
    st.pyplot(fig)

else:
    st.info("Please upload a marks CSV to continue.")