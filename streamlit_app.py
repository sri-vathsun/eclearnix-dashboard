import streamlit as st
import pandas as pd
import plotly.express as px

# Set Streamlit wide layout with custom theme
st.set_page_config(layout="wide", page_title="ECLEARNIX Dashboard", page_icon="📊")

# Custom CSS for soft-colored background and style
st.markdown("""
    <style>
        .stApp {
            background-color: #e6f2ff;
        }
        .main, .block-container {
            background-color: #f0f4f8;
            color: #202030;
            font-family: 'Segoe UI', sans-serif;
            padding: 1rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1f2937;
        }
        .css-1d391kg, .css-1v3fvcr, .css-10trblm {
            color: #0f172a;
        }
    </style>
""", unsafe_allow_html=True)

# Load the clustered or predicted data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/ECLEARNIX_predicted.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv("data/ECLEARNIX_clustered.csv")
        except FileNotFoundError:
            return pd.read_csv("data/ECLEARNIX.csv")

df = load_data()

st.title("📊 ECLEARNIX Data Intelligence Dashboard")

# Sidebar Filters
st.sidebar.header("🔎 Filter Users")
st.sidebar.markdown("Customize your view using the filters below.")
region_filter = st.sidebar.multiselect("🌍 Select Region", options=df['Region'].dropna().unique())
dept_filter = st.sidebar.multiselect("🏢 Select Department", options=df['Department'].dropna().unique())

# Apply filters
if region_filter:
    df = df[df['Region'].isin(region_filter)]
if dept_filter:
    df = df[df['Department'].isin(dept_filter)]

# Display dataset preview
with st.expander("📄 Show Raw Data"):
    st.dataframe(df.head(20), use_container_width=True)

# Define consistent color template
color_template = 'plotly'

# Course Completion
if 'Course_Completed' in df.columns:
    st.subheader("✅ Course Completion Distribution")
    fig = px.histogram(df, x='Course_Completed', color='Course_Completed', title='Course Completion Count', template=color_template)
    st.plotly_chart(fig, use_container_width=True)

# Region-wise Event Registration
if 'Region' in df.columns and 'Registered_for_Event' in df.columns:
    st.subheader("📍 Event Registration by Region")
    fig = px.histogram(df, x='Region', color='Registered_for_Event', barmode='group', title='Event Registration by Region', template=color_template)
    st.plotly_chart(fig, use_container_width=True)

# Feedback Rating Distribution
if 'Feedback_Rating' in df.columns:
    st.subheader("⭐ Feedback Rating Distribution")
    fig = px.histogram(df, x='Feedback_Rating', nbins=10, title='Feedback Rating Distribution', template=color_template)
    st.plotly_chart(fig, use_container_width=True)

# Time Spent by Department
if 'Department' in df.columns and 'Time_Spent_Total_Minutes' in df.columns:
    st.subheader("⏱️ Avg Time Spent by Department")
    avg_time = df.groupby('Department')['Time_Spent_Total_Minutes'].mean().reset_index()
    fig = px.bar(avg_time, x='Time_Spent_Total_Minutes', y='Department', orientation='h', title='Avg Time Spent by Department', template=color_template, color='Time_Spent_Total_Minutes')
    st.plotly_chart(fig, use_container_width=True)

# Newsletter Subscription
if 'Newsletter_Subscribed' in df.columns:
    st.subheader("📧 Newsletter Subscription")
    fig = px.pie(df, names='Newsletter_Subscribed', title='Newsletter Subscription Rate', hole=0.4, template=color_template)
    st.plotly_chart(fig, use_container_width=True)

# App Installation
if 'App_Installed' in df.columns:
    st.subheader("📱 App Installation")
    fig = px.pie(df, names='App_Installed', title='App Installation Rate', hole=0.4, template=color_template)
    st.plotly_chart(fig, use_container_width=True)

# Top 10 Departments by Course Completion
if 'Course_Completed' in df.columns and 'Department' in df.columns:
    st.subheader("🏫 Top Departments by Course Completion")
    top_depts = df[df['Course_Completed'] == 1]['Department'].value_counts().head(10).reset_index()
    top_depts.columns = ['Department', 'Count']
    fig = px.bar(top_depts, x='Department', y='Count', title='Top 10 Departments by Course Completion', template=color_template, color='Count')
    st.plotly_chart(fig, use_container_width=True)

# Clustering Results
if 'Cluster' in df.columns:
    st.subheader("👥 User Segments (Clustered)")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig = px.bar(cluster_counts, x='Cluster', y='Count', title='User Segments (Clusters)', color='Cluster', template=color_template)
    st.plotly_chart(fig, use_container_width=True)

# Predicted Completion Results
if 'Predicted_Completion' in df.columns:
    st.subheader("📌 Predicted Course Completion")
    pred_counts = df['Predicted_Completion'].value_counts().reset_index()
    pred_counts.columns = ['Predicted_Completion', 'Count']
    fig = px.bar(pred_counts, x='Predicted_Completion', y='Count', title='Predicted Course Completions', color='Predicted_Completion', template=color_template)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("🔍 View Sample Predictions"):
        st.dataframe(df[['User_ID', 'Predicted_Completion']].head(), use_container_width=True)