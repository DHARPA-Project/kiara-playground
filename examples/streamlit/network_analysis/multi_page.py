# -*- coding: utf-8 -*-
import streamlit as st

# Create an instance of the app
from kiara_modules.playground.markus.streamlit import MultiPageApp

# Custom imports


app = MultiPageApp()

# Title of the main page
st.title("Data Storyteller Application")

# Add all your applications (pages) here
app.add_page("Upload Data", data_upload.app)
app.add_page("Change Metadata", metadata.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Data Analysis", data_visualize.app)
app.add_page("Y-Parameter Optimization", redundant.app)

# The main app
app.run()
