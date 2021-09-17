# -*- coding: utf-8 -*-

import streamlit as st
from kiara import Kiara

# Custom imports
from multipage import MultiPage
from pages import (
    file_onboarding
)

app = MultiPage()

# Title of the main page
st.title("TM Streamlit test")

kiara = Kiara.instance()
st.session_state["kiara"] = kiara

# Add all your application here
app.add_page("1. Onboard data", file_onboarding.app)

# The main app
app.run()
