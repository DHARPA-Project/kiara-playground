# -*- coding: utf-8 -*-

import streamlit as st
from kiara import Kiara

# Custom imports
from multipage import MultiPage
from pages import (
    file_onboarding, map_selection, map
)

app = MultiPage()

# Title of the main page
st.title("Geolocation Streamlit prototype")

kiara = Kiara.instance()
st.session_state["kiara"] = kiara

# Add all your application here
app.add_page("1. Onboard data", file_onboarding.app)
app.add_page("2. Select map", map_selection.app)
app.add_page("3. Map statistics", map.app)

# The main app
app.run()
