# -*- coding: utf-8 -*-

import streamlit as st
from kiara import Kiara

# Custom imports
from multipage import MultiPage
from pages import (
    files_onboarding,
    get_file_metadata,
    text_preprocessing,
    timestamped_corpus,
    tokenization,
)

app = MultiPage()

# Title of the main page
st.title("TM Streamlit test")

kiara = Kiara.instance()
st.session_state["kiara"] = kiara

# Add all your application here
app.add_page("1. Onboard data", files_onboarding.app)
app.add_page("2. Get file metadata", get_file_metadata.app)
app.add_page("3. Timestamped data", timestamped_corpus.app)
app.add_page("4. Tokenization", tokenization.app)
app.add_page("5. Text pre-processing", text_preprocessing.app)

# The main app
app.run()
