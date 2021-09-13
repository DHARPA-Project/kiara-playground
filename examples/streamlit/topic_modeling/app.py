# -*- coding: utf-8 -*-
import shutil

import streamlit as st
from pages.onboarding import page as onboarding
from pages.timestamped_corpus import page as timestamped_corpus

# Custom imports
from kiara_modules.playground.markus.streamlit import MultiPageApp

app = MultiPageApp(st)

# Title of the main page
st.title("Kiara experiment: topic modeling multi page")

# Add all your application here
app.add_page("Select data", onboarding)
app.add_page("Timestamped corpus", timestamped_corpus)
# app.add_page("Timestamped data", timestamped_corpus.app)


# The main app
app.run()

sel = "-- n/a --"
if hasattr(st.session_state, "selected_corpus_name"):
    sel = st.session_state.selected_corpus_name
st.sidebar.markdown(f"Selected corpus: **{sel}**")

st.sidebar.markdown(
    "The button below is only for development, it clears the kiara data store."
)
clear_data_store = st.sidebar.button("Clear data store")
if clear_data_store:
    path = app.kiara.data_store._base_path
    print()
    print(f"Deleting folder: {path}...")
    shutil.rmtree(path=path, ignore_errors=True)
    print("Folder deleted.")

number_items = len(app.kiara.data_store.aliases)
st.sidebar.markdown(f"Number items in data store: {number_items}")

aliases = ""
for a in app.kiara.data_store.aliases:
    md = app.kiara.data_store.get_metadata_for_id(a)
    aliases = aliases + f" - **{a}** *({md.value_type})*\n"

if aliases:
    st.sidebar.write("### Items:")
    st.sidebar.markdown(aliases)
