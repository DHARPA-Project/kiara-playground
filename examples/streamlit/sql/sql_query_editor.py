# -*- coding: utf-8 -*-
import streamlit as st

import kiara_streamlit

st.set_page_config(page_title="Kiara experiment: sql", layout="wide")

kiara_streamlit.init()

st.title("Perform sql query against an arrow table")

desc = """You'll need to have some tables loaded into your data store in order for this to be useful."""
st.markdown(desc)

st.kiara.sql_query(
    show_preview_table_option=True,
    use_sidebar=True,
    show_table_metadata_option=True,
    show_save_option=True,
    key="sql_query",
    show_sampling_option=True,
)
