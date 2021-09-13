# -*- coding: utf-8 -*-
import streamlit as st

import kiara_streamlit
from kiara_streamlit import KiaraContext

kiara_streamlit.init()

st.set_page_config(page_title="Kiara experiment: sql", layout="wide")

ktx: KiaraContext = st.get_ktx()

st.title("Perform sql query against an arrow table")

desc = """You'll need to have some tables loaded into your data store in order for this to be useful."""
st.markdown(desc)

ktx.sql_query(
    show_table_option=True,
    use_sidebar=True,
    show_metadata_option=True,
    show_save_option=True,
    key="sql_query",
    show_sampling_option=True,
)
