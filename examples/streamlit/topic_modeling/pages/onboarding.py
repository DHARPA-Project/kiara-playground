# -*- coding: utf-8 -*-
import pyarrow as pa
import streamlit as st
from kiara import Kiara

from kiara_modules.playground.markus.streamlit import onboard_file_bundle


def check_alias_valid(kiara: Kiara, alias: str):

    if not alias:
        return False
    all_aliases = kiara.data_store.aliases
    if alias in all_aliases:
        return False
    else:
        return True


def page(kiara: Kiara):

    with st.form(key="onboard_corpus"):

        st.markdown("## Onboard text corpus")
        st.markdown(
            "Select sample data from one publication folder located in  kiara-playground/examples/newspaper_corpora/CI_newspaper_subcorpora"
        )
        st.markdown("Once you're ready, go to the next step via select box on the left")

        alias = st.text_input("Corpus name")
        uploaded_files = st.file_uploader(
            "Add files", type="txt", accept_multiple_files=True
        )
        alias_valid = check_alias_valid(kiara, alias)
        onboard_button = st.form_submit_button(label="Onboard")

    if onboard_button and alias_valid and uploaded_files:
        onboard_file_bundle(kiara=kiara, uploaded_files=uploaded_files, aliases=[alias])
        st.markdown(f"Onboarded corpus: {alias}")
    else:
        if onboard_button:
            if not alias_valid:
                st.markdown(
                    "Corpus name not valid or already exists. Not onboarding any files..."
                )
            else:
                st.markdown("Corpus name valid, but no files selected.")

    available_tables = []
    for item in kiara.data_store.aliases:
        md = kiara.data_store.get_metadata_for_id(item)
        if md.value_type == "table":
            column_names = md.metadata["table"]["metadata_item"]["column_names"]
            if "content" in column_names and "file_name" in column_names:
                available_tables.append(item)

    st.markdown("## Select text corpus")
    st.markdown("Select the corpus you want to work.")
    selected = st.selectbox("Corpus alias", available_tables)

    if selected:
        table_value = kiara.data_store.load_value(selected)
        table: pa.Table = table_value.get_value_data()
        st.dataframe(table.to_pandas())

        st.session_state.selected_corpus_name = selected
