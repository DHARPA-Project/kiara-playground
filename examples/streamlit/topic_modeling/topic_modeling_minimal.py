# -*- coding: utf-8 -*-


import os

import streamlit as st

from kiara import Kiara
from kiara_modules.playground.markus.streamlit import (
    onboard_file_bundle,
)

st.title("Kiara/streamlit experiment - Topic modelling minimal")

kiara = Kiara.instance()

corpus_alias = "markus_text_corpus"

with st.form(key="onboard_corpus"):

    st.markdown("## Onboard text corpus")
    st.markdown(
        "Select sample data from one publication folder located in  kiara-playground/examples/newspaper_corpora/CI_newspaper_subcorpora"
    )
    st.markdown("Once you're ready, go to the next step via select box on the left")

    uploaded_files = st.file_uploader(
        "Add files", type="txt", accept_multiple_files=True
    )
    onboard_button = st.form_submit_button(label="Onboard")


if onboard_button and uploaded_files:
    onboard_file_bundle(
        kiara=kiara, uploaded_files=uploaded_files, aliases=[corpus_alias]
    )
    st.markdown(f"Onboarded corpus: {corpus_alias}")
else:

    st.markdown("Nothing onboarded (yet).")
    st.stop()

table_value = kiara.data_store.load_value(corpus_alias)

# load the pipeline file and create a workflow
augment_pipeline = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "newspaper_corpora",
    "augment_newspaper_corpora_table.json",
)
workflow = kiara.create_workflow(augment_pipeline)
# set our table as input to the workflow
workflow.inputs.set_value("value_id", table_value.id)

# retrieve the actual table value
augmented_table_value = workflow.outputs.get_value_obj("table")

# if you wanted to print the table, you could do:
# table = augmented_table_value.get_value_data()
# st.dataframe(table.to_pandas())


sql_query_day = "SELECT YEAR(date) as year, MONTH(date) as month, DAY(date) as day, pub_name, count(*) as count FROM data group by YEAR(date), MONTH(date), DAY(date), pub_name order by year, month, day, pub_name"
sql_query_month = "SELECT YEAR(date) as year, MONTH(date) as month, pub_name, count(*) as count FROM data group by YEAR(date), MONTH(date), pub_name ORDER BY year, month, pub_name"
sql_query_year = "SELECT YEAR(date) as year, pub_name, count(*) as count FROM data group by YEAR(date), pub_name ORDER BY year, pub_name"

unit = "year"

if unit == "day":
    query = sql_query_day
elif unit == "month":
    query = sql_query_month
else:
    query = sql_query_year

query_workflow = kiara.create_workflow("table.query.sql")
query_workflow.inputs.set_values(table=augmented_table_value, query=query)

query_result_value = query_workflow.outputs.get_value_obj("query_result")
query_result_table = query_result_value.get_value_data()

st.dataframe(query_result_table.to_pandas())
