# -*- coding: utf-8 -*-
import os

import pyarrow as pa
import streamlit as st
from kiara import Kiara
from kiara.data import Value


def augment_table(kiara: Kiara, table_value: Value):

    pipeline_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "newspaper_corpora",
        "augment_newspaper_corpora_table.json",
    )

    workflow = kiara.create_workflow(pipeline_file)
    workflow.inputs.set_value("value_id", table_value.id)
    return workflow.outputs.get_value_obj("table")


def run_sql(kiara: Kiara, table_value: Value, unit: str):

    pipeline_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "newspaper_corpora",
        f"query_newspaper_corpora_{unit}.json",
    )
    workflow = kiara.create_workflow(pipeline_file)
    workflow.inputs.set_value("value_id", table_value.id)

    return workflow.outputs.get_value_obj("query_result")


def page(kiara: Kiara):

    if hasattr(st.session_state, "selected_corpus_name"):
        selected = st.session_state.selected_corpus_name
    else:
        selected = None

    table_value: Value = kiara.data_store.load_value(selected)

    column_names = table_value.get_metadata("table")["table"]["column_names"]

    if "date" in column_names:
        # this means the table was already processed
        table = table_value
    else:
        augmented_table = augment_table(kiara=kiara, table_value=table_value)
        new_alias = f"{selected}_augmented"
        saved_md = augmented_table.save(aliases=[new_alias])
        saved_id = saved_md.value_id
        table = kiara.data_store.load_value(saved_id)

    unit = st.selectbox("Select unit", ["day", "month", "year"])

    query_result = run_sql(kiara=kiara, table_value=table, unit=unit)

    table: pa.Table = query_result.get_value_data()
    st.write(table.to_pandas())
    # st.dataframe(augmented_table.get_value_data())
    # st.dataframe(augmented_table.get_value_data())

    # table = table_value.get_value_data()
    # st.dataframe(table.to_pandas())

    # data = list(df.to_dict(orient="index").values())
    # data_json = json.dumps(data, default=str)
    # cleaned_data = json.loads(data_json)
    #
    # timeSelected = st.sidebar.radio(
    #     label="Select time span", index=3, options=["day", "week", "month", "year"]
    # )
    #
    # observers = observable(
    #     "Test",
    #     notebook="d/50b89c7d50524163",
    #     targets=["viewof chart", "style"],
    #     observe=["corpus_agg"],
    #     redefine={"timeSelected": timeSelected, "corpus": cleaned_data},
    # )
    #
    # data = observers.get("corpus_agg")
    # df = pd.DataFrame(data)
    # st.dataframe(df)
