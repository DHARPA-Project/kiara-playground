# -*- coding: utf-8 -*-
import os

import streamlit as st
from kiara import Kiara
from st_aggrid import AgGrid


def app():

    kiara: Kiara = st.session_state["kiara"]

    table_value = st.session_state.data

    st.markdown(
        "Wait for file preview to be displayed, before proceeding to the next step"
    )
    st.markdown("*Temporary screen for file names metadata step*")
    st.markdown("*This module will be completed at a later stage *")

    process_metadata = st.radio("Do your file names contain metadata?", ("no", "yes"))

    st.write("Supported pattern: '/sn86069873/1900-01-05/'")
    st.write("LCCN title information and publication date (yyyy-mm-dd)")

    if process_metadata:
        if process_metadata == "no":
            st.session_state.metadata = False
            st.session_state.augmented_data = False

        elif process_metadata == "yes":

            # CHANGED
            # let's register the pipeline into kiara, because then it becomes a module and we don't even need to create a workflow but can use the module 'run' method directly (like in the onboarding step)
            # usually, we'd probably register all pipelines we intend to use at the start page, but I wanted to keep this on this page so it's easier to see what's happening
            if "augment_corpus_table" not in kiara.available_pipeline_module_types:
                # we don't want to register it twice, but that would happen if we visited this streamlit page twice, because streamlit would execute that code again
                augment_pipeline = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "newspaper_corpora",
                    "augment_newspaper_corpora_table.json",
                )

                kiara.register_pipeline_description(
                    augment_pipeline, "augment_corpus_table"
                )

            # CHANGED
            # load the pipeline file and create a workflow
            # workflow = kiara.create_workflow(augment_pipeline)
            # workflow.inputs.set_value("value_id", table_value.id)
            # workflow.inputs.set_value("table", table_value)
            # retrieve the actual table value
            # augmented_table_value = workflow.outputs.get_value_obj("table")

            # I modified the 'augment_newspaper_corpora_table.json' pipeline a bit, so the inputs are different to before now, and take the actual table value, and not its id
            augment_module = kiara.create_module("augment_corpus_table")
            augment_result = augment_module.run(table=table_value)
            augmented_table_value = augment_result.get_value_obj("table")

            st.session_state.augmented_data = augmented_table_value
            st.session_state.metadata = True

            table = augmented_table_value.get_value_data()
            df = table.to_pandas()
            st.write("Result preview")
            AgGrid(df.head(50))
