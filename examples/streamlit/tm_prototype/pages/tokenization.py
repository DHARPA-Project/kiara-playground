# -*- coding: utf-8 -*-
import streamlit as st
from kiara import Kiara


def app():
    st.markdown("## Tokenization")

    kiara: Kiara = st.session_state["kiara"]

    data = (
        st.session_state.data.get_value_obj()
        if not st.session_state.augmented_data
        else st.session_state.augmented_data
    )

    st.write(
        "For languages based on latin characters, use default tokenization option (by word)."
    )
    st.write(
        "This first pre-processing step is necessary to proceed further. Depending on your corpus size, it could take several minutes"
    )
    tokenize = st.selectbox("Tokenize by", ("word", "character"), key="0")
    token_button = st.button("Proceed")

    if token_button:
        # CHANGED
        # again, we don't really need a workflow here, just use the module directly
        # tokenize_workflow = kiara.create_workflow(
        #     "playground.mariella.language.tokenize"
        # )
        # tokenize_workflow.inputs.set_values(table=data, column_name="content")
        # tokenized_table_value = tokenize_workflow.outputs.get_value_obj("tokens_array")

        tokenize_op = kiara.get_operation("playground.mariella.language.tokenize")
        tokenize_result = tokenize_op.module.run(
            table=data, column_name="content", tokenize_by_word=tokenize == "word"
        )  # not sure if you actually want the 'tokenize' value here, it wasn't in the code above
        tokenized_table_value = tokenize_result.get_value_obj("tokens_array")

        st.session_state.tokenized_data = tokenized_table_value

        table = tokenized_table_value.get_value_data()

        if table:
            # if the output exists, we write it as a pandas Series (since streamlit supports that natively)
            df = table.to_pandas()
            st.write("Result preview")
            st.dataframe(df.head(50))
        else:
            st.write("No result")
