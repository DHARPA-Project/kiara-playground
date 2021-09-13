# -*- coding: utf-8 -*-
import streamlit as st


def app():
    st.markdown("## Text pre-processing")

    kiara = st.session_state["kiara"]

    st.write("#### 1. Lowercase")
    lowercase = st.selectbox(" ", ("no", "yes"), key="1")
    # isalnum,isalph,isdigit
    st.write("#### 2. Numbers and punctuation")
    display_preprocess = [
        "None",
        "Remove all tokens that include numbers (e.g. ex1ample)",
        "Remove all tokens that include punctuation and numbers (e.g. ex1a.mple)",
        "Remove all tokens that contain numbers only (e.g. 876)",
    ]
    preprocess = st.radio(
        " ",
        options=range(len(display_preprocess)),
        format_func=lambda x: display_preprocess[x],
    )
    st.write("#### 3. Words length")
    display_shorttokens = ["None", "1", "2", "3", "4", "5"]
    shorttokens = st.selectbox(
        "Remove words shorter than ... characters",
        options=range(len(display_shorttokens)),
        format_func=lambda x: display_shorttokens[x],
    )

    confirmation = st.button("Proceed")

    if confirmation:
        preprocess_workflow = kiara.create_workflow(
            "playground.mariella.text_preprocessing.preprocess"
        )
        preprocess_workflow.inputs.set_values(
            array=st.session_state.tokenized_data,
            lowercase=lowercase == "yes",
            preprocess_method=preprocess,
            remove_short_tokens=shorttokens,
        )

        # retrieve the actual table value
        preprocessed_table_value = preprocess_workflow.outputs.get_value_obj(
            "preprocessed_array"
        )
        st.session_state.preprocessed_text = preprocessed_table_value

        table = preprocessed_table_value.get_value_data()

        if table:
            # if the output exists, we write it as a pandas Series (since streamlit supports that natively)
            df = table.to_pandas()
            st.write("Result preview")
            st.dataframe(df.head(50))
        else:
            st.write("No result")
