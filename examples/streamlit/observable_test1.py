# -*- coding: utf-8 -*-
import streamlit as st

# data = observers.get("corpus_agg")
# df = pd.DataFrame(data)
# st.dataframe(df)
#
#
# uploaded_files = st.file_uploader("Add files", type = 'txt', accept_multiple_files=True)
# print(uploaded_files)
from kiara import Kiara

# st.markdown('<style>@import url("https://fonts.googleapis.com/css2?family=Palanquin:wght@400;700&display=swap")</style>', unsafe_allow_html=True)
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))


# timeSelected = st.sidebar.radio(
#     label="Select time span", index=3, options=["day", "week", "month", "year"]
# )
#
#
# observers = observable(
#     "",
#     notebook="d/50b89c7d50524163",
#     targets=["viewof chart", "style"],
#     # observe=["corpus_agg"],
#     redefine={"timeSelected": timeSelected},
# )


path = st.text_input("Path")
button = st.button("Import")

if button:
    kiara = Kiara()
    module = kiara.create_module("table.import.from_local_folder")
    result = module.run(path=path)
    table_obj = result.get_value_obj("table")
    aliases = ["my_first_table"]
    saved_metadata = table_obj.save(aliases=aliases)
    st.write(saved_metadata.dict())

    # table = result.get_value_data("table")
    # st.write(table.to_pandas())


# if not st.session_state.get("counter"):
#     st.session_state.counter = 0
#
# something_else = st.checkbox("Just an example")
#
# path = st.text_input("Path")
# if path:
#     st.session_state.counter = st.session_state.counter + 1
#     st.write(f"IMPORTING: {st.session_state.counter}")
