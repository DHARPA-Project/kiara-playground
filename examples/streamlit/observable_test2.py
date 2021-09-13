# -*- coding: utf-8 -*-
import json

import pandas as pd
import pyarrow as pa
import streamlit as st
from kiara import Kiara

# before running this, the dataset must be imported into the local kiara data store, with the id_or_alias 'df_distrib':
#
# kiara run table.import.from_local_file path=/home/markus/projects/dharpa/kiara-playground/data/mariella/v2/df_distrib-3.csv ids_or_aliases=df_distrib
from kiara.data import Value
from pandas import DataFrame
from streamlit_observable import observable

# st.markdown('<style>@import url("https://fonts.googleapis.com/css2?family=Palanquin:wght@400;700&display=swap")</style>', unsafe_allow_html=True)
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))


kiara = Kiara.instance()

table_value: Value = kiara.data_store.load_value("df_distrib")

table: pa.Table = table_value.get_value_data()
df: DataFrame = table.to_pandas()

data = list(df.to_dict(orient="index").values())
data_json = json.dumps(data, default=str)
cleaned_data = json.loads(data_json)

timeSelected = st.sidebar.radio(
    label="Select time span", index=3, options=["day", "week", "month", "year"]
)

observers = observable(
    "Test",
    notebook="d/50b89c7d50524163",
    targets=["viewof chart", "style"],
    observe=["corpus_agg"],
    redefine={"timeSelected": timeSelected, "corpus": cleaned_data},
)

data = observers.get("corpus_agg")
df = pd.DataFrame(data)
st.dataframe(df)
