import streamlit as st
from kiara import Kiara
from streamlit_observable import observable
import json 

def app():

    kiara: Kiara = st.session_state["kiara"]
    table_value = st.session_state.data

    sql_query_null = "SELECT * FROM data WHERE latitude isnull or longitude isnull"
    sql_query_not_null = "SELECT * FROM data WHERE latitude IS NOT NULL or longitude IS NOT NULL"

    query_module = kiara.get_operation("table.query.sql")
    query_result = query_module.module.run(table=table_value, query=sql_query_not_null)
    #query_result = query_module.module.run(table=table_value, query=sql_query_null)
    query_result_value = query_result.get_value_obj("query_result")
    query_result_table = query_result_value.get_value_data()

    data = list(query_result_table.to_pandas().to_dict(orient="index").values())
    
    data_json = json.dumps(data, default=str)
    
    cleaned_data = json.loads(data_json)
    


    observers = observable(
            "Test",
            notebook="@mariellacc/geolocation",
            targets=["viewof map"],
            redefine={
                "data": cleaned_data,
            },
            observe=["unmapItems"],
        )

    unmapItems = observers.get("unmapItems")

    #print(unmapItems)

    if unmapItems is None:
        st.write("All observations mappable")
    
    if unmapItems is not None:
        #print(unmapItems)
        st.write('Unmappable observations')
        st.dataframe(unmapItems)





