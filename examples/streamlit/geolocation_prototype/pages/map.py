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
    query_result_value = query_result.get_value_obj("query_result")
    query_result_table = query_result_value.get_value_data()
    query_result_df = query_result_table.to_pandas()
    data = list(query_result_table.to_pandas().to_dict(orient="index").values())
    data_json = json.dumps(data, default=str)
    cleaned_data = json.loads(data_json)
    
    query_result_null = query_module.module.run(table=table_value, query=sql_query_null)
    query_result_null_value = query_result_null.get_value_obj("query_result")
    query_result_null_table = query_result_null_value.get_value_data()
    query_result_null_df = query_result_null_table.to_pandas()

    
    map_json = st.session_state.map
    my_expander = st.sidebar.expander(label='Settings')

    with my_expander:
        unit = st.selectbox("Projection", ('Mercator', 'Equal Earth', 'Geo Orthographic'))

    map_points = observable(
            "Test",
            notebook="@mariellacc/geolocation",
            targets=["container", "svgLayer", "canvasLayer"],
            redefine={ 
                "userData": cleaned_data,
                "mapData": map_json,
                "chooseProjection": unit,
            },
            observe=["unmapItems"],
        )

    unmapItems = map_points.get("unmapItems")

    col1, col2, col3 = st.columns(3)

    with col1:
        mapped_items = st.button(label='Mappable observations')
        
    with col2:
        missing_coords = st.button(label='Missing coordinates')
    
    with col3:
        unmap_items = st.button(label='Unmappable observations')

    
    if mapped_items:
        st.dataframe(query_result_df)
    
    if missing_coords:
        st.dataframe(query_result_null_df)
    
    if unmap_items:
        if unmapItems is not None and len(unmapItems) >0:
            st.dataframe(unmapItems)
        elif len(unmapItems) == 0:
            st.write("No unmappable items") 