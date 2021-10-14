import streamlit as st
from kiara import Kiara
from streamlit_observable import observable
import json 

def app():

    kiara: Kiara = st.session_state["kiara"]
    table_value = st.session_state.data

    sql_query_null = "SELECT * FROM data WHERE latitude isnull or longitude isnull"
    
    sql_query_not_null = "SELECT * FROM data WHERE latitude IS NOT NULL or longitude IS NOT NULL"

    sql_query_unique_points = "SELECT latitude, longitude, COUNT(*) as count FROM data GROUP BY latitude,longitude"

    query_module = kiara.get_operation("table.query.sql")
    
    # get json data for not null items to pass to the viz and pandas df to display table
    query_result = query_module.module.run(table=table_value, query=sql_query_not_null)
    query_result_value = query_result.get_value_obj("query_result")
    query_result_table = query_result_value.get_value_data()
    query_result_df = query_result_table.to_pandas()
    data = list(query_result_table.to_pandas().to_dict(orient="index").values())
    data_json = json.dumps(data, default=str)
    cleaned_data = json.loads(data_json)
    
    # get observations that contain nan values in lat/long to display dataframe
    query_result_null = query_module.module.run(table=table_value, query=sql_query_null)
    query_result_null_value = query_result_null.get_value_obj("query_result")
    query_result_null_table = query_result_null_value.get_value_data()
    query_result_null_df = query_result_null_table.to_pandas()

    # create table with unique lats & longs and count, to only draw one point per
    # location on map (and avoid perf issues as nr of points impacts perf)
    # Quantity can then be visualised either by darker color when many points at same
    # place or by scaling points size. (query is done on not null items)
    query_result_unique = query_module.module.run(table=query_result_table, query=sql_query_unique_points)
    query_result_unique_value = query_result_unique.get_value_obj("query_result")
    query_result_unique_table = query_result_unique_value.get_value_data()
    data_unique = list(query_result_unique_table.to_pandas().to_dict(orient="index").values())
    data_unique_json_prep = json.dumps(data_unique, default=str)
    data_unique_json = json.loads(data_unique_json_prep)

    
    map_json = st.session_state.map
    my_expander = st.sidebar.expander(label='Settings')

    #table_columns = table_value.get_value_data().to_pandas().columns
    table_columns = table_value.get_value_data().column_names
    table_columns.insert(0, 'None')

    with my_expander:
        unit = st.selectbox("Projection", ('Mercator', 'Equal Earth', 'Geo Orthographic'))
        scale = st.selectbox("Scale overlapping points", ('Color', 'Size'))
        
        categorize = st.selectbox("Select column to categorize", table_columns)
        
        if categorize != 'None':
            st.write('work in progress')
            #sql_query_col_unique = f"SELECT COUNT(DISTINCT {categorize}) AS count FROM data"
            #sql_query_col_unique_result = query_module.module.run(table=table_value, query=sql_query_col_unique)
            #sql_query_col_unique_result_value = sql_query_col_unique_result.get_value_obj("query_result")
        
        zoom_options = {"zoom_on": "Zoom on", "zoom_fr": "Zoom freeze", "zoom_off": "Zoom off/reset"}

        def format_func(option):
            return zoom_options[option]

        zoom = st.selectbox("Zoom settings", options=list(zoom_options.keys()), format_func=format_func)

    
    # query to check if selected column contains less than 11 values (maximum in colorbrewer color scales)
    # as the categories will be represented as colors, it wouldn't make sense visually to allow for more
    #sql_query_col_unique = "SELECT COUNT(DISTINCT column_name) AS some_alias FROM table_name"


    map_points = observable(
            "geolocation map",
            notebook="@mariellacc/geolocation",
            targets=["container", "svgLayer", "canvasLayer"],
            redefine={ 
                "userData": cleaned_data,
                "mapData": map_json,
                "chooseScale": scale,
                "chooseProjection": unit,
                "uniqueLatLong": data_unique_json,
                "zoomOptions": zoom
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