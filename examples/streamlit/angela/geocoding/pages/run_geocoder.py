# -*- coding: utf-8 -*-
import streamlit as st
from kiara import Kiara


def app():

    kiara: Kiara = st.session_state["kiara"]
    table_value = st.session_state.data
    
    
    geocoder = st.selectbox("Select geocoder", ('Nominatim', 'GeoNames (not yet implemented'))
    table_columns = table_value.get_value_data().column_names
    
#up to here - questions: how to associate column names with scales of data for geocoder?
   

    button = st.button("Run")

    if button:

        file_import_operation = kiara.get_operation(
            "import.file.from.file_path"
        )

        import_result = file_import_operation.module.run(file_path=path)
        imported_file = import_result.get_value_obj("file")

        alias = "my_first_table" 
        imported_file.save(
            aliases=[f"{alias}__source_files"]
        )  

        table_convert_module = kiara.create_module("create.table.from.csv_file", module_config={"source_type": "file", "target_type": "table", "ignore_errors": True})
        convert_result = table_convert_module.run(csv_file=imported_file)
        imported_table = convert_result.get_value_obj("table")

        imported_table.save(aliases=[alias])

        table = imported_table.get_value_data()
        df = table.to_pandas()
        st.write(df.head())


        st.session_state.data = imported_table

        st.write("Success! Select the next step from the top left nav menu.")
