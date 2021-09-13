# -*- coding: utf-8 -*-
import json

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_observable import observable


def app():
    st.markdown("## Timestamped Corpus")

    if st.session_state.metadata:

        kiara = st.session_state["kiara"]

        augmented_table_value = st.session_state.augmented_data

        sql_query_day = "SELECT strptime(concat(day, '/', month, '/', year), '%d/%m/%Y') as date, pub_name, count FROM (SELECT YEAR(date) as year, MONTH(date) as month, DAY(date) as day, pub_name, count(*) as count FROM data group by YEAR(date), MONTH(date), DAY(date), pub_name ORDER BY year, month, day, pub_name) as agg"
        sql_query_month = "SELECT strptime(concat('01/', month, '/', year), '%d/%m/%Y') as date, pub_name, count FROM (SELECT YEAR(date) as year, MONTH(date) as month, pub_name, count(*) as count FROM data group by YEAR(date), MONTH(date), pub_name ORDER BY year, month, pub_name) AS agg"
        sql_query_year = "SELECT strptime(concat('01/01/', year), '%d/%m/%Y') as date, pub_name, count FROM (SELECT YEAR(date) as year, pub_name, count(*) as count FROM data group by YEAR(date), pub_name ORDER BY year, pub_name) AS agg"

        my_expander = st.sidebar.expander(label="Settings")

        with my_expander:
            unit = st.selectbox("Aggregate by", ("year", "month", "day"))

            scaleType = st.selectbox("Scale by", ("color", "height"))

            axisLabel = st.selectbox("Axis", ("5-year", "year", "month", "day"))

        if unit == "day":
            query = sql_query_day
        elif unit == "month":
            query = sql_query_month
        else:
            query = sql_query_year

        # CHANGED
        # again, it's not really necessary anymore to create a workflow for just executing a single module
        # query_workflow = kiara.create_workflow("table.query.sql")
        # query_workflow.inputs.set_values(table=augmented_table_value, query=query)
        #
        # query_result_value = query_workflow.outputs.get_value_obj("query_result")
        # query_result_table = query_result_value.get_value_data()

        query_module = kiara.get_operation("table.query.sql")
        query_result = query_module.module.run(table=augmented_table_value, query=query)
        query_result_value = query_result.get_value_obj("query_result")
        query_result_table = query_result_value.get_value_data()

        data = list(query_result_table.to_pandas().to_dict(orient="index").values())
        data_json = json.dumps(data, default=str)
        cleaned_data = json.loads(data_json)

        observers = observable(
            "Test",
            notebook="d/d1e17c291019759e",
            targets=["viewof chart", "style"],
            redefine={
                "timeSelected": unit,
                "data": cleaned_data,
                "scaleType": scaleType,
                "axisLabel": axisLabel,
            },
            observe=["dateInfo"],
        )

        timeInfo = observers.get("dateInfo")

        col1, col2 = st.columns(2)

        if "preview_choice" not in st.session_state:
            st.session_state.preview_choice = "data"

        with col1:
            data_preview = st.button(label="Aggregated data")

        with col2:
            source_view = st.button(label="Sources list by time period")

        if data_preview:
            st.session_state.preview_choice = "data"

        if source_view:
            st.session_state.preview_choice = "source"

        display_choice = st.session_state.preview_choice

        if display_choice == "data":

            AgGrid(query_result_table.to_pandas())

        else:

            if timeInfo is None:
                st.markdown("Hover over chart and click on date that appears on top")

            if timeInfo is not None:

                sql_query_day2 = f"SELECT pub_name, date, content FROM data WHERE DATE_PART('year', date) = {timeInfo[0]} AND DATE_PART('month', date) = {timeInfo[1]} and DATE_PART('day', date) = {timeInfo[2]}"
                sql_query_month2 = f"SELECT pub_name, date, content FROM data WHERE DATE_PART('year', date) = {timeInfo[0]} AND DATE_PART('month', date) = {timeInfo[1]}"
                sql_query_year2 = f"SELECT pub_name, date, content FROM data WHERE DATE_PART('year', date) = {timeInfo[0]}"

                if unit == "day":
                    query2 = sql_query_day2
                elif unit == "month":
                    query2 = sql_query_month2
                else:
                    query2 = sql_query_year2

                # CHANGED
                # same as above, replacing workflow with operation/module
                # query_workflow2 = kiara.create_workflow("table.query.sql")
                # query_workflow2.inputs.set_values(
                #     table=augmented_table_value, query=query2
                # )
                # query_result_value2 = query_workflow2.outputs.get_value_obj(
                #     "query_result"
                # )
                # query_result_table2 = query_result_value2.get_value_data()

                # we can re-use the 'query_module' object from above
                query_result2 = query_module.module.run(
                    table=augmented_table_value, query=query2
                )
                query_result_value2 = query_result2.get_value_obj("query_result")
                query_result_table2 = query_result_value2.get_value_data()

                df2 = query_result_table2.to_pandas()
                gb = GridOptionsBuilder.from_dataframe(df2)

                gb.configure_column("content", maxWidth=800, tooltipField="content")

                AgGrid(df2.head(100), gridOptions=gb.build())

    else:
        st.write(
            "This optional step is only activated if your corpus files contain dates. Please go to the next step."
        )
