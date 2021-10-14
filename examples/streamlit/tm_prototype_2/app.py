# -*- coding: utf-8 -*-
import json
import os

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit.delta_generator import DeltaGenerator
from streamlit_observable import observable

import kiara_streamlit
from kiara_streamlit.pipelines import PipelineApp
from kiara_streamlit.pipelines.pages import PipelinePage

st.set_page_config(page_title="Kiara-streamlit auto-rendered pipeline", layout="wide")

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})


# =======================================================================================
# create one class per page, the only method that needs to be implemented is 'run_page'

class OnboardingPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        st.markdown(
            "Download the corpus on your computer, unzip and copy local folder path"
        )
        st.markdown(
            "https://zenodo.org/record/4596345/files/ChroniclItaly_3.0_original.zip?download"
        )
        st.markdown("Paste local folder path into input below")
        st.markdown(
            "Wait for the success message, and then select next page in top left nav menu"
        )

        current_onboard_data_input = self.get_pipeline_inputs_for_step("onboard_data")
        onboard_user_input = st.kiara.valueset_input(
            current_onboard_data_input, key=self.get_page_key("path_to_corpus"), defaults=current_onboard_data_input
        )
        # set the inputs we got from the user
        self.set_pipeline_inputs(inputs=onboard_user_input, render_errors=True)

        button = st.button("Onboard")
        if button:
            # process the onboarding step
            print("PROCESSING STEP: 'onboard_data'")
            with st.spinner("Importing text files..."):
                onboarding_result = self.process_step(
                    "onboard_data", render_result=False)

            if onboarding_result != "Success":
                st.error(onboarding_result)
                return

            # now we need to execute the convert to table step
            print("PROCESSING STEP: 'convert_table'")
            with st.spinner("Creating corpus table..."):
                convert_table_result = self.process_step("convert_table")

            if convert_table_result != "Success":
                st.error(onboarding_result)
                return

        show_table = st.checkbox("Show imported corpus", value=False, key=self.get_page_key("show_corpus_checkbox"))

        table = self.get_step_outputs("convert_table").get_value_obj("value_item")
        if table.item_is_valid():
            if show_table:
                st.dataframe(table.get_value_data().to_pandas())

            st.write("Success! Select the next step from the top left nav menu.")
        else:
            if show_table:
                st.write("Corpus not ready yet.")


class AugmentCorpusMetadataPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

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
                # if that is not selected, subsequent steps don't make sense
                pass
            elif process_metadata == "yes":
                print("PROCESSING STEP: 'augment_corpus_data'")
                with st.spinner("Extracting metadata from file names..."):
                    augment_corpus_result = self.process_step("augment_corpus_data")

                if augment_corpus_result != "Success":
                    st.error(augment_corpus_result)
                    return

            table = self.get_step_outputs("augment_corpus_data").get_value_obj("table")
            if table.item_is_valid():
                st.write("Result preview")
                AgGrid(table.get_value_data().to_pandas().head(50))


class TimestampedCorpusPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        # this is basically unchanged from the other prototype, since it doesn't involve any processing on the actual workflow

        augmented_table_value = self.get_step_outputs("augment_corpus_data").get_value_obj("table")

        if not augmented_table_value.item_is_valid():

            st.write("Augmented table not created yet, please do so before continuing.")
            return

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

        query_module = st.kiara.get_operation("table.query.sql")
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


class TokenizationPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        st.write(
            "For languages based on latin characters, use default tokenization option (by word)."
        )
        st.write(
            "This first pre-processing step is necessary to proceed further. Depending on your corpus size, it could take several minutes"
        )
        tokenize = st.selectbox("Tokenize by", ("word", "character"), key="0")
        token_button = st.button("Proceed")

        if token_button:

            self.set_pipeline_inputs(inputs={"tokenize_by_word": tokenize == "word"})
            print("PROCESSING STEP: 'tokenization'")
            with st.spinner('Tokenizing corpus, this might take a while...'):
                tokenize_result = self.process_step("tokenization")

            if tokenize_result != "Success":
                st.error(tokenize_result)
                return

        tokenized_table_value = self.get_step_outputs("tokenization").get_value_obj("tokens_array")

        if tokenized_table_value.item_is_valid():
            # if the output exists, we write it as a pandas Series (since streamlit supports that natively)
            df = tokenized_table_value.get_value_data().to_pandas()
            st.write("Result preview")
            st.dataframe(df.head(50))
        else:
            st.write("No result")


class TextPreprocessingPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

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

            step_inputs = {
                "apply_lowercase": lowercase == "yes",
                "preprocess_methodology": preprocess,
                "min_token_length": shorttokens
            }
            with st.spinner("Pre-processing texts..."):
                self.set_pipeline_inputs(inputs=step_inputs)

                print("PROCESSING STEP: 'text_pre_processing'")
                preprocess_result = self.process_step("text_pre_processing")

            if preprocess_result != "Success":
                st.error(preprocess_result)

        # retrieve the actual table value
        preprocessed_table_value = self.get_step_outputs("text_pre_processing").get_value_obj(
            "preprocessed_array"
        )

        if preprocessed_table_value.item_is_valid():
            # if the output exists, we write it as a pandas Series (since streamlit supports that natively)
            df = preprocessed_table_value.get_value_data().to_pandas()
            st.write("Result preview")
            st.dataframe(df.head(50))
        else:
            st.write("No result")

# ===============================================================================================================
# main app

main_pipeline = os.path.join(pipelines_folder, "tm_pipeline.yaml")

app = PipelineApp.create(
    pipeline=main_pipeline
)

if not app.pages:
    app.add_page(OnboardingPage(id="Onboarding"))
    app.add_page(AugmentCorpusMetadataPage(id="Get file metadata"))
    app.add_page(TimestampedCorpusPage(id="Timestamped data"))
    app.add_page(TokenizationPage(id="Tokenization"))
    app.add_page(TextPreprocessingPage(id="Text pre-processing"))

app.run()
