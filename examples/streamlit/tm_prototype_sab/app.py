# -*- coding: utf-8 -*-
import json
import os

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit.delta_generator import DeltaGenerator
from streamlit_observable import observable

import kiara_streamlit
from kiara import Kiara
from kiara.data import Value
from kiara.data.onboarding.batch import BatchOnboard
from kiara_modules.language_processing.tokens import get_stopwords
from kiara_streamlit.pipelines import PipelineApp
from kiara_streamlit.pipelines.pages import PipelinePage

st.set_page_config(page_title="Kiara-streamlit auto-rendered pipeline", layout="wide")

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})


# =======================================================================================
# create one class per page, the only method that needs to be implemented is 'run_page'

class AugmentCorpusMetadataPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        selected_table: Value = st.kiara.value_input_table(label="Select table", add_no_value_option=True, onboard_options={"enabled": True, "source_default": "folder"}, key=self.get_page_key("selected_table"))

        preview_table = st.checkbox("Preview table (first 50 rows)")
        if selected_table and selected_table.item_is_valid():
            if preview_table:
                st.dataframe(selected_table.get_value_data().to_pandas().head(50))
            self.set_pipeline_inputs(inputs={"corpus_table": selected_table}, render_errors=True)
        else:
            st.markdown("No (valid) table selected, not doing anything...")
            return

        # TODO: here we could check whether we have the required columns ('file_name', specifically)

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
                st.write("### Result preview (first 50 rows)")
                AgGrid(table.get_value_data().to_pandas().head(50))

class TimestampedCorpusPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        # this is basically unchanged from the other prototype, since it doesn't involve any processing on the actual workflow

        augmented_table_value = self.get_step_outputs("augment_corpus_data").get_value_obj("table")


        #st.write(self.pipeline.get_current_state().dict())

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

            st.table(query_result_table.to_pandas())

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

                st.dataframe(df2.head(100))

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
            st.write("### Result preview (first 50 rows)")
            st.table(df.head(50))
        else:
            st.write("No result")


class TextPreprocessingPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        left, center, right = st.columns([2, 4, 2])
        left.write("#### 1. Lowercase")
        lowercase = left.checkbox("Convert to lowercase")
        # isalnum,isalph,isdigit
        center.write("#### 2. Numbers and punctuation")
        remove_alphanumeric = center.checkbox("Remove all tokens that include numbers (e.g. ex1ample).")
        remove_non_alpha = center.checkbox("Remove all tokens that include punctuation and numbers (e.g. ex1a.mple).")
        remove_all_numeric = center.checkbox("Remove all tokens that contain numbers only (e.g. 876).")

        right.write("#### 3. Words length")
        display_shorttokens = [0, 1, 2, 3, 4, 5]
        def _temp(token_len):
            if token_len == 0:
                return "Incl. all words"
            else:
                return str(token_len)
        shorttokens = right.selectbox(
            "Remove words shorter than ... characters",
            options=display_shorttokens,
            format_func=lambda x: _temp(x),
        )

        st.write("#### 4. Remove stopwords")
        all_stopword_languages = get_stopwords().fileids()
        languages = st.multiselect("Include stopwords for languages...", options=sorted(all_stopword_languages), help="This downloads stopword lists included in the nltk package.")
        if languages:
            stopwords_op = st.kiara.get_operation("playground.markus.topic_modeling.assemble_stopwords")
            stopword_result = stopwords_op.run(languages=languages)
            stopword_list = stopword_result.get_value_data("stopwords")
        else:
            stopword_list = []
        stopword_expander = st.expander("Current stopwords")
        if stopword_list:
            stopword_expander.dataframe(stopword_list)
        else:
            stopword_expander.write("*No stopwords (yet).*")

        tokens = self.get_step_outputs("tokenization")["tokens_array"]

        preview = None
        if tokens.item_is_valid():
            sample_op = st.kiara.get_operation("array.sample.rows")
            sample_token_array = self._cache.get("preprocess_sample_array", None)
            if not sample_token_array:
                sample_token_array = sample_op.run(value_item=tokens, sample_size=7).get_value_obj("sampled_value")
                self._cache["preprocess_sample_array"] = sample_token_array
            preview_op = st.kiara.get_operation("playground.markus.topic_modeling.preprocess")
            inputs = {
                "to_lowercase": lowercase,
                "remove_alphanumeric": remove_alphanumeric,
                "remove_non_alpha": remove_non_alpha,
                "remove_all_numeric": remove_all_numeric,
                "remove_short_tokens": shorttokens,
                "remove_stopwords": stopword_list
            }
            preview = preview_op.run(token_lists=sample_token_array, **inputs)
        preview_pre_processing = st.checkbox("Preview randomly sampled data (using current inputs)", value=True)
        if preview_pre_processing and preview:
            st.table(preview.get_value_data("preprocessed_token_lists").to_pandas())
        elif preview_pre_processing:
            st.write("No data (yet).")

        confirmation = st.button("Proceed")

        if confirmation:

            step_inputs = {
                "to_lowercase": lowercase,
                "remove_alphanumeric": remove_alphanumeric,
                "remove_non_alpha": remove_non_alpha,
                "remove_all_numeric": remove_all_numeric,
                "remove_short_tokens": shorttokens,
                "remove_stopwords": stopword_list
            }
            with st.spinner("Pre-processing texts..."):
                self.set_pipeline_inputs(inputs=step_inputs)

                print("PROCESSING STEP: 'text_pre_processing'")
                preprocess_result = self.process_step("text_pre_processing")

            if preprocess_result != "Success":
                st.error(preprocess_result)

        # retrieve the actual table value
        preprocessed_table_value = self.get_step_outputs("text_pre_processing").get_value_obj(
            "preprocessed_token_lists"
        )

        if preprocessed_table_value.item_is_valid():
            # if the output exists, we write it as a pandas Series (since streamlit supports that natively)
            df = preprocessed_table_value.get_value_data().to_pandas()
            st.write("### Result preview (first 50 rows)")
            st.table(df.head(50))
        else:
            st.write("No result")

# class LemmatizeTextPage(PipelinePage):
#
#     def run_page(self, st: DeltaGenerator):
#
#         st.write("Here Lorella would write some explanation about what is happening, and why.")
#
#         button = st.button("Lemmatize")
#         if button:
#             with st.spinner("Lemmatizing tokens, this might take a while..."):
#                 self.process_step("lemmatize")
#
#         lemmatized = self.get_step_outputs("lemmatize").get_value_obj(
#             "tokens_array"
#         )
#
#         if lemmatized.item_is_valid():
#             st.dataframe(lemmatized.get_value_data().to_pandas())

class LDAPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        st.write("Here Lorella would write some explanation about what is happening, and why.")

        compute_coherence = st.checkbox("Compute coherence")
        if not compute_coherence:
            number_of_topics_min = st.slider("Number of topics", min_value=1, max_value=40, value=7)
            number_of_topics_max = number_of_topics_min
        else:
            number_of_topics_range = st.slider("Number of topics", 0, 40, (3, 25))
            number_of_topics_min, number_of_topics_max = number_of_topics_range

        button = st.button("Generate LDA")
        if button:
            self.pipeline.inputs.set_values(number_of_topics_min=number_of_topics_min, number_of_topics_max=number_of_topics_max, compute_coherence=compute_coherence)
            with st.spinner("Generating LDA, this might take a while..."):
                self.process_step("generate_lda")

        topic_models = self.get_step_outputs("generate_lda").get_value_obj(
            "topic_models"
        )
        coherence_table = self.get_step_outputs("generate_lda").get_value_obj("coherence_table")

        st.write("### Coherence table")
        if not compute_coherence:
            st.write("Coherence not considered.")
        else:
            if not coherence_table.is_none:
                st.table(coherence_table.get_value_data().to_pandas())
            else:
                st.write("No coherence computed (yet).")

        st.write("### Model details")
        if not topic_models.item_is_valid():
            st.write("No models available (yet).")
        else:
            all_topic_models = topic_models.get_value_data()
            if not compute_coherence:
                selected_model_idx = number_of_topics_min
            else:
                selected_model_idx = st.selectbox("Number of topics", options=range(number_of_topics_min, number_of_topics_max+1))

            try:
                selected_model_table = all_topic_models[selected_model_idx]
                st.table(selected_model_table.to_pandas())
            except KeyError:
                st.write(f"No model for {selected_model_idx} number of topics.")


def onboard_folder(kiara: Kiara, pipeline_folder: str, corpus_path: str, value_alias: str):

    inputs = {
        "corpus_path": corpus_path
    }
    pipeline = os.path.join(pipeline_folder, "example_onboard.yaml")

    store_config_dict = {
        "outputs": [
            {"alias_template": value_alias}
        ]
    }

    onboard_config = {
        "module_type": pipeline,
        "inputs": inputs,
        "store_config": store_config_dict
    }

    # store_config = ValueStoreConfig(**store_config_dict)
    onboarder = BatchOnboard.create(kiara=kiara, **onboard_config)
    print(f"kiara data store: {kiara.data_store.base_path}")
    with st.spinner('Onboarding example data...'):
        results = onboarder.run("test")
        aliases = set()
        for _a in results.values():
            aliases.update(_a)
        print(f"Onboarded example data, available aliases: {', '.join(aliases)}")


@st.experimental_singleton
def onboard_files(_kiara: Kiara):

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    aliases = _kiara.data_store.alias_names
    if aliases:
        all_match = True
        for alias in ["CI_newspaper_subcorpora"]:
            if alias not in aliases:
                all_match = False
                break
        if all_match:
            print("Data already onboarded.")
            return

    base_folder = os.path.dirname(__file__)
    data_folder = os.path.join(base_folder, "data")
    pipeline_folder = os.path.join(base_folder, "pipelines")
    corpus_path = os.path.join(data_folder, "CI_newspaper_subcorpora")

    onboard_folder(kiara=_kiara, pipeline_folder=pipeline_folder, corpus_path=corpus_path, value_alias="CI_newspaper_subcorpora")


    for dir in os.listdir(corpus_path):
        path = os.path.join(corpus_path, dir)
        if os.path.isdir(path):
            name = dir.replace("'", "_")
            onboard_folder(kiara=_kiara, pipeline_folder=pipeline_folder, corpus_path=path, value_alias=name)

    return True

onboarded = onboard_files(_kiara=st.kiara)

# ===============================================================================================================
# main app

main_pipeline = os.path.join(pipelines_folder, "tm_pipeline.yaml")

app = PipelineApp.create(
    pipeline=main_pipeline, config={"show_pipeline_status": True, "show_prev_and_next_buttons": True}
)

if not app.pages:
    app.add_page(AugmentCorpusMetadataPage(id="Prepare qualified table"))
    app.add_page(TimestampedCorpusPage(id="Timestamped data"))
    app.add_page(TokenizationPage(id="Tokenization"))
    app.add_page(TextPreprocessingPage(id="Text pre-processing"))
    # app.add_page(LemmatizeTextPage(id="Lemmatize"))
    app.add_page(LDAPage(id="LDA"))
app.run()
