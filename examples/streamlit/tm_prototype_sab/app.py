# -*- coding: utf-8 -*-
import json
import os

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_observable import observable

import kiara_streamlit
from kiara import Kiara
from kiara.data import Value
from kiara.data.onboarding.batch import BatchOnboard
from kiara_modules.language_processing.tokens import get_stopwords
from kiara_streamlit.pipelines import PipelineApp
from kiara_streamlit.pipelines.pages import PipelinePage

from kiara.processing import JobStatus

st.set_page_config(page_title="Kiara-streamlit auto-rendered pipeline", layout="wide")

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})



# =======================================================================================
# create one class per page, the only method that needs to be implemented is 'run_page'

class AugmentCorpusMetadataPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):
        st.caption('A qualified table is a table prepared for Kiara')
        st.info('User story: "I, as a workflow user, want to be able to record, maintain and share metadata that helps computational processes understand the type and schema of research data, so it can be used effectively as an input for a workflow."')

        selected_table: Value = st.kiara.value_input_table(label="Select table", add_no_value_option=True, onboard_options={"enabled": True, "source_default": "folder"}, key=self.get_page_key("selected_table"))
                        
        preview_table = st.checkbox("Preview table")
        if selected_table and selected_table.item_is_valid():
            if preview_table:
                st.dataframe(selected_table.get_value_data().to_pandas().head(50))
            self.set_pipeline_inputs(inputs={"corpus_table": selected_table}, render_errors=True)
        else:
            st.markdown("No (valid) table selected, not doing anything...")
            return

        # TODO: here we could check whether we have the required columns ('file_name', specifically)

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
                preview_table = st.checkbox("Preview results")
                if preview_table:
                   st.dataframe(table.get_value_data().to_pandas().head(50))

class TimestampedCorpusPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        # this is basically unchanged from the other prototype, since it doesn't involve any processing on the actual workflow

        st.info('User story: "I, as a workflow developer or user, want to create one or several visualizations that make my sources easier to understand, and others (or myself) can interact with."')

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
            data_preview = st.button(label="Sources according to selection", help='Display your sources according to the parameters settings on the left panel')
            
        with col2:
            source_view = st.button(label="Sources by time period", help="Hover over the above chart and click on the displayed date to obtain a preview of the content in the table below")

        if data_preview:
            st.session_state.preview_choice = "data"

        if source_view:
            st.session_state.preview_choice = "source"

        display_choice = st.session_state.preview_choice

        if display_choice == "data":

            st.table(query_result_table.to_pandas())

        else:

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

                query_result2 = query_module.module.run(
                    table=augmented_table_value, query=query2
                )
                query_result_value2 = query_result2.get_value_obj("query_result")
                query_result_table2 = query_result_value2.get_value_data()

                df2 = query_result_table2.to_pandas()

                st.dataframe(df2.head(100))

class TokenizationPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        #container.markdown("## Metadata")
        #st.kiara.write_module_type_metadata(module=module, container=container)
        #container.markdown("## Source")
        #st.kiara.write_module_processing_code(module=module, container=container)

        st.info('User story: "I, as a workflow user, want to be able to keep track of every processing step a piece of data goes through, so I can refer back to this later on, and if necessary be able to re-process the source data."')

        step = self.pipeline.get_step("tokenization")
        module = step.module
        
        expander = st.expander("Module info")
        write_module_info_page(module=module, container=expander)

        st.write(
            "For latin-based languages, the default tokenization option is by word"
        )
        st.write(
            "Tokenization is necessary to proceed further. It may take several minutes depending on your corpus size"
        )
        tokenize = st.selectbox("Tokenize by", ("word", "character"), key="0")
        token_button = st.button("Confirm")

        if token_button:

            self.set_pipeline_inputs(inputs={"tokenize_by_word": tokenize == "word"})
            print("PROCESSING STEP: 'tokenization'")
            with st.spinner('Tokenizing corpus, this may take a while...'):
                tokenize_result = self.process_step("tokenization")

            if tokenize_result != "Success":
                st.error(tokenize_result)
                return

        tokenized_table_value = self.get_step_outputs("tokenization").get_value_obj("tokens_array")

        if tokenized_table_value.item_is_valid():
            preview_table = st.checkbox("Preview results")
            if preview_table:
                df = tokenized_table_value.get_value_data().to_pandas()
                st.dataframe(df.head(50))
        else:
            st.write("No result")


class TextPreprocessingPage(PipelinePage):

    def run_page(self, st: DeltaGenerator):

        st.info('User story: "I, as a workflow developer or user, want to preprocess my data so it can be used in subsequent steps."')
        st.info('User story: "I, as a workflow user, would like to see the output of a data process in the context of my dataframe [...]. This would help me explore the preview of a given data process to help me make a critical decision, before applying the process(es)/method(s) to my entire dataframe."')

        
        left, center, right = st.columns([2, 4, 2])
        left.write("##### 1. Lowercase")
        lowercase = left.checkbox("Convert to lowercase")
        # isalnum,isalph,isdigit
        center.write("##### 2. Numbers and punctuation")
        remove_alphanumeric = center.checkbox("Remove all words that contain numbers (e.g. ex1ample)")
        remove_non_alpha = center.checkbox("Remove all words that contain punctuation and numbers (e.g. ex1a.mple)")
        remove_all_numeric = center.checkbox("Remove numbers only (e.g. 876)")

        right.write("##### 3. Words length")
        display_shorttokens = [0, 1, 2, 3, 4, 5]
        def _temp(token_len):
            if token_len == 0:
                return "Select number of characters"
            else:
                return str(token_len)
        shorttokens = right.selectbox(
            "Remove words shorter than X characters",
            options=display_shorttokens,
            format_func=lambda x: _temp(x),
        )

        st.write("##### 4. Remove stopwords")
        all_stopword_languages = get_stopwords().fileids()
        languages = st.multiselect("Select the preferred language(s) for the stopword list(s) (NLTK)", options=sorted(all_stopword_languages))
        if languages:
            stopwords_op = st.kiara.get_operation("playground.markus.topic_modeling.assemble_stopwords")
            stopword_result = stopwords_op.run(languages=languages)
            stopword_list = stopword_result.get_value_data("stopwords")
        else:
            stopword_list = []
        stopword_expander = st.expander("Selected stopwords")
        if stopword_list:
            stopword_expander.dataframe(stopword_list)
        else:
            stopword_expander.write("*No stopwords (yet).*")

        tokens = self.get_step_outputs("tokenization")["tokens_array"]

        preview = None
        if tokens.item_is_valid():
            sample_op = st.kiara.get_operation("sample.array.rows")
            sample_token_array = self._cache.get("preprocess_sample_array", None)
            if not sample_token_array:
                sample_token_array = sample_op.run(array=tokens, sample_size=7).get_value_obj("sampled_value")
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
        preview_pre_processing = st.checkbox("Test settings on a sample", value=True)
        if preview_pre_processing and preview:
            st.dataframe(preview.get_value_data("preprocessed_token_lists").to_pandas())
        elif preview_pre_processing:
            st.write("No data (yet).")

        confirmation = st.button("Confirm")

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
            preview= st.checkbox("Preview results", value=True)
            if preview:
                st.dataframe(df.head(50))
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

        st.info('User story: "I, as a workflow user, want to save the intermediate outputs in an exportable and publishable format (e.g., csv)"')
        
        st.write("You can now train your topic model. If 'Compute coherence' is selected, you can train several models within a range decided by you. The coherence score assesses the composition of the topics based on how interpretable they are (Röder, Both and Hinneburg 2015). The highest coherence value would indicate the 'optimal' number of topics, as displayed in the coherence chart below. Please notice that a mathematically more accurate number does not automatically entail that the topics will be more interpretable (Jacobi et al., 2015, p. 7). You can decide the number of topics without computing coherence by unselecting 'Compute coherence'.")

        compute_coherence = st.checkbox("Compute coherence")
        if not compute_coherence:
            number_of_topics_min = st.number_input('Number of topics', min_value = 1, value=7)
            number_of_topics_max = number_of_topics_min
        else:
            number_of_topics_min = st.number_input('Min number of topics', min_value = 1, value=4)
            number_of_topics_max = st.number_input('Max number of topics', min_value=1, value=7)

        button = st.button("Generate topics")
        if button:
            self.pipeline.inputs.set_values(number_of_topics_min=number_of_topics_min, number_of_topics_max=number_of_topics_max, compute_coherence=compute_coherence)
            with st.spinner("Generating topics, this may take a while..."):
                self.process_step("generate_lda")

        topic_models = self.get_step_outputs("generate_lda").get_value_obj(
            "topic_models"
        )
        coherence_table = self.get_step_outputs("generate_lda").get_value_obj("coherence_table")
        coherence_map = self.get_step_outputs("generate_lda").get_value_obj("coherence_map")

        #left, right = st.columns([7, 2])
        #left.write("### Coherence table")
        #if not compute_coherence:
        #    left.write("Coherence not considered.")
        #else:
        #    if not coherence_table.is_none:
        #        left.table(coherence_table.get_value_data().to_pandas())
        #     else:
        #        left.write("No coherence computed (yet).")

        st.write("### Coherence score")
        if not compute_coherence:
            st.write("Coherence not considered.")
        else:
            if not coherence_map.is_none:
                c_map = coherence_map.get_value_data()
                
                df_coherence = pd.DataFrame(c_map.keys(), columns=['Number of topics'])
                df_coherence['Coherence'] = c_map.values()


                st.vega_lite_chart(df_coherence, {
                    "mark": {"type": "line", "point": True, "tooltip": True},
                    "encoding": {
                        "x": {"field": "Number of topics", "type": "quantitative", "axis": {"format": ".0f", "tickCount": len(df_coherence)-1}},
                        "y": {"field": "Coherence", "type": "quantitative", "format": ".3f"}
                    }
                    
                }, use_container_width=True)

                # .0f

                st.table(df_coherence)
                save = st.checkbox("Save coherence table", key=self.get_page_key("save_selected_model"))
                if save:
                    alias = st.text_input("Alias")
                    save_btn = st.button("Save")
                    if save_btn:
                        if not alias:
                            st.info("Not saving table, no alias provided.")
                        else:
                            saved = coherence_table.save(aliases=[alias])
                            st.info(f"Coherence table saved with alias '{alias}', value id: {saved.id}")

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



def write_module_info_page(
        module, container: DeltaGenerator = st
    ) -> None:
        """Write all available information for a module."""

        full_doc = module.get_type_metadata().documentation.full_doc

        pipeline_str = ""
        if module.is_pipeline() and not pipeline_str == "pipeline":
            pipeline_str = " (pipeline module)"
        container.markdown(
            f"##### Module documentation for: *{module._module_type_id}*{pipeline_str}"  # type: ignore
        )
        container.markdown(full_doc)

        container.caption("Inputs")
        
        st.kiara.valueset_schema_info(module.input_schemas, container=container)

        container.markdown("####")
        container.caption("Outputs")

        st.kiara.valueset_schema_info(
            module.output_schemas,
            show_required=False,
            show_default=False,
            container=container,
        )

        container.markdown("####")
        container.caption("Metadata")
        st.kiara.write_module_type_metadata(module=module, container=container)
        container.markdown("####")
        container.caption("Source")
        st.kiara.write_module_processing_code(module=module, container=container)

def pipeline_status(
        pipeline, container: DeltaGenerator = st
    ) -> None:

        md = "### **Inputs**\n"
        for stage, fields in pipeline.get_pipeline_inputs_by_stage().items():
            md = f"{md}\n"
            for field in fields:
                value = pipeline.inputs[field]
                md = f"{md}* **{field}**: *{value.item_status()}*\n"

        md = f"\n{md}### **Steps**\n"
        for stage, steps in pipeline.get_steps_by_stage().items():
            md = f"{md}\n"
            for step_id in steps.keys():
                job_details = pipeline._controller.get_job_details(step_id)
                if job_details is None:
                    status = "not run yet"
                elif job_details.status == JobStatus.FAILED:
                    status = "failed"
                else:
                    status = "finished"

                md = f"{md}* **{step_id}**: *{status}*\n"

        md = f"\n{md}### **Outputs**\n"
        for stage, fields in pipeline.get_pipeline_outputs_by_stage().items():
            md = f"{md}\n"
            for field in fields:
                value = pipeline.outputs[field]
                status = "ready" if value.is_set else "not ready"
                md = f"{md}* **{field}**: *{status}*\n"

        container.markdown(md)

app = PipelineApp.create(
    pipeline=main_pipeline, config={"show_pipeline_status": False, "show_prev_and_next_buttons": True}
)

if not app.pages:
    app.add_page(AugmentCorpusMetadataPage(id="Prepare your qualified table"))
    app.add_page(TimestampedCorpusPage(id="Visualise your corpus composition"))
    app.add_page(TokenizationPage(id="Tokenization"))
    app.add_page(TextPreprocessingPage(id="Text pre-processing"))
    # app.add_page(LemmatizeTextPage(id="Lemmatize"))
    app.add_page(LDAPage(id="Prepare your topic model"))
app.run()

expander_moduleinfo = st.sidebar.expander(label="Workflow status")

with expander_moduleinfo:
    pipeline_status(pipeline=app.pipeline)

html_string = '''
<style>
    .contributors {
        padding-top: 20%;
        font-size: .8em;
    }
    .contributors h3 {
        font-size: 1.2em;
        padding-bottom: 0.5em;}
    .uni {
        font-size: .9em;
        font-style: italic;
        padding-top:0.5em;
        
    }
</style>
<html>
    <div class="contributors">
        <h3>Created by:</h3>
        Dr Lorella Viola <br>
        Markus Binsteiner <br>
        Mariella De Crouy Chanel<br>
        <p class="uni">
    C2DH/University of Luxembourg
    </p>
    </div>
    
</html>
'''

st.sidebar.write(html_string, unsafe_allow_html=True)

