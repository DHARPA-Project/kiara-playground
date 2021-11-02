# -*- coding: utf-8 -*-
import os
import typing

import pandas as pd
import pyarrow as pa
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import kiara_streamlit
from kiara import Kiara
from kiara.data import ValueSet, Value
from kiara.data.onboarding.batch import BatchOnboard
from kiara_modules.language_processing.tokens import get_stopwords
from kiara_streamlit.data_centric import DataCentricApp, OperationPage

st.set_page_config(page_title="Kiara experiment: data-centric workflows", layout="wide")

if "workflow_pages" not in st.session_state.keys():
    st.session_state.workflow_pages = {}

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})

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

class AugmentTableOperationPage(OperationPage):

    def __init__(self):
        super().__init__(operation=st.kiara.get_operation("playground.augment_corpus_table"))

    def _run_page(self, value: Value) -> typing.Optional[ValueSet]:

        st.caption('A qualified table is a table prepared for Kiara')

        # preview_table = st.checkbox("Preview table")
        # if value and value.item_is_valid():
        #     if preview_table:
        #         st.dataframe(value.get_value_data().to_pandas().head(50))
        # else:
        #     st.markdown("No (valid) table selected, not doing anything...")
        #     return

        # TODO: here we could check whether we have the required columns ('file_name', specifically)

        process_metadata = st.radio("Do your file names contain metadata?", ("no", "yes"))

        st.write("Supported pattern: '/sn86069873/1900-01-05/'")
        st.write("LCCN title information and publication date (yyyy-mm-dd)")

        augment_corpus_result = None
        if process_metadata:
            if process_metadata == "no":
                # if that is not selected, subsequent steps don't make sense
                pass
            elif process_metadata == "yes":

                with st.spinner("Extracting metadata from file names..."):
                    try:
                        augment_corpus_result = self.operation.run(table=value)
                    except Exception as e:
                        st.error(str(e))

        return augment_corpus_result

class TokenizeCorpusPage(OperationPage):

    def __init__(self):
        super().__init__(operation=st.kiara.get_operation("playground.mariella.language.tokenize"))

    def _run_page(self, value: Value) -> typing.Optional[ValueSet]:

        module = self.operation.module

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

        tokenize_result = None
        if token_button:

            inputs={"tokenize_by_word": tokenize == "word", "table": value}
            print("PROCESSING STEP: 'tokenization'")
            with st.spinner('Tokenizing corpus, this may take a while...'):
                try:
                    tokenize_result = self.operation.run(**inputs)
                except Exception as e:
                    st.error(str(e))

        return tokenize_result


class TextPreprocessingPage(OperationPage):

    def __init__(self):
        super().__init__(operation=st.kiara.get_operation("playground.markus.topic_modeling.preprocess"))

    def _run_page(self, value: Value) -> typing.Optional[ValueSet]:

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

        preview = None
        if value.item_is_valid():
            sample_op = st.kiara.get_operation("array.sample.rows")
            sample_token_array = self._cache.get("preprocess_sample_array", None)
            if not sample_token_array:
                sample_token_array = sample_op.run(value_item=value, sample_size=1).get_value_obj("sampled_value")
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

        preview_left, preview_center, preview_right, _ = st.columns([3, 3, 3, 6])
        preview_pre_processing = preview_left.checkbox("Test settings on a sample", value=True, key=self.get_page_key("preview_sample"))
        if preview_pre_processing and preview:
            preview_full = preview_center.checkbox("Preview full text", value=False, key=self.get_page_key("preview_full"))
            # preview_select = preview_center.selectbox("Choose text", options=["x", "y"])

            preview_table: pa.ListArray = preview.get_value_data("preprocessed_token_lists")
            preview_list = preview_table.tolist()
            if preview_full:
                preview_string = ' '.join(preview_list[0])
            else:
                preview_length = 100
                if len(preview_list[0]) <= preview_length:
                    preview_string = ' '.join(preview_list[0])
                else:
                    preview_string = ' '.join(preview_list[0][0:preview_length]) + " ... ... ..."
            md = f"""<style>
  code {{
    white-space : pre-wrap !important;
    word-break: break-word;
  }}
</style>
```
{preview_string}
```
"""
            st.write(md, unsafe_allow_html=True)
            # st.dataframe(preview.get_value_data("preprocessed_token_lists").to_pandas())
        elif preview_pre_processing:
            st.write("No data (yet).")

        confirmation = st.button("Confirm")

        preprocess_result = None
        if confirmation:

            step_inputs = {
                "token_lists": value,
                "to_lowercase": lowercase,
                "remove_alphanumeric": remove_alphanumeric,
                "remove_non_alpha": remove_non_alpha,
                "remove_all_numeric": remove_all_numeric,
                "remove_short_tokens": shorttokens,
                "remove_stopwords": stopword_list
            }
            with st.spinner("Pre-processing texts..."):

                print("PROCESSING STEP: 'text_pre_processing'")
                try:
                    preprocess_result = self.operation.run(**step_inputs)
                except Exception as e:
                    st.error(str(e))

        return preprocess_result


class LDAPage(OperationPage):

    def __init__(self):
        super().__init__(operation=st.kiara.get_operation("playground.markus.topic_modeling.LDA"))

    def _run_page(self, value: Value) -> typing.Optional[ValueSet]:

        st.write("You can now train your topic model. If 'Compute coherence' is selected, you can train several models within a range decided by you. The coherence score assesses the composition of the topics based on how interpretable they are (Röder, Both and Hinneburg 2015).\n\n The highest coherence value would indicate the 'optimal' number of topics, as displayed in the coherence chart below. Please notice that a mathematically more accurate number does not automatically entail that the topics will be more interpretable (Jacobi et al., 2015, p. 7).\n\nYou can decide the number of topics without computing coherence by unselecting 'Compute coherence'.")

        compute_coherence = st.checkbox("Compute coherence")
        if not compute_coherence:
            number_of_topics_min = st.slider("Number of topics", min_value=1, max_value=40, value=7)
            number_of_topics_max = number_of_topics_min
        else:
            number_of_topics_range = st.slider("Number of topics", 0, 40, (3, 25))
            number_of_topics_min, number_of_topics_max = number_of_topics_range

        button = st.button("Generate topics")
        lda_result = None
        if button:
            with st.spinner("Generating topics, this may take a while..."):
                try:
                    lda_result  = self.operation.run(tokens_array=value, num_topics_min=number_of_topics_min, num_topics_max=number_of_topics_max, compute_coherence=compute_coherence)
                except Exception as e:
                    st.error(str(e))


        topic_models = None
        coherence_table = None
        coherence_map = None
        if lda_result:
            topic_models = lda_result.get_value_obj(
                "topic_models"
            )
            coherence_table = lda_result.get_value_obj("coherence_table")
            coherence_map = lda_result.get_value_obj("coherence_map")


        st.write("### Coherence score")
        if not compute_coherence:
            st.write("Coherence not considered.")
        else:
            if coherence_map and not coherence_map.is_none:
                c_map = coherence_map.get_value_data()

                df_coherence = pd.DataFrame(c_map.keys(), columns=['Number of topics'])
                df_coherence['Coherence'] = c_map.values()

                st.vega_lite_chart(df_coherence, {
                    "mark": {"type": "line", "point": True, "tooltip": True},
                    "encoding": {
                        "x": {"field": "Number of topics", "type": "quantitative", "axis": {"format": ".0f"}},
                        "y": {"field": "Coherence", "type": "quantitative", "format": ".3f"}
                    }

                },use_container_width=True)

                st.table(df_coherence)
            else:
                st.write("No coherence computed (yet).")

        st.write("### Model details")
        if not topic_models or (topic_models and not topic_models.item_is_valid()):
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

        return lda_result


class GraphAssemblyPage(OperationPage):

    def __init__(self):
        super().__init__(operation=st.kiara.get_operation("network_graph.from_edges_table"))

    def find_likely_index(self, options: typing.Iterable, keyword: str):

        for idx, alias in enumerate(options):
            if keyword.lower() in alias.lower():
                return idx

        return 0

    def get_table_column_names(self, id):

        if not id:
            return []
        md = st.kiara.data_store.get_value_obj(id)
        if not md:
            return []
        return md.metadata["table"]["metadata_item"]["column_names"]

    def _run_page(self, value: Value) -> typing.Optional[ValueSet]:

        edge_column_names = self.get_table_column_names(value)

        default_source_name = self.find_likely_index(edge_column_names, "source")
        default_target_name = self.find_likely_index(edge_column_names, "target")
        default_weight_name = self.find_likely_index(edge_column_names, "weight")

        input_source, input_target, input_weight, input_nodes_index = st.columns([1, 1, 1, 1])

        source_column_name = input_source.selectbox(
            "Source column name", edge_column_names, index=default_source_name
        )
        target_column_name = input_target.selectbox(
            "Target column name", edge_column_names, index=default_target_name
        )
        weight_column_name = input_weight.selectbox(
            "Weight column name", edge_column_names, index=default_weight_name
        )

        create_button = st.button(label="Create graph")

        outputs = None
        if create_button:
            inputs = {
                "edges_table": value,
                "source_column": source_column_name,
                "target_column": target_column_name,
                "weight_column": weight_column_name
            }

            try:
                outputs = self.operation.run(**inputs)
            except Exception as e:
                st.error(str(e))

        return outputs

class GraphPropertyPage(OperationPage):

    def __init__(self):
        super().__init__(operation=st.kiara.get_operation("network_graph.properties"))

    def _run_page(self, value: Value) -> typing.Optional[ValueSet]:

        md = "### Graph properties"

        properties = self.operation.run(graph=value)

        for prop, value in properties.get_all_value_data().items():
            md = f"{md}\n- {prop}: *{value}*"

        st.write(md)

        return properties

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
    data_folder = os.path.join(base_folder, "..", "tm_prototype_sab", "data")
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

st.header("Data-centric workflows")

op_pages = {
    "playground.augment_corpus_table": AugmentTableOperationPage(),
    "playground.mariella.language.tokenize": TokenizeCorpusPage(),
    "playground.markus.topic_modeling.preprocess": TextPreprocessingPage(),
    "playground.markus.topic_modeling.LDA": LDAPage(),
    "network_graph.from_edges_table": GraphAssemblyPage(),
    "network_graph.properties": GraphPropertyPage()
}


app = DataCentricApp.create(operation_pages=op_pages, config={"show_pipeline_status": True})

app.run()
