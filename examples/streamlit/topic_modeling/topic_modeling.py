# -*- coding: utf-8 -*-

# A streamlit wrapper for the 'topic_modeling_end_to_end.json" kiara pipeline. This is currently only implementing some
# selected steps of the topic modeling workflow, and there is also some issue with spacy (see below).
#
# This pipeline needs an extra library that is provided by spacy. Before you do anything else, run:
#
# python -m spacy download it_core_news_sm
#
# Now, to run the streamlit app, execute:
#
# streamlit run examples/streamlit/topic_modeling/topic_modeling.py
#
# In some/all cases, it seems that some spacy threading/multi-processing setup prevents streamlit from shutting down
# properly, and Ctrl+C doesn't help. If that happens to you, you can kill the process(es) by using:
#
# killall -9 streamlit
#
#
# To use the pipeline using the kiara command-line, run something like below (adjust your own values).
#
# To display the pipeline structure (needs an additional library installed -- it tells you in the output):
#
# kiara pipeline execution-graph examples/pipelines/topic_modeling_end_to_end.json
# or, more detailed:
# kiara pipeline data-flow-graph examples/pipelines/topic_modeling_end_to_end.json
#
# To get help about which inputs it expects:
#
# kiara run examples/pipelines/topic_modeling_end_to_end.json
#
# To actually run it:
#
# kiara run examples/pipelines/topic_modeling_end_to_end.json path=/home/markus/projects/dharpa/notebooks/TopicModelling/data_tm_workflow earliest="1919-01-01" latest="2000-01-01" languages=italian languages=german


import os

import streamlit as st

from kiara_modules.playground.markus.streamlit import (
    get_step_output,
    init_session,
    process_to_stage,
    set_workflow_input,
)

st.title("Kiara/streamlit experiment - Topic modelling")

# This is the workflow we are going to use, check out the json file for details.
workflow_file = os.path.join(
    os.path.dirname(__file__), "..", "..", "pipelines", "topic_modeling_end_to_end.json"
)
# this is a preparation step that:
#  - creates the kiara session
#  - creates a workflow from our json file
#  - saves the workflow object in the streamlit session state object
#
# The last step is useful, because a kiara workflow handles the workflow state itself (e.g. keeping already processed
# results in memory, only re-compute certain steps if relevant inputs have changed, etc.
# This means that we don't need (or want) streamlit to get in the way with its own caching (although in some cases we'll
# take advantage of that too.
kiara, workflow = init_session(st, module_type=workflow_file)

# the variable that holds the user input that points to the folder that contains the txt files/corpus.
path = None
with st.form(key="create_graph"):
    path = st.text_input("Path to corpus")
    load_path = st.form_submit_button(label="Load corpus")

# setting up some input controls on the sidebar
# compute coherence and num_topic are not used at the moment because of some bug with spacy (or gensim -- not sure)
languages = st.sidebar.multiselect("Languages", ["german", "italian", "english"])
# compute_coherence = st.sidebar.checkbox("Compute coherence", value=False)
# num_topics = st.sidebar.slider("Num topic", min_value=1, value=7, max_value=20)

# if no path was specified, it makes no sense to continue
if not path:
    st.stop()

# if we are here, it means the user specfied a path
# all we do here is set the path input in our workflow
set_workflow_input(workflow, path=path)

# now we want to process the parts of the workflow that augments the table with the necessary
# metadata (dates, publication name)
# to find the right processing stage, you can use the kiara cli:
#
# kiara pipeline structure examples/pipelines/topic_modeling_end_to_end.json
process_to_stage(workflow, 5)

# in order to create the date range slider, we need to find the first and last dates of our data
# to do that, we ask kiara to geth us the 'array' output field from the 'extract_date_from_file_name' step.
# Again, use the `kiara pipeline structure ...` command from above to find the right field.
dates = get_step_output(workflow, "extract_date_from_file_name", "array")
ts_series = dates.get_value_data().to_pandas()
start_date = ts_series.min().date()
end_date = ts_series.max().date()
format = "MMM YYYY"  # format output
slider = st.sidebar.slider(
    "Select date",
    min_value=start_date,
    value=(start_date, end_date),
    max_value=end_date,
    format=format,
)

# Now that the slider is created, we connect its min/max values to the appropriate workflow inputs (earliest/latest).
# Again, `kiara pipeline structure ...' is your friend for this. Just be aware that this is the
set_workflow_input(
    workflow,
    earliest=str(slider[0]),
    latest=str(slider[1]),
    languages=languages,
    # num_topic=num_topic,
    # compute_choerence=compute_coherence,
)

# now that we have the earliest/latest user inputs, let's write the intermediate result of the 'filtered_table' step
# to the page, so the user knows what is going on
process_to_stage(workflow, 6)
table = get_step_output(workflow, "filtered_table", "table")
if table.item_is_valid():
    st.write("## Filtered table:")
    st.write(table.get_value_data().to_pandas())

# and also print out the other current inputs
st.write(
    f"## Used inputs:\n\n  - earliest: {slider[0]}\n  - latest: {slider[1]}\n  - Languages: {', '.join(languages)}"
)
# st.write(num_topic)
# st.write(compute_coherence)

# now we need a control that kicks off the processing, once the user is happy with the inputs they set
if st.button("Process"):
    print("BUTTON CLICKED")
    # instead of stopping at an intermediate stage, we tell kiara to process everything (I just used 1000 as stage nr because I know we have less stages)
    process_to_stage(workflow, 1000)
    print("BUTTON FINISHED")

# now, all that is left to do is retrieving the data for the right workflow output, in this case the one named 'tokens_array'
outputs = workflow.outputs.get_value_data("tokens_array")
if outputs:
    # if the output exists, we write it as a pandas Series (since streamlit supports that natively)
    st.write(outputs.to_pandas())
else:
    st.write("No result")
