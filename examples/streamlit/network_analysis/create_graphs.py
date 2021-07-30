# -*- coding: utf-8 -*-

# run with:
#
# streamlit run examples/streamlit/network_analysis/create_graphs.py

import os
import tempfile
import typing

import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile
from kiara import Kiara
from kiara.processing import JobStatus
from kiara.workflow.kiara_workflow import KiaraWorkflow
from kiara_modules.playground.markus.streamlit import init_session, check_workflow_status, onboard_file, \
    find_all_aliases_of_type

st.title("Kiara/streamlit experiment - create a network graph")

# we'll use a pipeline description here, instead of a Python kiara module name
pipeline_file = os.path.join(os.path.dirname(__file__), "..", "..", "pipelines", "create_graph_from_tables.json")
kiara, workflow = init_session(st, module_type=pipeline_file)

fp = st.sidebar.file_uploader("Import table(s) from csv file(s)", type=["csv"], accept_multiple_files=True)
if fp:
    onboard_file(kiara=kiara, st=st, uploaded_file=fp)

st.sidebar.write("## All your tables:")
all_table_aliases = find_all_aliases_of_type(kiara, value_type="table")
if not all_table_aliases:
    st.sidebar.write(" -- no tables --")
else:
    for a in all_table_aliases:
        st.sidebar.write(a)


def create_graph(alias, edges, nodes, source_column, target_column, weight_column, node_index):

    if not alias:
        return ("No alias specified, doing nothing...", None)

    all_graph_aliases = find_all_aliases_of_type(kiara, value_type="network_graph")
    if alias in all_graph_aliases:
        return (f"Alias '{alias}' already registered.", None)

    if not edges:
        return ("No edges table specified, doing nothing...", None)

    inputs = {
        "edges_table_id": edges,
        "nodes_table_id": nodes,
        "source_column": source_column,
        "target_column": target_column,
        "weight_column": weight_column,
        "index_column_name": node_index
    }

    try:
        workflow = kiara.create_workflow(pipeline_file)
        workflow.inputs.set_values(**inputs)

        graph_value = workflow.outputs.get_value_obj("graph")
        graph_value.save([alias])

        return (f"", graph_value)
    except Exception as e:
        return (f"Error creating graph: {e}", None)

    return ("CREATED GRAPH", None)


def get_table_column_names(table_id):

    if not table_id:
        return []
    md = kiara.data_store.get_metadata_for_id(table_id)
    if not md:
        return []
    return md.metadata["table"]["metadata_item"]["column_names"]


def find_likely_index(options: typing.Iterable, keyword: str):

    for idx, alias in enumerate(options):
        if keyword.lower() in alias.lower():
            return idx

    return 0


graph = None
with st.form(key="create_graph"):

    st.write("Create a new graph")

    graph_alias = st.text_input("The alias for the graph")

    default_edge_table = find_likely_index(all_table_aliases, "edge")
    default_node_table = find_likely_index(all_table_aliases, "node")

    select_edges = st.selectbox("Edges", all_table_aliases, index=default_edge_table)
    # select_nodes = st.selectbox("Nodes", ["-- no nodes --"] + all_files)
    select_nodes = st.selectbox("Nodes", all_table_aliases, index=default_node_table)

    edge_column_names = get_table_column_names(select_edges)
    nodes_column_names = get_table_column_names(select_nodes)

    default_source_name = find_likely_index(edge_column_names, "source")
    default_target_name = find_likely_index(edge_column_names, "target")
    default_weight_name = find_likely_index(edge_column_names, "weight")
    default_id_name = find_likely_index(nodes_column_names, "id")

    source_column_name = st.selectbox("Source column name", edge_column_names, index=default_source_name)
    target_column_name = st.selectbox("Target column name", edge_column_names, index=default_target_name)
    weight_column_name = st.selectbox("Weight column name", edge_column_names, index=default_weight_name)
    nodes_index_name = st.selectbox("Nodes table_index", nodes_column_names, index=default_id_name)

    create_button = st.form_submit_button(label="Create graph")
    if create_button:
        result, graph = create_graph(alias=graph_alias, edges=select_edges, nodes=select_nodes, source_column=source_column_name, target_column=target_column_name, weight_column=weight_column_name,  node_index=nodes_index_name)
        st.write(result)

st. sidebar.write("## All your graphs:")
all_graph_aliases = find_all_aliases_of_type(kiara, value_type="network_graph")
if not all_graph_aliases:
    st.sidebar.write(" -- no graphs --")
else:
    for a in all_graph_aliases:
        st.sidebar.write(a)

if graph is None:
    st.stop()

all_table_aliases = find_all_aliases_of_type(kiara, value_type="table")

st.write("Graph properties")

props = kiara.run("network.graph.properties", inputs={"graph": graph}, resolve_result=True)
st.write(props)

# try:
#     workflow.inputs.set_values(edges_path=edge_path)
#     workflow.controller.process_pipeline()
# except Exception as e:
#     print(e)
#     pass
#
# try:
#     workflow.inputs.set_values(nodes_path=nodes_path)
#     workflow.controller.process_pipeline()
# except Exception as e:
#     print(e)
#     pass
#
# # st.write(workflow.steps.steps.get("load_edges_table").dict())
#
# edges_table = workflow.pipeline.get_step_outputs("load_edges_table")
# edge_column_names = []
# try:
#     edge_column_names = edges_table.get_value_obj("table").get_metadata("table")["table"]["column_names"]
# except:
#     pass
#
# nodes_table = workflow.pipeline.get_step_outputs("load_nodes_table")
# nodes_column_names = []
# try:
#     nodes_column_names = nodes_table.get_value_obj("table").get_metadata("table")["table"]["column_names"]
# except:
#     pass
#
#
# with st.form(key="map_columns"):
#     source_column_name = st.selectbox("Source column name", edge_column_names)
#     target_column_name = st.selectbox("Target column name", edge_column_names)
#     weight_column_name = st.selectbox("Weight column name", edge_column_names)
#     nodes_index_name = st.selectbox("Nodes table_index", nodes_column_names)
#     create_button = st.form_submit_button(label="Map columns")
#
# try:
#     workflow.inputs.set_values(source_column=source_column_name, target_column=target_column_name, nodes_table_index=nodes_index_name, weight_column=weight_column_name)
#     workflow.controller.process_pipeline()
# except Exception as e:
#     print(e)
#     pass
#
#
# success, status = check_workflow_status(workflow)
# st.write('\n'.join(status))
#
# if success:
#     st.write(workflow.pipeline.outputs.get_all_value_data())
# # st.write(workflow.current_state.dict())



