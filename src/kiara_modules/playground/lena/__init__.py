# -*- coding: utf-8 -*-
import typing

import networkx as nx
from kiara import KiaraModule
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from kiara.module_config import ModuleTypeConfigSchema
from networkx import Graph
from pydantic import Field

KIARA_METADATA = {
    "authors": [{"name": "Lena Jaskov", "email": "helena.jaskov@uni.lu"}],
}

# Here comes my attempt at building a find largest component module. Maybe need to add config for setting graph type as in CreateGraphFromEdgesTableModule


class FindLargestComponentsModuleConfig(ModuleTypeConfigSchema):

    find_largest_component: bool = Field(
        description="Find the largest component of a graph.", default=True
    )

    number_of_components: bool = Field(
        description="Count the number of components.", default=True
    )


class GraphComponentsModule(KiaraModule):
    """Counts all graph components and creates new graph from largest component."""

    _config_cls = FindLargestComponentsModuleConfig
    _module_type_name = "graph_components"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {"graph": {"type": "network_graph", "doc": "The network graph."}}

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        result = {}
        if self.get_config_value("find_largest_component"):
            result["largest_component"] = {
                "type": "network_graph",
                "doc": "A sub-graph of the largest component of the graph.",
            }

        if self.get_config_value("number_of_components"):
            result["number_of_components"] = {
                "type": "integer",
                "doc": "The number of components in the graph.",
            }

        return result

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        if self.get_config_value("find_largest_component"):
            input_graph: Graph = inputs.get_value_data("graph")
            print(f"INPUT: {input_graph}")
            undir_graph = nx.to_undirected(input_graph)
            undir_components = nx.connected_components(undir_graph)
            lg_component = max(undir_components, key=len)
            subgraph = input_graph.subgraph(lg_component)
            print(f"subgraph: {subgraph}")

            outputs.set_values(largest_component=subgraph)

        if self.get_config_value("number_of_components"):
            input_graph = inputs.get_value_data("graph")
            undir_graph = nx.to_undirected(input_graph)
            number_of_components = nx.number_connected_components(undir_graph)

            outputs.set_values(number_of_components=number_of_components)


# hmm.. how do I run this? kiara run playground.playground.find_largest_component
