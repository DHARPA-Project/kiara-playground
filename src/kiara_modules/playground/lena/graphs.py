# -*- coding: utf-8 -*-
import typing
from enum import Enum

import networkx as nx
from kiara import KiaraModule
from kiara.exceptions import KiaraProcessingException
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from kiara.module_config import ModuleTypeConfigSchema
from networkx import Graph
from pydantic import Field, validator

class GraphTypesEnum(Enum):

    undirected = "undirected"
    directed = "directed"
    multi_directed = "multi_directed"
    multi_undirected = "multi_undirected"

KIARA_METADATA = {
    "authors": [{"name": "Lena Jaskov", "email": "helena.jaskov@uni.lu"}],
}

class CreateGraphConfig(ModuleTypeConfigSchema):
    class Config:
        use_enum_values = True

    graph_type: typing.Optional[str] = Field(
        description="The type of the graph. If not specified, a 'graph_type' input field will be added which will default to 'directed'.",
        default=None,
    )

    @validator("graph_type")
    def _validate_graph_type(cls, v):

        try:
            GraphTypesEnum[v]
        except Exception:
            raise ValueError("Invalid graph type name: {v}")

        return v


class CreateGraphFromEdgesTableModule(KiaraModule):
    """Create a directed or undirected network graph object from table data (default to undirected)."""

    _config_cls = CreateGraphConfig
    _module_type_name = "improved_from_edges_table"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        inputs = {
            "edges_table": {
                "type": "table",
                "doc": "The table to extract the edges from.",
            },
            "source_column": {
                "type": "string",
                "default": "source",
                "doc": "The name of the column that contains the edge source in edges table.",
            },
            "target_column": {
                "type": "string",
                "default": "target",
                "doc": "The name of the column that contains the edge target in the edges table.",
            },
            "weight_column": {
                "type": "string",
                "default": "weight",
                "doc": "The name of the column that contains the edge weight in edges table.",
            },
        }

        if self.get_config_value("graph_type") is None:
            inputs["graph_type"] = {
                "type": "string",
                "default": "undirected",
                "doc": "The type of the graph. Allowed: 'undirected', 'directed', 'multi_directed', 'multi_undirected'.",
            }
        return inputs

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "graph": {"type": "network_graph", "doc": "The (networkx) graph object."},
        }

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        import pyarrow as pa

        if self.get_config_value("graph_type") is not None:
            _graph_type = self.get_config_value("graph_type")
        else:
            _graph_type = inputs.get_value_data("graph_type")

        graph_type = GraphTypesEnum[_graph_type]

        edges_table_value = inputs.get_value_obj("edges_table")
        edges_table_obj: pa.Table = edges_table_value.get_value_data()

        source_column = inputs.get_value_data("source_column")
        target_column = inputs.get_value_data("target_column")
        weight_column = inputs.get_value_data("weight_column")

        errors = []
        if source_column not in edges_table_obj.column_names:
            errors.append(source_column)
        if target_column not in edges_table_obj.column_names:
            errors.append(target_column)
        if weight_column not in edges_table_obj.column_names:
            errors.append(weight_column)

        if errors:
            raise KiaraProcessingException(
                f"Can't create network graph, source table missing column(s): {', '.join(errors)}. Available columns: {', '.join(edges_table_obj.column_names)}."
            )

        min_table = edges_table_obj.select(
            (source_column, target_column, weight_column)
        )
        pandas_table = min_table.to_pandas()

        if graph_type is GraphTypesEnum.undirected:
            graph_cls = nx.Graph
        if graph_type is GraphTypesEnum.directed:
            graph_cls = nx.DiGraph
        if graph_type is GraphTypesEnum.multi_directed:
            raise NotImplementedError("Only 'directed' and 'undirected' graphs supported at the moment.")
        if graph_type is GraphTypesEnum.multi_undirected:
            raise NotImplementedError("Only 'directed' and 'undirected' graphs supported at the moment.")
        

        graph: nx.DiGraph = nx.from_pandas_edgelist(
            pandas_table,
            source_column,
            target_column,
            edge_attr=True,
            create_using=graph_cls,
        )
        outputs.set_value("graph", graph)