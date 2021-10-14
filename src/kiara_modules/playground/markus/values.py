import typing

import networkx as nx

from kiara import KiaraModule
from kiara.data import ValueSet
from kiara.data.values import ValueSchema, Value, ValueLineage
from kiara.exceptions import KiaraProcessingException


class ExtractLineageGraphModule(KiaraModule):

    _module_type_name: "extract_lineage_graph"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "value_item": {
                "type": "any",
                "doc": "The value in question."
            }
        }

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        return {
            "lineage_graph": {
                "type": "network_graph",
                "doc": "The values lineage as a graph object."
            }
        }

    def process(self, inputs: ValueSet, outputs: ValueSet):

        value_item: Value = inputs.get_value_obj("value_item")

        lineage: typing.Optional[ValueLineage] = value_item.get_lineage()
        if not lineage:
            raise KiaraProcessingException(f"Selected value (id: {value_item.id}) does not have lineage attached to it.")

        graph = lineage.create_graph()
        outputs.set_value("lineage_graph", graph)
