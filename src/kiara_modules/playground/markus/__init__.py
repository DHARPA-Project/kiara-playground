# -*- coding: utf-8 -*-
import typing

from kiara import KiaraModule
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from kiara_modules.core.metadata_schemas import KiaraFile
from pandas import DataFrame
from pyarrow import csv


class CreateGraphFromFileModule(KiaraModule):
    """Load table-like data from a *kiara* file object (not a path!)."""

    _module_type_name = "from_file"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        inputs = {
            "file": {
                "type": "file",
                "doc": "The file that contains table data.",
                "optional": False,
            }
        }

        if self.get_config_value("allow_column_filter"):

            inputs["columns"] = {
                "type": "array",
                "doc": "If provided, only import the columns that match items in this list.",
                "optional": False,
            }

        return inputs

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        return {"table": {"type": "table", "doc": "the imported table"}}

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        input_file: KiaraFile = inputs.get_value_data("file")
        imported_data = csv.read_csv(input_file.path)

        if self.get_config_value("allow_column_filter"):
            if self.get_config_value("columns"):
                imported_data = imported_data.select(
                    self.get_config_value("only_columns")
                )

        outputs.set_value("table", imported_data)


class MyFirstModule(KiaraModule):

    _module_type_name = "filter_table_by_date"

    def create_input_schema(self):
        return {
            "table_input": {"type": "table", "doc": "The table that will be filtered."},
            "date": {
                "type": "date",
                "doc": "The minimum date, earlier dates will be filtered out.",
            },
        }

    def create_output_schema(self):
        return {"table_output": {"type": "table", "doc": "The filtered table."}}

    def process(self, inputs, outputs):

        table_obj = inputs.get_value_data("table_input")
        date_input = inputs.get_value_data("date")

        df: DataFrame = table_obj.to_pandas()

        after_date = df[df["birthday"] >= date_input.date()]

        import pyarrow as pa

        result_table = pa.Table.from_pandas(after_date, preserve_index=False)

        outputs.set_value("table_output", result_table)
