# -*- coding: utf-8 -*-
import re
import typing
from concurrent.futures import ThreadPoolExecutor

from kiara import KiaraModule
from kiara.data import ValueSet
from kiara.data.values import ValueSchema


class ExtractDateAndPubRefModule(KiaraModule):
    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "file_name_array": {"type": "array", "doc": "An array of file names."},
            "pub_name_replacement_map": {
                "type": "dict",
                "doc": "A map with pub_refs as keys, and publication names as values.",
            },
        }

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "dates": {"type": "array", "doc": "An array of extracted dates."},
            "pub_refs": {"type": "array", "doc": "An array of publication references."},
            "pub_names": {"type": "array", "doc": "An array of publication names."},
        }

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        import pyarrow as pa
        from dateutil import parser

        file_names = inputs.get_value_data("file_name_array").to_pylist()

        replacement_map = inputs.get_value_data("pub_name_replacement_map")

        date_extract_regex = r"_(\d{4}-\d{2}-\d{2})_"
        pub_ref_extract_regex = r"(\w+\d+)_\d{4}-\d{2}-\d{2}_"

        def extract_date(file_name):
            date_match = re.findall(date_extract_regex, file_name)
            assert date_match
            d_obj = parser.parse(date_match[0])  # type: ignore
            return d_obj

        def extract_pub_ref(file_name):

            pub_ref_match = re.findall(pub_ref_extract_regex, file_name)
            assert pub_ref_match
            return pub_ref_match[0], replacement_map[pub_ref_match[0]]

        executor = ThreadPoolExecutor()
        results_date: typing.Any = executor.map(extract_date, file_names)
        results_pub_ref: typing.Any = executor.map(extract_pub_ref, file_names)

        executor.shutdown(wait=True)
        dates = list(results_date)
        pub_refs, pub_names = zip(*results_pub_ref)

        outputs.set_values(
            dates=pa.array(dates),
            pub_refs=pa.array(pub_refs),
            pub_names=pa.array(pub_names),
        )
