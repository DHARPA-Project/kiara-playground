# -*- coding: utf-8 -*-
import re
import typing
from concurrent.futures import ThreadPoolExecutor

from kiara import KiaraModule
from kiara.data import ValueSet
from kiara.data.values import ValueSchema
from kiara.exceptions import KiaraProcessingException


class TokenizeModuleMarkus(KiaraModule):
    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "table": {
                "type": "table",
                "doc": "The table that contains the column to tokenize.",
            },
            "column_name": {
                "type": "string",
                "doc": "The name of the column that contains the content to tokenize.",
                "default": "content",
            },
            "tokenize_by_word": {
                "type": "boolean",
                "doc": "Whether to tokenize by word (default), or character.",
                "default": True,
            },
        }

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "tokens_array": {
                "type": "array",
                "doc": "The tokenized content, as an array of lists of strings.",
            }
        }

    def process(self, inputs: ValueSet, outputs: ValueSet):

        import pyarrow as pa

        table: pa.Table = inputs.get_value_data("table")
        column_name: str = inputs.get_value_data("column_name")
        tokenize_by_word: bool = inputs.get_value_data("tokenize_by_word")

        if column_name not in table.column_names:
            raise KiaraProcessingException(
                f"Can't tokenize table: input table does not have a column named '{column_name}'."
            )


        import nltk
        import vaex
        import warnings
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

        df = vaex.from_arrow_table(table)

        def word_tokenize(word):
            result = nltk.word_tokenize(word)
            return result

        tokenized = df.apply(word_tokenize, arguments=[df[column_name]])
        result_array = tokenized.to_arrow(convert_to_native=True)

        # result_array = pa.Array.from_pandas(tokenized)

        outputs.set_values(tokens_array=result_array)


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

        file_names = inputs.get_value_data("file_name_array")

        replacement_map = inputs.get_value_data("pub_name_replacement_map")

        date_extract_regex = r"(?P<pub_ref>\w+\d+)_(?P<date>\d{4}-\d{2}-\d{2})_"

        result = pa.compute.extract_regex(file_names, pattern=date_extract_regex)
        pub_refs, date_strings = result.flatten()

        dates = pa.compute.strptime(date_strings, format='%Y-%m-%d', unit='s')

        def find_pub_name(pub_ref):
            return replacement_map[pub_ref]

        # import vaex
        # temp_df = vaex.from_arrays(pub_refs=pub_refs)
        # temp_df['pub_names'] = temp_df.apply(find_pub_name, arguments=[temp_df.pub_refs])
        # temp_df.drop("pub_refs")
        # temp_table: pa.Table = temp_df.to_arrow_table()
        # pub_names = temp_table.column("pub_names")

        # this seems to be slightly faster
        pub_names_list = []
        for pub_ref in pub_refs.to_pylist():
            pub_names_list.append(find_pub_name(pub_ref))
        pub_names = pa.array(pub_names_list)

        outputs.set_values(
            dates=dates,
            pub_refs=pub_refs,
            pub_names=pub_names
        )

    # # TODO: replace this with code below after new kiara_modules.core release
    # def process(self, inputs: ValueSet, outputs: ValueSet) -> None:
    #
    #     import pyarrow as pa
    #     from dateutil import parser
    #
    #     file_names = inputs.get_value_data("file_name_array").to_pylist()
    #
    #     replacement_map = inputs.get_value_data("pub_name_replacement_map")
    #
    #     date_extract_regex = r"_(\d{4}-\d{2}-\d{2})_"
    #     pub_ref_extract_regex = r"(\w+\d+)_\d{4}-\d{2}-\d{2}_"
    #
    #     def extract_date(file_name):
    #         date_match = re.findall(date_extract_regex, file_name)
    #         assert date_match
    #         d_obj = parser.parse(date_match[0])  # type: ignore
    #         return d_obj
    #
    #     def extract_pub_ref(file_name):
    #
    #         pub_ref_match = re.findall(pub_ref_extract_regex, file_name)
    #         assert pub_ref_match
    #         return pub_ref_match[0], replacement_map.get(pub_ref_match[0], None)
    #
    #     executor = ThreadPoolExecutor()
    #     results_date: typing.Any = executor.map(extract_date, file_names)
    #     results_pub_ref: typing.Any = executor.map(extract_pub_ref, file_names)
    #
    #     executor.shutdown(wait=True)
    #     dates = list(results_date)
    #     pub_refs, pub_names = zip(*results_pub_ref)
    #
    #     outputs.set_values(
    #         dates=pa.array(dates),
    #         pub_refs=pa.array(pub_refs),
    #         pub_names=pa.array(pub_names),
    #     )
