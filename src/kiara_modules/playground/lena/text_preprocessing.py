# -*- coding: utf-8 -*-
import typing

from kiara import KiaraModule
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from pandas import Series


class PreprocessModuleLena(KiaraModule):
    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "array": {
                "type": "array",
                "doc": "The column to pre-process.",
            },
            "lowercase": {
                "type": "boolean",
                "doc": "Apply lowercasing to the text.",
                "default": False,
            },
            "preprocess_method": {
                "type": "integer",
                "doc": "Pre-processing methodology",
                "default": 0,
            },
            "remove_short_tokens": {
                "type": "integer",
                "doc": "Remove tokens shorter than a certain length",
                "default": 0,
            },
        }

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "preprocessed_array": {
                "type": "array",
                "doc": "The pre-processed content, as an array of lists of strings.",
            }
        }

    def process(self, inputs: ValueSet, outputs: ValueSet):

        import pyarrow as pa

        array: pa.Array = inputs.get_value_data("array")
        lowercase: bool = inputs.get_value_data("lowercase")
        preprocess_method: int = inputs.get_value_data("preprocess_method")
        remove_short_tokens: int = inputs.get_value_data("remove_short_tokens")

        pandas_series: Series = array.to_pandas()

        if lowercase is True:
            pandas_series = pandas_series.apply(lambda x: [w.lower() for w in x])

        # 0 none, 1 isalnum, 2 isalpha, 3 isdigit

        if preprocess_method != 0:

            if preprocess_method == 1:
                pandas_series = pandas_series.apply(
                    lambda x: [w for w in x if w.isalnum()]
                )
            elif preprocess_method == 2:
                pandas_series = pandas_series.apply(
                    lambda x: [w for w in x if w.isalpha()]
                )
            elif preprocess_method == 3:
                pandas_series = pandas_series.apply(
                    lambda x: [w for w in x if not w.isdigit()]
                )

        if remove_short_tokens != 0:
            pandas_series = pandas_series.apply(
                lambda x: [w for w in x if len(w) > remove_short_tokens]
            )

        result_array = pa.Array.from_pandas(pandas_series)

        outputs.set_values(preprocessed_array=result_array)
