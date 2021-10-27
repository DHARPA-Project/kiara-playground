# -*- coding: utf-8 -*-
import typing

from kiara import KiaraModule
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from kiara.exceptions import KiaraProcessingException
from pandas import Series


class TokenizeModule(KiaraModule):
    """Split sentences into words or words into characters.
    In other words, this operation establishes the word boundaries (i.e., tokens) a very helpful way of finding patterns. It is also the typical step prior to stemming and lemmatization
    """

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

        column: pa.Array = table.column(column_name)

        import nltk

        pandas_series: Series = column.to_pandas()

        tokenized = pandas_series.apply(lambda x: nltk.word_tokenize(x))

        result_array = pa.Array.from_pandas(tokenized)

        outputs.set_values(tokens_array=result_array)