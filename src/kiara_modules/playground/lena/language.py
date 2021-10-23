# -*- coding: utf-8 -*-
import typing

from kiara import KiaraModule
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from kiara.exceptions import KiaraProcessingException
from pandas import Series


class TokenizeModuleLena(KiaraModule):
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
            "tokenizing_method": {
                "type": "boolean",
                "doc": "Whether to tokenize with nltk (default), or fugashi.",
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
        tokenizing_method: bool = inputs.get_value_data("tokenizing_method")

        if column_name not in table.column_names:
            raise KiaraProcessingException(
                f"Can't tokenize table: input table does not have a column named '{column_name}'."
            )

        column: pa.Array = table.column(column_name)

        pandas_series: Series = column.to_pandas()

        if tokenizing_method is True:
            import nltk

            tokenized = pandas_series.apply(lambda x: nltk.word_tokenize(x))

        elif tokenizing_method is False:
            import fugashi

            tagger = fugashi.Tagger()

            taggered = pandas_series.apply(lambda x: tagger(x))

            def token(x):
                return [str(item.surface) for item in tagger(x)]

            #lemma is currently not used, but could be an additional output, since it can be done in the same step
            def lemma(x):
                return [str(item.feature.lemma) for item in tagger(x)]
            lemmatized = pandas_series.apply(lambda x: lemma(x))

            surfaced = pandas_series.apply(lambda x: token(x))  

            tokenized = surfaced.apply(lambda x: str(x))

            #retrieve tokenized words with item.surface and lemmas with item.featue.lemma
            # for sublist in taggered:
            #     for item in sublist:
            #         print(item.surface)

            #none of this worked...
            # def unidic_node_to_str(node_list):
            #     return [str(word) for word in node_list]
            #tokenized = taggered.apply(lambda x: str(x))

            #tokenized = unidic_node_to_str(taggered)

        #stringified = pandas_series.apply(unidic_node_to_str)
        # import nltk
        #tokenized = unidic_node_to_str(taggered)

        # print(tokenized)
        # print("=========================================")

        # this is how you can get a Pandas Series from the column
        # print("=========================================")
        # pandas_series: Series = column.to_pandas()
        # print(pandas_series)
        # print("=========================================")

        # do your stuff here

        # then convert your result into an Arrow Array again
        # below is just a fake result, but should give you an idea how to do it
        # fake_result = [['x', 'y'], ['a', 'b'], ['c', 'd']]
        # fake_result_series = Series(fake_result)

        result_array = pa.Array.from_pandas(tokenized)

        outputs.set_values(tokens_array=result_array)
