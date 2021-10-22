# -*- coding: utf-8 -*-
import re
import typing
from concurrent.futures import ThreadPoolExecutor

from pandas import Series

from kiara import KiaraModule
from kiara.data import ValueSet
from kiara.data.values import ValueSchema
from kiara.exceptions import KiaraProcessingException
from kiara_modules.language_processing.tokens import get_stopwords


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
        import numpy as np
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


class AssembleStopwordsModule(KiaraModule):
    """Create a list of stopwords from one or multiple sources.

    This will download nltk stopwords if necessary, and merge all input lists into a single, sorted list without duplicates.
    """

    _module_type_name = "assemble_stopwords"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "languages": {
                "type": "list",
                "doc": "A list of languages, will be used to retrieve language-specific stopword from nltk.",
                "optional": True
            },
            "stopword_lists": {
                "type": "list",
                "doc": "A list of lists of stopwords.",
                "optional": True
            }
        }

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        return {
            "stopwords": {
                "type": "list",
                "doc": "A sorted list of unique stopwords."
            }
        }

    def process(self, inputs: ValueSet, outputs: ValueSet):

        stopwords = set()
        languages = inputs.get_value_data("languages")
        if languages:
            all_stopwords = get_stopwords()
            for language in languages:
                if language not in all_stopwords.fileids():
                    raise KiaraProcessingException(
                        f"Invalid language: {language}. Available: {', '.join(all_stopwords.fileids())}."
                    )
                stopwords.update(get_stopwords().words(language))

        stopword_lists = inputs.get_value_data("stopword_lists")
        if stopword_lists:
            for stopword_list in stopword_lists:
                if isinstance(stopword_list, str):
                    stopwords.add(stopword_list)
                else:
                    stopwords.update(stopword_list)

        outputs.set_value("stopwords", sorted(stopwords))


class PreprocessModule(KiaraModule):
    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "token_lists": {
                "type": "array",
                "doc": "The column to pre-process.",
            },
            "to_lowercase": {
                "type": "boolean",
                "doc": "Apply lowercasing to the text.",
                "default": False,
            },
            "remove_alphanumeric": {
                "type": "boolean",
                "doc": "Remove all tokens that include numbers (e.g. ex1ample).",
                "default": False
            },
            "remove_non_alpha": {
                "type": "boolean",
                "doc": "Remove all tokens that include punctuation and numbers (e.g. ex1a.mple).",
                "default": False
            },
            "remove_all_numeric": {
                "type": "boolean",
                "doc": "Remove all tokens that contain numbers only (e.g. 876).",
                "default": False
            },
            "remove_short_tokens": {
                "type": "integer",
                "doc": "Remove tokens shorter than a certain length. If value is <= 0, no filtering will be done.",
                "default": False,
            },
            "remove_stopwords": {
                "type": "list",
                "doc": "Remove stopwords.",
                "optional": True
            }
        }

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "preprocessed_token_lists": {
                "type": "array",
                "doc": "The pre-processed content, as an array of lists of strings.",
            }
        }

    def process(self, inputs: ValueSet, outputs: ValueSet):

        import pyarrow as pa

        array: pa.Array = inputs.get_value_data("token_lists")
        lowercase: bool = inputs.get_value_data("to_lowercase")
        remove_alphanumeric: bool = inputs.get_value_data("remove_alphanumeric")
        remove_non_alpha: bool = inputs.get_value_data("remove_non_alpha")
        remove_all_numeric: bool = inputs.get_value_data("remove_all_numeric")
        remove_short_tokens: int = inputs.get_value_data("remove_short_tokens")

        remove_stopwords: list = inputs.get_value_data("remove_stopwords")
        pandas_series: Series = array.to_pandas()

        # it's better to have one method every token goes through, then do every test seperately for the token list
        # because that way each token only needs to be touched once (which is more effective)
        def check_token(token: str) -> typing.Optional[str]:

            # remove short tokens first, since we can save ourselves all the other checks (which are more expensive)
            if remove_short_tokens > 0:
                if len(token) <= remove_short_tokens:
                    return None

            if lowercase:
                token = token.lower()

            if remove_non_alpha:
                token = token if token.isalpha() else None
                if token is None:
                    return None

            # if remove_non_alpha was set, we don't need to worry about tokens that include numbers, since they are already filtered out
            if remove_alphanumeric and not remove_non_alpha:
                token = token if token.isalnum() else None
                if token is None:
                    return None

            # all-number tokens are already filtered out if any of the other methods above ran
            if remove_all_numeric and not remove_non_alpha and not remove_alphanumeric:
                token = None if token.isdigit() else token
                if token is None:
                    return None

            if remove_stopwords and token in remove_stopwords:
                return None

            return token


        processed = pandas_series.apply(lambda token_list: [x for x in (check_token(token) for token in token_list) if x is not None])
        result_array = pa.Array.from_pandas(processed)

        outputs.set_values(preprocessed_token_lists=result_array)
