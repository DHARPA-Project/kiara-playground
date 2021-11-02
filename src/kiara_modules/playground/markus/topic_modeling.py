# -*- coding: utf-8 -*-
import re
import typing
import logging
from concurrent.futures import ThreadPoolExecutor

from pandas import Series

from kiara import KiaraModule
from kiara.data import ValueSet
from kiara.data.values import ValueSchema
from kiara.exceptions import KiaraProcessingException
from kiara_modules.language_processing.tokens import get_stopwords

KIARA_METADATA = {
    "tags": ["language_processing"],
}

# class TokenizeModuleMarkus(KiaraModule):
#     def create_input_schema(
#         self,
#     ) -> typing.Mapping[
#         str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
#     ]:
#
#         return {
#             "table": {
#                 "type": "table",
#                 "doc": "The table that contains the column to tokenize.",
#             },
#             "column_name": {
#                 "type": "string",
#                 "doc": "The name of the column that contains the content to tokenize.",
#                 "default": "content",
#             },
#             "tokenize_by_word": {
#                 "type": "boolean",
#                 "doc": "Whether to tokenize by word (default), or character.",
#                 "default": True,
#             },
#         }
#
#     def create_output_schema(
#         self,
#     ) -> typing.Mapping[
#         str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
#     ]:
#
#         return {
#             "tokens_array": {
#                 "type": "array",
#                 "doc": "The tokenized content, as an array of lists of strings.",
#             }
#         }
#
#     def process(self, inputs: ValueSet, outputs: ValueSet):
#
#         import pyarrow as pa
#
#         table: pa.Table = inputs.get_value_data("table")
#         column_name: str = inputs.get_value_data("column_name")
#         tokenize_by_word: bool = inputs.get_value_data("tokenize_by_word")
#
#         if column_name not in table.column_names:
#             raise KiaraProcessingException(
#                 f"Can't tokenize table: input table does not have a column named '{column_name}'."
#             )
#
#
#         import nltk
#         import vaex
#         import warnings
#         import numpy as np
#         warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#
#         df = vaex.from_arrow_table(table)
#
#         def word_tokenize(word):
#             result = nltk.word_tokenize(word)
#             return result
#
#         tokenized = df.apply(word_tokenize, arguments=[df[column_name]])
#         result_array = tokenized.to_arrow(convert_to_native=True)
#
#         # result_array = pa.Array.from_pandas(tokenized)
#
#         outputs.set_values(tokens_array=result_array)


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
    """Preprocess lists of tokens, incl. lowercasing, remove special characers, etc.

    Lowercasing: Lowercase the words. This operation is a double-edged sword. It can be effective at yielding potentially better results in the case of relatively small datasets or datatsets with a high percentage of OCR mistakes. For instance, if lowercasing is not performed, the algorithm will treat USA, Usa, usa, UsA, uSA, etc. as distinct tokens, even though they may all refer to the same entity. On the other hand, if the dataset does not contain such OCR mistakes, then it may become difficult to distinguish between homonyms and make interpreting the topics much harder.

    Removing stopwords and words with less than three characters: Remove low information words. These are typically words such as articles, pronouns, prepositions, conjunctions, etc. which are not semantically salient. There are numerous stopword lists available for many, though not all, languages which can be easily adapted to the individual researcher's needs. Removing words with less than three characters may additionally remove many OCR mistakes. Both these operations have the dual advantage of yielding more reliable results while reducing the size of the dataset, thus in turn reducing the required processing power. This step can therefore hardly be considered optional in TM.

    Noise removal: Remove elements such as punctuation marks, special characters, numbers, html formatting, etc. This operation is again concerned with removing elements that may not be relevant to the text analysis and in fact interfere with it. Depending on the dataset and research question, this operation can become essential.
    """

    KIARA_METADATA = {
    "tags": ["tokens", "preprocess"],
    }
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

            # all-number tokens are already filtered out if the remove_non_alpha methods above ran
            if remove_all_numeric and not remove_non_alpha:
                token = None if token.isdigit() else token
                if token is None:
                    return None

            if remove_stopwords and token in remove_stopwords:
                return None

            return token


        processed = pandas_series.apply(lambda token_list: [x for x in (check_token(token) for token in token_list) if x is not None])
        result_array = pa.Array.from_pandas(processed)

        outputs.set_values(preprocessed_token_lists=result_array)


class LDAModule(KiaraModule):
    """Perform Latent Dirichlet Allocation on a tokenized corpus.

    This module computes models for a range of number of topics provided by the user.
    """

    _module_type_name = "LDA"

    KIARA_METADATA = {
        "tags": ["LDA", "tokens"],
    }

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        inputs: typing.Dict[str, typing.Dict[str, typing.Any]] = {
            "tokens_array": {"type": "array", "doc": "The text corpus."},
            "num_topics_min": {
                "type": "integer",
                "doc": "The minimal number of topics.",
                "default": 7,
            },
            "num_topics_max": {
                "type": "integer",
                "doc": "The max number of topics.",
                "optional": True
            },
            "compute_coherence": {
                "type": "boolean",
                "doc": "Whether to compute the coherence score for each model.",
                "default": False
            },
            "words_per_topic": {
                "type": "integer",
                "doc": "How many words per topic to put in the result model.",
                "default": 10
            }
        }
        return inputs

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        outputs = {
            "topic_models": {
                "type": "dict",
                "doc": "A dictionary with one coherence model table for each number of topics."
            },
            "coherence_table": {
                "type": "table",
                "doc": "Coherence details.",
                "optional": True
            },
            "coherence_map": {
                "type": "dict",
                "doc": "A map with the coherence value for every number of topics."
            }
        }
        return outputs

    def create_model(self, corpus, num_topics: int, id2word: typing.Mapping[str, int]):
        from gensim.models import LdaModel

        model = LdaModel(
            corpus, id2word=id2word, num_topics=num_topics, eval_every=None
        )
        return model

    def compute_coherence(self, model, corpus_model, id2word: typing.Mapping[str, int]):

        from gensim.models import CoherenceModel

        coherencemodel = CoherenceModel(
            model=model, texts=corpus_model, dictionary=id2word, coherence="c_v",processes=1
        )
        coherence_value = coherencemodel.get_coherence()
        return coherence_value

    def assemble_coherence(self, models_dict: typing.Mapping[int, typing.Any], words_per_topic: int):

        import pandas as pd
        import pyarrow as pa

        # Create list with topics and topic words for each number of topics
        num_topics_list = []
        topics_list = []
        for num_topics, model, in models_dict.items():

            num_topics_list.append(num_topics)
            topic_print = model.print_topics(num_words=words_per_topic)
            topics_list.append(topic_print)

        df_coherence_table = pd.DataFrame(columns=["topic_id", "words", "num_topics"])

        idx = 0
        for i in range(len(topics_list)):
            for j in range(len(topics_list[i])):
                df_coherence_table.loc[idx] = ""
                df_coherence_table["topic_id"].loc[idx] = j + 1
                df_coherence_table["words"].loc[idx] = ", ".join(
                    re.findall(r'"(\w+)"', topics_list[i][j][1])
                )
                df_coherence_table["num_topics"].loc[idx] = num_topics_list[i]
                idx += 1

        coherence_table = pa.Table.from_pandas(df_coherence_table, preserve_index=False)
        return coherence_table

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        from gensim.models import LdaModel
        import pandas as pd
        from gensim import corpora
        from pyarrow import Table

        logging.getLogger("gensim").setLevel(logging.ERROR)
        tokens_array = inputs.get_value_data("tokens_array")
        tokens = tokens_array.to_pylist()

        words_per_topic = inputs.get_value_data("words_per_topic")

        num_topics_min = inputs.get_value_data("num_topics_min")
        num_topics_max = inputs.get_value_data("num_topics_max")
        if num_topics_max is None:
            num_topics_max= num_topics_min

        compute_coherence = inputs.get_value_data("compute_coherence")
        id2word = corpora.Dictionary(tokens)
        corpus = [id2word.doc2bow(text) for text in tokens]

        # model = gensim.models.ldamulticore.LdaMulticore(
        #     corpus, id2word=id2word, num_topics=num_topics, eval_every=None
        # )

        models = {}
        model_tables = {}
        coherence = {}

        # multi_threaded = False
        # if not multi_threaded:

        for nt in range(num_topics_min, num_topics_max+1):
            model = self.create_model(corpus=corpus, num_topics=nt, id2word=id2word)
            models[nt] = model
            topic_print_model = model.print_topics(num_words=words_per_topic)
            df = pd.DataFrame(topic_print_model, columns=["topic_id", "words"])
            # TODO: create table directly
            result_table = Table.from_pandas(df)
            model_tables[nt] = result_table
            if compute_coherence:
                coherence_result = self.compute_coherence(model=model, corpus_model=tokens, id2word=id2word)
                coherence[nt] = coherence_result

        # else:
        #     def create_model(num_topics):
        #         model = self.create_model(corpus=corpus, num_topics=num_topics, id2word=id2word)
        #         topic_print_model = model.print_topics(num_words=30)
        #         df = pd.DataFrame(topic_print_model, columns=["topic_id", "words"])
        #         # TODO: create table directly
        #         result_table = Table.from_pandas(df)
        #         coherence_result = None
        #         if compute_coherence:
        #             coherence_result = self.compute_coherence(model=model, corpus_model=tokens, id2word=id2word)
        #         return (num_topics, model, result_table, coherence_result)
        #
        #     executor = ThreadPoolExecutor()
        #     results: typing.Any = executor.map(create_model, range(num_topics_min, num_topics_max+1))
        #     executor.shutdown(wait=True)
        #     for r in results:
        #         models[r[0]] = r[1]
        #         model_tables[r[0]] = r[2]
        #         if compute_coherence:
        #             coherence[r[0]] = r[3]

        # df_coherence = pd.DataFrame(coherence.keys(), columns=["Number of topics"])
        # df_coherence["Coherence"] = coherence.values()

        if compute_coherence:
            coherence_table = self.assemble_coherence(models_dict=models, words_per_topic=words_per_topic)
        else:
            coherence_table = None

        outputs.set_values(topic_models=model_tables, coherence_table=coherence_table, coherence_map=coherence)


def install_and_import_spacy_package(package):
    import importlib
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        import spacy
        print("INSTALLING")
        spacy.cli.download(package)
        pkg = importlib.import_module(package)
        globals()[package] = pkg

    return pkg

class LemmatizeTokensArrayModule(KiaraModule):
    """Lemmatize an array of token lists.

    Compared to using the ``lemmatize_tokens`` module in combination with ``map``, this is much faster, since it uses
    a spacy [pipe](https://spacy.io/api/language#pipe) under the hood.
    """

    KIARA_METADATA = {
        "tags": ["lemma", "tokens", "lemmatize"],
    }

    _module_type_name = "lemmatize_tokens_array"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        inputs = {
            "tokens_array": {"type": "array", "doc": "An array of lists of tokens."}
        }
        return inputs

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        outputs = {
            "tokens_array": {
                "type": "array",
                "doc": "An array of lists of lemmatized tokens.",
            }
        }
        return outputs

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        import pyarrow as pa
        from spacy.tokens import Doc
        from spacy.util import DummyTokenizer

        tokens: pa.Array = inputs.get_value_data("tokens_array")

        # it_core_news_sm = install_and_import_spacy_package("it-core-news-sm")
        import it_core_news_sm
        it_nlp = it_core_news_sm.load(disable=["tagger", "parser", "ner"])
        # it_nlp = it_core_news_sm.load(disable=["tagger"])

        use_pipe = False

        if not use_pipe:

            result = []
            for t_list in tokens:
                t = []
                for w in t_list:
                    w_lemma = [token.lemma_ for token in it_nlp(w.as_py())]
                    t.append(w_lemma[0])
                result.append(t)

        else:

            class CustomTokenizer(DummyTokenizer):
                def __init__(self, vocab):
                    self.vocab = vocab

                def __call__(self, words):
                    return Doc(self.vocab, words=words)

            it_nlp.tokenizer = CustomTokenizer(it_nlp.vocab)
            result = []

            for doc in it_nlp.pipe(
                tokens.to_pylist(),
                batch_size=32,
                n_process=3,
                disable=["parser", "ner", "tagger"],
            ):
                result.append([tok.lemma_ for tok in doc])

        print("FINISHED")
        outputs.set_value("tokens_array", pa.array(result))
