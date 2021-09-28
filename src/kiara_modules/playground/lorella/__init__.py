# -*- coding: utf-8 -*-

"""Top-level package for kiara-playground."""


import logging
import os

from kiara import KiaraEntryPointItem, find_kiara_modules_under

KIARA_METADATA = {
    "authors": [{"name": "Lorella Viola", "email": "YOUR EMAIL"}]
}

from kiara import KiaraModule

class MyFirstModule(KiaraModule):
    '''Tokenize text at sentence level'''

    _module_type_name = "tokenize_sentences"

    def create_input_schema(self):
        return {
            "text_file": {
                "type": "file",
                "doc": "The source text"
            },
       
        }

    def create_output_schema(self):
        return {
            'sentences': {
                'type': 'table',
                'doc': 'Tokenized sentences'

            },
        }
 
    def process(self, inputs, outputs) -> None:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        import pandas as pd
        import pyarrow as pa
       

        text_file = inputs.get_value_data("text_file") #using kiara command to retrieve data
        
        with open(text_file.path, 'r', encoding='utf8') as infile: #opening file in read mode

            text = infile.read()

        sentences = sent_tokenize(text)

        df = pd.DataFrame(columns=['Sentence']) #building a df where each row is a sentence

        for r in sentences:
            df.loc[len(df)] = [r]

        table = pa.Table.from_pandas(df) #creating a variable 'table' with arrow
        outputs.set_values(sentences=table) #using kiara command to assign sentences as the table output 
