# -*- coding: utf-8 -*-

"""Top-level package for kiara-playground."""
import typing

import logging
import os
import pandas as pd
import pyarrow as pa
import geopy

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from kiara import KiaraEntryPointItem, find_kiara_modules_under

from kiara import KiaraModule
from kiara.data.values import ValueSchema
from kiara.data.values.value_set import ValueSet
from kiara.module_config import ModuleTypeConfigSchema


KIARA_METADATA = {
    "authors": [{"name": "Angela Cunningham", "email": "angela.cunningham@uni.lu"}]
}

from kiara import KiaraModule

class GeoCode(KiaraModule):
    '''Derive latitude and longitude coordinates from text string(s)'''

    _module_type_name = "geocode_strings"

    

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:

        return {
            "raw_table": {
                "type": "table",
                "doc": "The imported table of places names or addresses.",
            },
            
            "city_column": {
                "type": "string",
                "default": "city",
                "doc": "The name of the column in the table with city names.",
            },
            
            "country_column": {
                "type": "string",
                "default": "country",
                "doc": "The name of the column in the table with country names.",
            },
            
        }


    def create_output_schema(self):
        return {
            'geocoded_table': {
                'type': 'table',
                'doc': 'Geocoded table'

            },
        }



#use a structured query
#https://stackoverflow.com/questions/61756302/using-structured-queries-to-geocode-records-in-a-pandas-dataframe-using-geopy 
    def process(self, inputs, outputs) -> None:

#set up geocoder
        Ngeolocator = Nominatim(user_agent="myGeocoderNew")
        Ngeocode = RateLimiter(Ngeolocator.geocode, min_delay_seconds=1)

#get data from kiara table        
        raw_table_value=inputs.get_value_obj("raw_table")
        raw_table_obj: pa.Table=raw_table_value.get_value_data()

        #what if don't have address?
        
        city_column=inputs.get_value_data("city_column")
        country_column=inputs.get_value_data("country_column")
        

        min_table= raw_table_obj.select((city_column, country_column))
        pandas_table=min_table.to_pandas()
            
        pandas_table["geocoderesult"] = pandas_table.apply(
            lambda row: Ngeocode(
                {
                    "city": row["city_column"],
                    "country": row["country_column"],
                },
                language="en",
                addressdetails=True,
            ).raw,
            axis=1,
        )
        
        pandas_table['longitude'] = pandas_table['geocoderesult'].apply(lambda x: (x.point[1]) if x else None)
        pandas_table['latitude'] = pandas_table['geocoderesult'].apply(lambda x: (x.point[0]) if x else None)

#back to pyarrow
        result_table: pa.Table.from_pandas(pandas_table)

        outputs.set_values(geocoded_table=result_table)
        
        

