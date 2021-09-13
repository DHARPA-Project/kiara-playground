# -*- coding: utf-8 -*-

"""This module contains the metadata models that are used in the ``kiara-playground`` package.

Metadata models are convenience wrappers that make it easier for *kiara* to find, create, manage and version metadata that
is attached to data, as well as *kiara* modules. It is possible to register metadata using a JSON schema string, but
it is recommended to create a metadata model, because it is much easier overall.

Metadata models must be a sub-class of [kiara.metadata.MetadataModel][kiara.metadata.MetadataModel].
"""
from kiara import KiaraEntryPointItem
from kiara.utils.class_loading import find_metadata_schemas_under

metadata_schemas: KiaraEntryPointItem = (
    find_metadata_schemas_under,
    ["kiara_modules.playground.metadata_schemas"],
)
