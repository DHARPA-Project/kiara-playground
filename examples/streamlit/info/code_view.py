# -*- coding: utf-8 -*-
import streamlit as st
from kiara import Kiara

# it's best to encapsulate functionality like this in a function, so we can re-use it
# the 'container' argument is only there so we can potentially write the module info/codeview
# onto another component (like a column), and not just the root page, you can ignore that for now
# just use 'container.write(...)' whenever you would use 'st.write(...)')
from kiara.operations import Operation
from streamlit.delta_generator import DeltaGenerator


def write_operation_info(
    kiara: Kiara, operation_id: str, container: DeltaGenerator = st
):

    # this retrieve the object for the operation we are interested in
    operation: Operation = kiara.operation_mgmt.profiles[operation_id]

    # this is all operation specific data that is available, pick what you need
    container.markdown("## Operation info")
    container.markdown("### Type metadata")
    type_metadata = operation.module_cls.get_type_metadata()
    container.write(type_metadata.dict())
    # you can access specific elements via the Python object attributes, like:
    # type_metadata.context.references
    container.markdown(("### Inputs"))
    for field_name, schema in operation.input_schemas.items():
        container.write(f"#### {field_name}")
        container.write(schema.dict())
    container.markdown(("### Outputs"))
    for field_name, schema in operation.output_schemas.items():
        container.write(f"#### {field_name}")
        container.write(schema.dict())


kiara = Kiara()

# get all module types
kiara_module_types = kiara.operation_mgmt.profiles.keys()
selected_operation = st.selectbox("Select the module type", kiara_module_types)


write_operation_info(kiara=kiara, operation_id=selected_operation)
