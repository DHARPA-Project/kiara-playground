# -*- coding: utf-8 -*-
import streamlit as st
from kiara import Kiara


def app():

    kiara: Kiara = st.session_state["kiara"]

    st.markdown(
        "Download the corpus on your computer, unzip and copy local folder path"
    )
    st.markdown(
        "https://zenodo.org/record/4596345/files/ChroniclItaly_3.0_original.zip?download"
    )
    st.markdown("Paste local folder path into input below")
    st.markdown(
        "Wait for the success message, and then select next page in top left nav menu"
    )

    path = st.text_input("Path to files folder")

    button = st.button("Onboard")

    if button:
        # CHANGED
        # module = kiara.operation_mgmt.profiles[
        #     "table.import_from.folder_path.string"
        # ].module
        # aliases = ["my_first_table"]
        # result = module.run(source=path, aliases=aliases)

        # the 'table.import_from.folder_path.string' operation was removed -- we'll use 2 separate ones instead here (without the user needing to know)
        file_import_operation = kiara.get_operation(
            "file_bundle.import_from.local.folder_path"
        )
        # note: currently this operation also supports 'save' and 'aliases' inputs, but we'll do this manually, because I think I might remove those later
        import_result = file_import_operation.module.run(source=path)
        imported_file_bundle = import_result.get_value_obj("value_item")
        # we are going to save this imported value now manually (otherwise it would not be persisted in the kiara data store -- which would be ok in this example, but we might want to offer users to re-use already imported file_bundles later)

        alias = "my_first_table"  # we can use a streamlit text field for this later
        imported_file_bundle.save(
            aliases=[f"{alias}__source_files"]
        )  # an alias to make it easier to see what this value represents (the source for the table)

        # now we need to convert the file to a table object
        table_convert_op = kiara.get_operation("file_bundle.convert_to.table")
        convert_result = table_convert_op.module.run(value_item=imported_file_bundle)
        imported_table = convert_result.get_value_obj("value_item")
        # let's also save the table (for a potential next run)
        imported_table.save(aliases=[alias])

        # we can just re-use the table object we got as a result earlier, so the first line her is not necessary anymore
        # table_value = kiara.data_store.load_value(aliases[0])
        # st.session_state.data = table_value

        st.session_state.data = imported_table

        st.write("Success! Select the next step from the top left nav menu.")
