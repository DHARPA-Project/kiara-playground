# -*- coding: utf-8 -*-
import streamlit as st
from kiara import Kiara
from streamlit_observable import observable
import json


def app():

    kiara: Kiara = st.session_state["kiara"]

    st.write("Enter local path to json file:")

    st.write("Example: <local path>/kiara-playground/examples/data/geolocation/world_1914.json")

    path = st.text_input("Path to map")

    button = st.button("Load map")

    st.session_state.map_projection = "Mercator"

    if "map_display" not in st.session_state:
        st.session_state.map_display = 0

    if button:
        with open(path) as f:
            map = json.load(f)
        st.session_state.map = json.dumps(map, default=str)

        st.session_state.map_display = 1
    
    if st.session_state.map_display == 1:

        my_expander = st.sidebar.expander(label='Settings')
        
        with my_expander:
            unit = st.selectbox("Projection", ('Mercator', 'Equal Earth', 'Geo Orthographic'))
            selection = st.button("Confirm Selection")
            if selection:
                st.session_state.map_projection  == unit



        observers = observable(
            "Test",
            notebook="@mariellacc/basemap-projections",
            targets=["viewof svgLayer"],
            redefine={
                  "mapData": st.session_state.map,
                  "chooseProjection": unit
              },
        )

        
        
        
      
