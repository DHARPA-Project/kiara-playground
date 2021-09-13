# Network-analysis related streamlit experiments

This folder contains network analysis-related streamlit apps.

The apps in here are seperate, but you could easily imagine having a multi-page app where graphs can be created on one page,
and properties of a graph computed on another (where the list of available graphs gets auto-updated, whenever a new graph is created on the first page).

## Create grapsh (not functional at the moment)

Lets users import csv files and create tables from them, then create graph data from any of the available tables.

```
streamlit run examples/streamlit/network_analysis/create_graphs.py
```
