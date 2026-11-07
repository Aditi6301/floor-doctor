import streamlit as st
from databricks import sql

USE_MOCK = True  # switch to False at the end to use real tables

def table(name):
    return f"workspace.casino.{'mock_' if USE_MOCK else ''}{name}"

@st.cache_data(ttl=600)
def query(q):
    with sql.connect(
        server_hostname=st.secrets["DATABRICKS_HOST"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"],
    ) as conn:
        return conn.cursor().execute(q).fetchall_arrow().to_pandas()
