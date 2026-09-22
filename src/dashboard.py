"""Customer segmentation dashboard.

    uv run streamlit run src/dashboard.py

Holds no model of its own: segment statistics and predictions come from the
API, and customer-level data comes from a mounted folder. Both locations are
read from environment variables so the same code works locally and in Compose.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

# In Compose this is http://api:8000 - the API's service name on the private
# network. Locally it falls back to the API running on your own machine.
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = os.getenv("DATA_PATH", "data/segmented_customers.csv")

st.set_page_config(page_title="Customer segments", layout="wide")
st.title("Customer segments")


@st.cache_data(ttl=60)
def load_segments() -> pd.DataFrame:
    response = requests.get(f"{API_URL}/segments", timeout=5)
    response.raise_for_status()
    return pd.DataFrame(response.json())


try:
    segments = load_segments()
except requests.RequestException as err:
    st.error(f"Cannot reach the API at {API_URL}: {err}")
    st.stop()

champions = segments.loc[segments["Monetary"].idxmax()]
st.markdown(
    f"**{champions['segment'].capitalize()}** are {champions['pct_customers']:.1f}% "
    f"of customers and {champions['pct_sales']:.1f}% of sales."
)

st.subheader("Share of customers vs share of sales")
st.bar_chart(
    segments.set_index("segment")[["pct_customers", "pct_sales"]],
    stack=False,
)

st.subheader("Segment profile")
st.dataframe(
    segments[["segment", "Customers", "Recency", "Frequency", "Monetary",
              "pct_customers", "pct_sales"]],
    hide_index=True,
)

st.subheader("Every customer")
if os.path.exists(DATA_PATH):
    customers = pd.read_csv(DATA_PATH)
    customers["segment"] = customers["Cluster"].map(
        dict(zip(segments["Cluster"], segments["segment"]))
    )
    # Spend runs from a few pounds to over GBP 250,000, so a linear axis
    # squashes nearly every customer into one corner.
    customers["log10 Monetary"] = np.log10(customers["Monetary"])
    st.scatter_chart(customers, x="Recency", y="log10 Monetary", color="segment")
else:
    st.info(f"Customer data not found at {DATA_PATH}.")

st.subheader("Score a customer")
with st.form("score"):
    recency = st.number_input("Days since last purchase", min_value=0, value=30)
    frequency = st.number_input("Number of orders", min_value=1, value=5)
    monetary = st.number_input("Total spend (GBP)", min_value=1.0, value=1200.0)
    submitted = st.form_submit_button("Find segment")

if submitted:
    response = requests.post(
        f"{API_URL}/predict",
        json={"recency": recency, "frequency": frequency, "monetary": monetary},
        timeout=5,
    )
    if response.ok:
        st.success(f"Segment: **{response.json()['segment']}**")
    else:
        st.error(f"The API rejected that input: {response.text}")
