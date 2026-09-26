"""HTTP API for customer segmentation.

    uv run uvicorn api:app --app-dir src --reload
"""

from __future__ import annotations

import os
import socket

from fastapi import FastAPI
from pydantic import BaseModel, Field

from serving import Segmenter

# Set at image build time, so the running code can report which version it is.
VERSION = os.getenv("APP_VERSION", "dev")

app = FastAPI(title="RFM Segmentation API", version=VERSION)

# Loaded once at startup, not per request: reading the model from disk on every
# call would dominate response time.
segmenter = Segmenter()


class Customer(BaseModel):
    recency: float = Field(ge=0, description="Days since the customer's last purchase")
    frequency: float = Field(ge=1, description="Number of orders placed")
    monetary: float = Field(gt=0, description="Total spend in GBP")


class Segment(BaseModel):
    cluster: int
    segment: str


@app.get("/health")
def health() -> dict[str, str]:
    # The container's hostname is its ID, so this identifies which replica
    # answered - useful for watching load balancing and rolling updates.
    return {"status": "ok", "version": VERSION, "replica": socket.gethostname()}


@app.get("/segments")
def segments() -> list[dict[str, object]]:
    return segmenter.segments()


@app.post("/predict", response_model=Segment)
def predict(customer: Customer) -> Segment:
    cluster, segment = segmenter.predict(customer.recency, customer.frequency, customer.monetary)
    return Segment(cluster=cluster, segment=segment)
