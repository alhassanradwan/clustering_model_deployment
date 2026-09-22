"""HTTP API for customer segmentation.

    uv run uvicorn api:app --app-dir src --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from serving import Segmenter

app = FastAPI(title="RFM Segmentation API")

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
    return {"status": "ok"}


@app.get("/segments")
def segments() -> list[dict[str, object]]:
    return segmenter.segments()


@app.post("/predict", response_model=Segment)
def predict(customer: Customer) -> Segment:
    cluster, segment = segmenter.predict(customer.recency, customer.frequency, customer.monetary)
    return Segment(cluster=cluster, segment=segment)
