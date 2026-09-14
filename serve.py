"""FastAPI service for the bank-marketing subscription-propensity pipeline.

Loads the fitted `.joblib` bundle ONCE at import time (module load), not per
request. If the artifact is missing or fails to unpickle, the app still starts
but every endpoint that needs the model returns 503 (not 500) until it's fixed.

Run locally:
    uvicorn serve:app --reload
    -> http://localhost:8000/docs
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Must be importable so joblib can unpickle the fitted MeanTargetEncoder inside
# the pipeline's ColumnTransformer.
from pipeline_def import MeanTargetEncoder  # noqa: F401

ARTIFACT_PATH = Path(__file__).parent / "pipeline.joblib"

# ---------------------------------------------------------------------------
# Load the artifact once at import time.
# ---------------------------------------------------------------------------
_bundle: dict | None = None
_load_error: str | None = None

try:
    _bundle = joblib.load(ARTIFACT_PATH)
except Exception as exc:  # noqa: BLE001 - we want to serve 503s for any load failure
    _load_error = f"{type(exc).__name__}: {exc}"

app = FastAPI(
    title="Bank Marketing Subscription-Propensity API",
    description=(
        "Serves a fitted scikit-learn Pipeline (custom MeanTargetEncoder + "
        "StandardScaler -> LogisticRegression) that estimates the probability "
        "a bank telemarketing contact subscribes to a term deposit."
    ),
    version="1.0.0",
)

# The frontend is a static site on a different origin (Vercel) than the API
# (Modal), so the browser needs an explicit CORS allowance. This is a public
# read-mostly demo API with no auth/cookies, so allow_origins="*" is fine here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_bundle() -> dict:
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact unavailable: {_load_error or 'unknown load failure'}",
        )
    return _bundle


# ---------------------------------------------------------------------------
# Request / response schemas with explicit bounds -> bad input becomes 422.
# ---------------------------------------------------------------------------

JobType = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MaritalType = Literal["divorced", "married", "single", "unknown"]
EducationType = Literal[
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]
YesNoUnknown = Literal["yes", "no", "unknown"]
ContactType = Literal["cellular", "telephone"]
MonthType = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeekType = Literal["mon", "tue", "wed", "thu", "fri"]
PoutcomeType = Literal["failure", "nonexistent", "success"]


class PredictRequest(BaseModel):
    age: int = Field(..., ge=17, le=100, description="Client age in years")
    job: JobType
    marital: MaritalType
    education: EducationType
    default: YesNoUnknown = Field(..., description="Has credit in default?")
    housing: YesNoUnknown = Field(..., description="Has a housing loan?")
    loan: YesNoUnknown = Field(..., description="Has a personal loan?")
    contact: ContactType
    month: MonthType = Field(..., description="Last contact month")
    day_of_week: DayOfWeekType = Field(..., description="Last contact day of week")
    campaign: int = Field(..., ge=1, le=60, description="Contacts made during this campaign, including this one")
    pdays: int = Field(..., ge=0, le=999, description="Days since last contact from a previous campaign (999 = not previously contacted)")
    previous: int = Field(..., ge=0, le=20, description="Number of contacts before this campaign")
    poutcome: PoutcomeType = Field(..., description="Outcome of the previous marketing campaign")
    emp_var_rate: float = Field(..., ge=-5.0, le=5.0, description="Employment variation rate (quarterly indicator)")
    cons_price_idx: float = Field(..., ge=85.0, le=105.0, description="Consumer price index (monthly indicator)")
    cons_conf_idx: float = Field(..., ge=-60.0, le=-10.0, description="Consumer confidence index (monthly indicator)")
    euribor3m: float = Field(..., ge=0.0, le=10.0, description="Euribor 3 month rate")
    nr_employed: float = Field(..., ge=4500.0, le=5500.0, description="Number of employees (quarterly indicator)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 41,
                "job": "technician",
                "marital": "married",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "thu",
                "campaign": 2,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0,
            }
        }
    }


class PredictResponse(BaseModel):
    subscribe_probability: float
    predicted_label: Literal["yes", "no"]
    baseline_training_rate: float


class InfoResponse(BaseModel):
    status: str
    metadata: dict
    cat_columns: list[str]
    num_columns: list[str]
    target_rate: float


class HealthResponse(BaseModel):
    status: str
    artifact_loaded: bool
    checked_at: str


# Map Pydantic snake_case field names to the raw dataset's dotted column names.
_FIELD_TO_COLUMN = {
    "emp_var_rate": "emp.var.rate",
    "cons_price_idx": "cons.price.idx",
    "cons_conf_idx": "cons.conf.idx",
    "nr_employed": "nr.employed",
}


@app.get("/", response_model=HealthResponse)
def health() -> HealthResponse:
    """Basic liveness/health check. Does not 503 - reports artifact status."""
    return HealthResponse(
        status="ok" if _bundle is not None else "degraded",
        artifact_loaded=_bundle is not None,
        checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Describes the loaded pipeline artifact. 503 if it isn't loaded."""
    bundle = _require_bundle()
    return InfoResponse(
        status="ok",
        metadata=bundle["metadata"],
        cat_columns=bundle["cat_columns"],
        num_columns=bundle["num_columns"],
        target_rate=bundle["target_rate"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Runs one client record through the fitted pipeline. 503 if the artifact
    isn't loaded; 422 automatically for any field violating its bounds/enum."""
    bundle = _require_bundle()
    pipeline = bundle["pipeline"]

    row = request.model_dump()
    record = {
        _FIELD_TO_COLUMN.get(k, k): v for k, v in row.items()
    }
    X = pd.DataFrame([record])[bundle["cat_columns"] + bundle["num_columns"]]

    try:
        proba = float(pipeline.predict_proba(X)[0, 1])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Pipeline inference failed: {exc}") from exc

    return PredictResponse(
        subscribe_probability=proba,
        predicted_label="yes" if proba >= 0.5 else "no",
        baseline_training_rate=bundle["target_rate"],
    )
