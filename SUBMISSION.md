# Submission

**Modal API URL:** https://awalla63--bank-marketing-propensity-api-fastapi-app.modal.run
**API /docs URL:** https://awalla63--bank-marketing-propensity-api-fastapi-app.modal.run/docs
**Vercel URL:** https://frontend-orcin-sigma-31.vercel.app
**Postman screenshots:** postman/screenshots/ (valid 200 + invalid 422)

## Write-up

This project estimates the probability that a bank telemarketing client will
subscribe to a term deposit, using the UCI/bankmarketing.csv dataset. Client and
contact fields (job, education, marital status, month, etc.) are encoded by a
custom `MeanTargetEncoder` transformer, which learns a smoothed per-category
subscription rate from the training data — that learned `mapping_` dict is the
"state" that makes the shipped `pipeline.joblib` meaningfully different from a
freshly-constructed, unfit pipeline. Those encoded features are combined with
`StandardScaler`-normalized macroeconomic/contact features and fed into a
`LogisticRegression` classifier (held-out ROC AUC ≈ 0.79) that outputs a
subscription probability rather than a bare label. Custom transformer:
`MeanTargetEncoder` (pipeline_def.py). scikit-learn version: 1.6.1.

## Repo contents

- `pipeline_def.py` — custom `MeanTargetEncoder` transformer
- `build_pipeline.py` — build/fit script that dumps `pipeline.joblib`
- `pipeline.joblib` — fitted bundle (pipeline + metadata)
- `serve.py` — FastAPI app (GET `/`, GET `/info`, POST `/predict`)
- `modal_serve.py` — Modal deployment definition
- `notebooks/model_development.ipynb` — interactive EDA / model development
- `frontend/` — static Vercel site calling the live Modal API
- `postman/bank_marketing_api.postman_collection.json` — Postman collection
- `data/bankmarketing.csv` — training data
