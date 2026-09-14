"""Modal deployment for the bank-marketing subscription-propensity API.

Ships exactly three application files into the image: serve.py, pipeline_def.py,
and pipeline.joblib. scikit-learn is pinned to the EXACT version recorded in the
artifact's metadata (see build_pipeline.py's metadata["sklearn_version"]) so the
container that unpickles the bundle matches the version that pickled it.

Deploy:
    modal deploy modal_serve.py
"""

import modal

# Keep this pinned to metadata["sklearn_version"] in pipeline.joblib. Mismatches
# between the fitting environment and the serving environment are a classic way
# to silently corrupt a fitted pipeline on unpickle.
SKLEARN_VERSION = "1.6.1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.141.1",
        f"scikit-learn=={SKLEARN_VERSION}",
        "pandas==2.2.3",
        "joblib==1.4.2",
        "pydantic>=2,<3",
    )
    .add_local_file("serve.py", "/root/serve.py")
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py")
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib")
)

app = modal.App("bank-marketing-propensity-api", image=image)


@app.function(min_containers=0, scaledown_window=120)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    # Import inside the function so it runs in the Modal container (where
    # serve.py/pipeline_def.py/pipeline.joblib actually live on disk at
    # /root), not in whatever environment is executing `modal deploy` locally.
    import sys

    sys.path.insert(0, "/root")
    from serve import app as web_app

    return web_app
