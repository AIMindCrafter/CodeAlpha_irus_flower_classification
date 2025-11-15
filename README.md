<img width="1202" height="630" alt="Irus_clasifier" src="https://github.com/user-attachments/assets/522dfbf5-4d10-4379-b4a5-3ad7846d5205" /># Iris Flask Deployment

This small Flask app serves the trained Iris classifier saved from the notebook.

Files added:
- `app.py` — Flask application that loads `final_iris_classifier_model.pkl` and `scaler_used_for_model.pkl` and exposes `/` (form) and `/predict` (POST) endpoints.
- `templates/index.html` — Simple UI to send measurements and get predictions.
- `requirements.txt` — Minimal Python dependencies.

Preconditions
- Make sure the files `final_iris_classifier_model.pkl` and `scaler_used_for_model.pkl` produced by the notebook are present in the project root (same folder as `app.py`).

Install and run (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the flask app
python app.py
```

The app will listen on `http://0.0.0.0:5000/`. Open that address in your browser and use the form.

API usage (JSON):

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

GitHub / MLOps deployment notes
--------------------------------

- Build and publish a container automatically to GitHub Container Registry (GHCR) on merge to `main`. See `.github/workflows/docker-publish.yml`.
- Run unit tests on push and PRs via `.github/workflows/ci.yml`.

Setup for CI and registry publishing
-----------------------------------
- No extra secrets are needed to push to GHCR when using `GITHUB_TOKEN` for the same repository; ensure `permissions: packages: write` are enabled for workflows.
- If you prefer pushing to Docker Hub or another registry, update `.github/workflows/docker-publish.yml` and set the credentials in repo Secrets.

Model artifact handling
-----------------------
- This repo stores model artifacts (`final_iris_classifier_model.pkl` and `scaler_used_for_model.pkl`) in the repo root for convenience. For production-grade workflows, consider storing models in:
  - An artifact store (S3 / GCS) with versioning
  - MLflow Model Registry
  - GitHub Releases or an artifact repository

Next steps / suggestions
------------------------
- Add tests that validate model metrics automatically during CI and fail if performance drops.
- Add a deployment workflow that deploys the built container to your cloud provider (e.g., AWS ECS, GKE, Azure Container Instances) or to a VM.
- Integrate MLflow tracking server and register models to the MLflow Model Registry for controlled promotion between dev/staging/prod.



<img width="1202" height="630" alt="Irus_clasifier" src="https://github.com/user-attachments/assets/f9146825-d565-420c-ba71-ac1b5e1d8018" />



