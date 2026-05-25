# Task 1: UFO Local ML Web App

This repository completes the Task 1 local deployment workflow for the UFO project.

It trains a machine learning model from `data/ufos.csv`, exposes the model through a FastAPI backend, and provides a Streamlit frontend that calls the backend on localhost.

## Architecture

```text
Streamlit frontend -> FastAPI backend -> saved scikit-learn model -> prediction result
```

## Project Structure

```text
.
├── data/
│   └── ufos.csv
├── model/
│   └── train_model.py
├── backend/
│   ├── __init__.py
│   └── main.py
├── frontend/
│   └── app.py
├── scripts/
│   └── smoke_test.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Train The Model

```bash
python model/train_model.py
```

This creates:

```text
model/ufo_model_bundle.pkl
```

The bundle contains:

- trained logistic regression model
- country label encoder
- selected feature names
- model accuracy

## 2. Run The FastAPI Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Open the API docs:

```text
http://localhost:8000/docs
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"seconds": 50, "latitude": 44, "longitude": -12}'
```

## 3. Run The Streamlit Frontend

In a second terminal:

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

Enter seconds, latitude, and longitude, then click predict. The frontend sends the input to FastAPI and displays the predicted UFO sighting country.

## Optional Smoke Test

After training the model:

```bash
python scripts/smoke_test.py
```

This checks that the saved model can be loaded and can produce a prediction locally.

## GitHub Actions

The workflow in `.github/workflows/task1-local-deployment.yml` lets GitHub run the core Task 1 check automatically:

1. install dependencies
2. train the UFO model
3. run the local model smoke test
4. call the FastAPI `/predict` endpoint with a sample payload

The generated model file is intentionally ignored by git. It is rebuilt in CI and can also be rebuilt locally with `python model/train_model.py`.
