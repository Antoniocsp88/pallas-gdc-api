# Pallas GDC API

FastAPI wrapper around NCI GDC endpoints for Pallas demos.

## Local dev
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python pallas_gdc_api.py   # or: uvicorn pallas_gdc_api:app --reload