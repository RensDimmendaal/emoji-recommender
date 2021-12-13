FROM tiangolo/uvicorn-gunicorn:python3.8-slim

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY get_model.py get_model.py
RUN python get_model.py

COPY ./app /app