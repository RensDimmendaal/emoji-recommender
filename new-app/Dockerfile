FROM python:3.8-slim-bullseye

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY get_model.py get_model.py
RUN python get_model.py

COPY ./app /app

COPY gunicorn_config.py .
CMD gunicorn app.main:app -c gunicorn_config.py