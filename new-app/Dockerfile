FROM python:3.8-slim-bullseye

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY get_model.py get_model.py
RUN python get_model.py

COPY ./app /app

ENV PORT 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]