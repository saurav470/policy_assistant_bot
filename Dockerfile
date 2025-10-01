# Build stage:
FROM python:3.12

WORKDIR /app

COPY requirements_new.txt requirements_new.txt

RUN pip install -r requirements_new.txt
# RUN pip install docx
COPY . .


EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --reload