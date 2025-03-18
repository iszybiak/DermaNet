FROM python:3.12
LABEL authors="Izabelka"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src /app/src
COPY model /app/model/
CMD ["python", "src/api.py"]
