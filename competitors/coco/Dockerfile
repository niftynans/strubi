FROM python:3.11-slim
WORKDIR /app
COPY . .
WORKDIR /app/experiments
RUN pip install -r requirements.txt 
CMD ["python", "examples.py"]