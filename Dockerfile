FROM python:3.11-slim

WORKDIR /stock_predictor

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "stock_predictor.py", "--server.port=8501", "--server.address=0.0.0.0"]
