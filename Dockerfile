FROM python:3.11-slim

WORKDIR /stock_predictor

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
CMD ["streamlit", "run", "stock_predictor.py", "--server.port=8501", "--server.address=0.0.0.0"]
