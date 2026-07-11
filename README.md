# Energy Sector Stock Predictor 📈🛢️

A machine learning web application that predicts stock price trends in the energy and oil sector. Built with Python and Streamlit, this application provides an interactive interface for data analysis and utilizes The Guardian API to fetch relevant market news.

The project is fully containerized using Docker, making it easy to deploy and run across any environment without the need for manual dependency management.

## Features
* **Interactive UI:** A clean, responsive dashboard built with Streamlit.
* **Machine Learning Integration:** Uses historical market data to predict future trends in the energy sector.
* **News Sentiment Context:** Integrates with The Guardian API to provide relevant(Oil sector) articles and context for market movements.
* **Dockerized:** Simple and reproducible setup using Docker.

## Tech Stack
* **Language:** Python 3.11
* **Framework:** Streamlit
* **Data Processing & ML:** Pandas, Scikit-Learn
* **Infrastructure:** Docker

## Prerequisites
Before running this project, ensure you have the following:
* [Docker](https://docs.docker.com/get-docker/) installed on your machine.
* A free API key from [The Guardian Open Platform](https://open-platform.theguardian.com/).

## Setup and Installation

**1. Clone the repository:**
```bash
git clone https://github.com/dalishev/stock_predictor.git
cd energy-stock-predictor
```

**2. Build the docker image:**

**Normal:**
```bash
docker build -t stock_predictor .
```

**On linux:**
```bash
docker build --platform=linux/arm64 -t stock_predictor .
```

**3. Run the container:**
Start the application by passing your Guardian API key as an environment variable.

```bash
docker run -p 8501:8501 -e GUARDIAN_API_KEY="your_api_key_here" stock_predictor
```

**4. Open your browser and go to this address:**
```bash
localhost:8501
```
