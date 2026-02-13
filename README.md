# Sentiment Analysis Pipeline 🚀

A comprehensive end-to-end Machine Learning web application that predicts the sentiment (Positive/Negative) of movie reviews. This project demonstrates a full MLOps workflow, from data ingestion to cloud deployment on Google Cloud Platform (GCP).

## 📖 Project Overview
This application takes a text input (user review), processes it through a trained Natural Language Processing (NLP) model, and outputs the sentiment confidence score. 

**Key Features:**


* **End-to-End Pipeline:** Includes modular coding for Data Ingestion, Text Preprocessing, and Model Training.


* **Interactive UI:** Built with **Streamlit** for real-time user interaction.


* **Containerized:** Dockerized application for consistent performance across environments.


* **Cloud Native:** Fully deployed and scalable on **Google Cloud Run**.



## 🛠 Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Containerization:** Docker
* **Cloud Provider:** Google Cloud Platform (GCP)
* **Service:** Cloud Run (Serverless)

## 📊 Dataset
The model was trained on the **IMDB Movie Reviews Dataset**. It involves preprocessing steps such as tokenization, stop-word removal, and vectorization to prepare the textual data for the model.

## 💻 Installation & Local Usage

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone "Paste the HTTP url of Github repository"

2. **Create a virtual environment:**

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
**Install dependencies:**
pip install -r requirements.txt

**Run the Streamlit App:**
streamlit run app.py

☁️ Deployment on Google Cloud Platform
This project is deployed using Google Cloud Run, a serverless platform that automatically scales containers. Below is the process used for deployment:

1. **Containerization (Docker)**
A Dockerfile was created to package the application and its dependencies. This ensures the app runs the same way in the cloud as it does locally.

2. **Building the Image**
The Docker image was built and uploaded to the Google Container Registry (GCR) using Cloud Build.

# Command used to build and submit the image
gcloud builds submit --tag gcr.io/sentiment-analysis-485306/sentiment-analysis


3. Deploying to Cloud Run


The container was deployed to Cloud Run in the asia-south1 region. We configured it to allow unauthenticated access (publicly accessible) and allocated sufficient memory.


# Command used for deployment
gcloud run deploy sentiment-analysis \
  --image gcr.io/sentiment-analysis-485306/sentiment-analysis \
  --region asia-south1 \
  --port 8080 \
  --memory 2Gi \
  --allow-unauthenticated \

  
**Why Cloud Run?**


**Serverless:** No need to manage or provision servers.

**Auto-scaling:**Scales down to zero when not in use to save costs.

📂 Project Structure
├── .github/workflows   
├── src/                # Source code for pipeline modules
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   └── predict.py
├── app.py              # Entry point for Streamlit app
├── Dockerfile          # Configuration for Docker container
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation