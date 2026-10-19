# BrainScan

Brain MRI classification application powered by a Convolutional Neural Network (CNN).

Upload a brain MRI image and the model classifies it into one of four categories:

| Class | Description |
|---|---|
| **glioma** | Glioma tumor |
| **meningioma** | Meningioma tumor |
| **notumor** | No tumor detected |
| **pituitary** | Pituitary tumor |

> **⚠️ Disclaimer:** This model is experimental and has **not** been clinically validated.
> It is intended for educational purposes only and must **not** be used as a substitute
> for professional medical diagnosis.

## Project Structure

```
brief-4/
├── Dockerfile
├── .dockerignore
├── compose.yaml
├── requirements.txt
├── README.md
├── app.py                  # Streamlit web interface
├── predict.py              # Model loading and prediction logic
├── models/
│   └── brainscan_best.keras  # Trained CNN model
└── notebooks/
    └── BrainScan.ipynb       # Training and evaluation notebook
```

## Run with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running.
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop).

### Build and Start

```bash
docker compose up --build -d
```

The application will be available at: **http://localhost:8501**

### View Logs

```bash
docker compose logs -f brainscan
```

### Stop the Application

```bash
docker compose down
```

## Troubleshooting

### Model loading errors

If you see a `FileNotFoundError` related to the model:

- Verify that `models/brainscan_best.keras` exists in the project directory.
- Ensure `.dockerignore` does **not** exclude the `models/` directory.
- Rebuild the image: `docker compose up --build -d`.

### Dependency or compatibility errors

If TensorFlow or Keras fails to import:

- Check that `requirements.txt` uses versions compatible with the trained model.
- The model was trained with Python 3.14, TensorFlow 2.22.0rc0, and keras-nightly 3.16.0.dev2026101404.
- Do **not** upgrade these packages without verifying model compatibility.

### Port conflicts

If port 8501 is already in use:

- Stop the other process using the port, or
- Change the host port mapping in `compose.yaml`:
  ```yaml
  ports:
    - "8502:8501"  # Use host port 8502 instead
  ```
- Then access the application at `http://localhost:8502`.

### Docker permissions (Linux)

If you get a "permission denied" error:

- Add your user to the `docker` group:
  ```bash
  sudo usermod -aG docker $USER
  ```
- Log out and back in for the change to take effect.
- Alternatively, run commands with `sudo`.

## Local Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
