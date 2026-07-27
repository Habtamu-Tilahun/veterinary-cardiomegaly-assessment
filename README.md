# AI-based Veterinary Cardiomegaly Assessment

A Flask-based web application for automatic cardiomegaly assessment from veterinary thoracic radiographs using deep learning. The application estimates two commonly used cardiac biomarkers:

* **Cardiothoracic Ratio (CTR)** from ventrodorsal (VD) radiographs
* **Buchanan Index (BI)** from lateral (LM) radiographs

Users can upload an X-ray image through a simple web interface, and the application automatically performs image segmentation, anatomical landmark detection, biomarker estimation, and visualization of the results.

---

## Features

* Automatic heart, thorax, T4 vertebra, and carina segmentation
* Automatic computation of:

  * Cardiothoracic Ratio (CTR)
  * Buchanan Index (BI)
* Visualization of predicted contours and anatomical measurements
* Flask web interface
* Modular software architecture
* Docker support
* Detectron2 Mask R-CNN inference

---

## Project Structure

```
.
├── app.py
├── config.py
├── predictor.py
├── services/
├── utils/
├── models/
├── static/
├── templates/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Technology Stack

* Python
* Flask
* Detectron2
* PyTorch
* OpenCV
* NumPy
* SciPy
* Docker

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd cardiomegaly-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```
http://localhost:5000
```

---

## Running with Docker

Build the Docker image:

```bash
docker-compose build
```

Start the application:

```bash
docker-compose up
```

Open:

```
http://localhost:5000
```

---

## Models

The application uses two Detectron2 Mask R-CNN models:

### Buchanan Index

Segments:

* Heart
* T4 vertebra
* Carina

Outputs:

* Cardiac long axis
* Cardiac short axis
* T4 size
* Buchanan Index
* Diagnostic finding

---

### Cardiothoracic Ratio

Segments:

* Heart
* Thorax

Outputs:

* Heart size
* Thorax size
* Cardiothoracic Ratio
* Diagnostic finding

---

## Example Workflow

1. Upload a thoracic radiograph.
2. The image is processed by the appropriate Detectron2 model.
3. Anatomical structures are segmented.
4. Cardiac measurements are computed automatically.
5. The application displays the estimated biomarker values and an annotated image.

---

## Future Improvements

* REST API
* Batch inference
* DICOM support
* Confidence scores
* GPU/CPU automatic selection
* Unit testing
* Continuous Integration (CI)
* Cloud deployment

---

## License

This project is intended for research and educational purposes.

---

## Author

Habtamu Mekonnen

Medical Image Analysis | Computer Vision | Deep Learning
