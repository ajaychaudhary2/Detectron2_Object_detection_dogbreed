# Dog Breed Detection using Detectron2

This project is a web application for detecting dog breeds using the Detectron2 object detection framework.

## Project Structure

```
Detectron2_ObjectDetection/
├── config/
│   ├── config.yaml
├── data/
│   ├── annotations/
│   ├── images/
├── env/  
├── model/
│   ├── model_final.pth
├── static/uploads/
│   ├── golden-retriever.jpg
│   ├── images_1.jpg
│   ├── images_2.jpg
│   ├── result.jpg
├── templates/
│   ├── index.html
├── app.py
├── prediction.py
├── rech.ipynb
├── requirements.txt
├── README.md
```

## Features
- Upload an image to detect dog breeds.
- Uses Detectron2 for object detection.
- Web interface built with Flask.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/detectron2-dog-breed.git
cd detectron2-dog-breed
```

### 2. Set Up a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Flask Application
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000/`.

### 2. Upload an Image
- Open the web page.
- Upload an image.
- View the predicted dog breed and confidence score.

## Model Details
- Pretrained on a dataset of dog breeds.
- Fine-tuned using Detectron2.

## Author
Ajay Chaudhary
ajaych2822@gmail.com 


