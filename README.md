# YOLO Custom Object Detection and Recognition

Welcome to the repository for **YOLO Custom Object Detection and Recognition**. This project is designed for license plate detection and character recognition using YOLOv10 and EasyOCR. It provides end-to-end implementation, from data preparation to model training and deployment.

## Dataset Preparation

The dataset used in this project focuses on traffic-related objects, particularly license plates. It consists of annotated images and videos for custom YOLO training.

1. **Source Dataset**: Place your raw dataset in the `Source_Dataset/` folder.
2. **Preprocessing**: Use `Data_Exploration.ipynb` to preprocess and explore the data.
3. **YOLO Formatting**: Use the scripts in `Yolo_custom_Dataset/` to convert annotations into YOLO format.
## Table of Contents

- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Detection and Recognition](#detection-and-recognition)
- [Presentation and Documentation](#presentation-and-documentation)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Folder Structure

```
.
├── .devcontainer/               # Development container configurations
├── Source_Dataset/              # Original dataset used for training
├── Training_Folder/             # Preprocessed dataset and training configurations
├── Yolo_custom_Dataset/         # YOLO-specific dataset format
├── uploads/                     # Folder for uploading and testing files
├── Data_Exploration.ipynb       # Jupyter notebook for data exploration and analysis
├── Yolo_Training.ipynb          # Jupyter notebook for training the YOLO model
├── detect_and_recognize.py      # Script for object detection and recognition
├── app.py                       # Flask app for deployment
├── requirements.txt             # Required Python packages
├── packages.txt                 # System-level dependencies for containerized environments
├── Team_4_Presentation.pptx     # Presentation slides
├── Team_4_Technical_Document.pdf # Technical documentation
├── traffic_video_detection_Compressed.mp4 # Example detection video
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- GPU with CUDA support (optional for faster training)
- Virtual environment (optional but recommended)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/YOLO-Custom-Detection.git
   cd YOLO-Custom-Detection
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install additional system dependencies:

   ```bash
   xargs -a packages.txt sudo apt-get install -y
   ```

4. Set up the environment (if using Docker):

   ```bash
   docker-compose up
   ```

## Dataset Preparation

The dataset used in this project focuses on traffic-related objects. It consists of annotated images and videos for custom YOLO training.

1. **Source Dataset**: Place your raw dataset in the `Source_Dataset/` folder.
2. **Preprocessing**: Use `Data_Exploration.ipynb` to preprocess and explore the data.
3. **YOLO Formatting**: Use the scripts in `Yolo_custom_Dataset/` to convert annotations into YOLO format.

## Model Training

1. Open `Yolo_Training.ipynb` and follow the steps to:
   - Configure the YOLO training environment.
   - Train the YOLO model using the dataset in `Training_Folder/`.
   - Save the trained model weights for inference.

2. Trained weights are stored in the appropriate directory for detection tasks.

## Detection and Recognition

1. Run `detect_and_recognize.py` to perform object detection and recognition on images or videos:

   ```bash
   python detect_and_recognize.py --input <input_file> --output <output_folder>
   ```

2. Deploy the model using the Flask app:

   ```bash
   python app.py
   ```

3. Access the web interface to upload files and view detection results.

## Presentation and Documentation

- **Technical Document**: Detailed explanation of the methodology, implementation, and results in `Team_4_Technical_Document.pdf`.
- **Presentation Slides**: Visual overview of the project in `Team_4_Presentation.pptx`.

## Demo

A sample video demonstrating object detection is included:

- `traffic_video_detection_Compressed.mp4`

Run the Flask app or the detection script to test with your own data.

## Contributing

We welcome contributions! Feel free to open issues or submit pull requests.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes and push them.
4. Open a pull request to the main branch.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Team Members


- Mohd Sharik
- Ayush Kumar
- Deepak Shinde

