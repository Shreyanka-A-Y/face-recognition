# Face Recognition System using FaceNet

A real-time face recognition system built with Python, OpenCV, and DeepFace library using the FaceNet model. This system can capture faces, train a dataset, and perform real-time face recognition with additional features like age, gender, and emotion detection.

## Features

- **Face Dataset Creation**: Capture and store face images for training
- **Face Training**: Train the model using FaceNet embeddings
- **Real-time Recognition**: Recognize faces in real-time with confidence scores
- **Additional Analysis**: Detect age, gender, and emotion
- **User-friendly Interface**: Simple command-line interface

## Requirements

### Dependencies

```bash
pip install opencv-python
pip install deepface
pip install numpy
```

### Hardware Requirements

- Webcam or camera device
- Python 3.7 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shreyanka-A-Y/face-recognition.git
cd face-recognition
```

2. Install required packages:
```bash
pip install opencv-python deepface numpy
```

3. Run the application:
```bash
python FaceRecognitation.py
```

## Usage

The application provides three main functionalities:

### 1. Create Face Dataset
- Select option `1` from the menu
- Enter the person's name
- Look at the camera and press `q` to stop capturing (captures up to 50 images)
- Images are automatically saved in the `Dataset/{person_name}/` directory

### 2. Train Face Dataset
- Select option `2` from the menu
- The system will process all images in the Dataset directory
- Creates face embeddings using the FaceNet model
- Saves the trained embeddings to `embedding.npy`

### 3. Recognize Faces
- Select option `3` from the menu
- The system will start real-time face recognition
- Shows recognized person's name with confidence score
- Displays additional information: age, gender, and emotion
- Press `q` to quit recognition mode

## Project Structure

```
face-recognition/
├── FaceRecognitation.py          # Main application file
├── FaceRecognitation copy.py     # Backup/alternative version
├── embedding.npy                 # Trained face embeddings (generated after training)
├── haarcascade_frontalface_default.xml  # Face detection classifier
├── tempCodeRunnerFile.py         # Temporary file
├── Dataset/                      # Directory for storing face images
│   └── {person_name}/           # Individual person directories
│       ├── {person_name}_1.jpg
│       ├── {person_name}_2.jpg
│       └── ...
└── README.md                     # This file
```

## How It Works

1. **Face Detection**: Uses Haar Cascade classifier for detecting faces in video frames
2. **Face Embedding**: Utilizes FaceNet model through DeepFace library to create 128-dimensional face embeddings
3. **Recognition**: Compares new face embeddings with stored embeddings using cosine similarity
4. **Threshold**: Recognition confidence threshold is set to 0.7 (70%)
5. **Additional Analysis**: Uses DeepFace for age, gender, and emotion detection

## Technical Details

- **Model**: FaceNet (via DeepFace library)
- **Face Detection**: Haar Cascade Classifier
- **Similarity Metric**: Cosine Similarity
- **Recognition Threshold**: 0.7
- **Image Format**: JPG
- **Video Capture**: OpenCV VideoCapture

## Troubleshooting

### Common Issues

1. **Camera not working**: 
   - Check if camera is connected and not being used by another application
   - Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **No embedding found error**:
   - Make sure you have created a dataset (option 1) and trained it (option 2) before recognition

3. **Poor recognition accuracy**:
   - Ensure good lighting conditions during dataset creation
   - Capture images from different angles
   - Increase the number of training images per person

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Shreyanka A Y**
- GitHub: [@Shreyanka-A-Y](https://github.com/Shreyanka-A-Y)

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) library for face recognition
- [OpenCV](https://opencv.org/) for computer vision operations
- [FaceNet](https://arxiv.org/abs/1503.03832) paper for the face recognition model
