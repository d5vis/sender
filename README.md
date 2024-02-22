# Sender

Tracking and analyzing your sends

## Prerequisites

- Python 3.x

## Installation

Clone the repository:

```bash
git clone https://github.com/d5vis/sender.git
```

Install the required packages:

```bash
pip3 install -r requirements.txt
```

## Running Sender

To run sender on a local file:

```bash
python3 sender.py YOUR_FILE_NAME.mov
```

To run sender on a live camera:

```bash
python3 sender.py
```

Sender will first ask you to mask out the holds using HSV sliders. Pressing `q` confirms your selection and starts the tracking.

Press `q` while sender is running to quit.

## Documentation

- [Mediapipe Landmarks/Poses](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
