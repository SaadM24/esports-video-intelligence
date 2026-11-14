
# eSports Video Intelligence

### Computer Vision-Based Player Detection, Tracking and Gameplay Analytics

A Python-based computer vision prototype for analysing Counter-Strike 2 (CS2) gameplay recordings.

The project investigates how object detection and multi-object tracking can transform unstructured gameplay footage into timestamped, structured analytics.

## Features

- Video ingestion and frame extraction using OpenCV
- Custom single-class player detection using YOLO11n
- Manual bounding-box annotation in YOLO format
- Training and comparison of five detector experiments
- Evaluation using precision, recall and mAP
- Multi-object tracking using ByteTrack
- Annotated gameplay video generation
- Frame-level and track-level CSV exports
- Player visibility, detection activity and track-duration analytics
- Matplotlib visualisations

## System Architecture

CS2 Gameplay Video
       |
       v
OpenCV Video Processing
       |
       v
YOLO11n Player Detection
       |
       v
ByteTrack Multi-Object Tracking
       |
       v
Timestamped Tracking Records
       |
       v
Pandas Analytics
       |
       v
CSV Reports and Visualisations

## Technologies

- Python
- PyTorch
- Ultralytics YOLO11n
- OpenCV
- ByteTrack
- NumPy and Pandas
- Matplotlib
- Scikit-learn

## Detection Experiments

Five YOLO11n experiments were trained using manually annotated gameplay frames.

The experiments explored the effects of different training examples, including distant players and negative background scenes.

### Validation Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| V1 | 0.8280 | 0.6667 | 0.7335 | 0.3011 |
| V5 | 0.7675 | 0.3333 | 0.4371 | 0.2025 |

Validation set: 17 images containing 12 annotated player instances.

### Diagnostic Evaluation

An additional 35-frame dataset was audited for frame overlap with the development datasets.

| Subset | Model | Precision | Recall | mAP50 |
|---|---|---:|---:|---:|
| Exact-overlap-free (32 images) | V1 | 0.2293 | 0.6705 | 0.2034 |
| Exact-overlap-free (32 images) | V5 | 0.3638 | 0.6250 | 0.3196 |
| Five-second-buffered (11 images) | V1 | 0.6303 | 0.6667 | 0.4433 |
| Five-second-buffered (11 images) | V5 | 0.9528 | 0.6667 | 0.6650 |

These are small, same-recording diagnostic subsets, not independent benchmarks. The additional frames were also examined during development, so these scores must not be interpreted as generalisation performance on new gameplay matches.

## Video Tracking Experiment

A 10-second segment of 1920x1080, 30-FPS CS2 footage was processed using YOLO11n and ByteTrack.

| Metric | V1 | V5 |
|---|---:|---:|
| Frames processed | 300 | 300 |
| Detection observations | 147 | 131 |
| Observations with tracking IDs | 133 | 121 |
| Distinct tracking IDs | 4 | 5 |
| Longest continuous track | 120 frames | 74 frames |

V1 maintained its longest continuous tracking ID for 4 seconds.

Tracking IDs represent algorithmic associations rather than verified player identities. Detection counts and tracking continuity are not equivalent to detection accuracy or identity-tracking accuracy.

## Project Structure

```text
src/
  ingestion/       Video loading and frame extraction
  preprocessing/   Contact sheet generation
  annotation/      Annotation and dataset validation
  detection/       YOLO detection and training
  evaluation/      Model comparison and test preparation
  tracking/        ByteTrack tracking and continuity analysis
  analytics/       CSV analytics and visualisation
```

## Local Setup

Developed on Windows with Python 3.11.9 and an NVIDIA GTX 1650 GPU.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install a compatible PyTorch and torchvision build following the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Data and Model Requirements

The repository does not redistribute the original gameplay recording, extracted video frames, training annotations or trained model checkpoints.

To reproduce the workflow, users must provide gameplay footage they are permitted to use, prepare a compatible YOLO-format dataset and train or supply a suitable player-detection checkpoint.

The existing scripts use relative paths within the project. Some paths and experiment names are configured directly in the scripts and may need adjusting for a new dataset or environment.

### Example Workflow

The following commands assume the required input files and model checkpoints have already been prepared:

```powershell
python src/ingestion/video_loader.py
python src/ingestion/frame_extractor.py
python src/annotation/check_yolo_dataset.py
python src/detection/train_player_detector.py
python src/tracking/player_tracker.py
python src/tracking/compare_tracking_runs.py
python src/analytics/gameplay_analytics.py
python src/analytics/visualize_gameplay.py
```

The tracking and analytics scripts use experiment-specific input and output paths. Edit their configuration sections when running a different experiment.

## Limitations

- Small, manually annotated dataset
- Training and evaluation drawn primarily from one gameplay recording
- Limited evidence of generalisation to different matches or maps
- Distant players and environmental backgrounds can cause detection errors
- Tracker IDs may fragment or duplicate the same player
- Player screen displacement does not directly measure in-game movement
- Currently processes recorded footage rather than a deployed live stream
- Kill-feed OCR, elimination recognition and other game-event extraction are not yet implemented

## Future Development

- Independent evaluation using a different gameplay recording
- Improving small-player detection
- Reducing false-positive detections and identity switches
- Gameplay event extraction using computer vision and OCR
- Latency and throughput benchmarking
- Interactive analytics dashboard

## Project Purpose

This prototype demonstrates practical experimentation with computer vision, machine learning, video processing and structured analytics in an eSports application.

It is a research and portfolio project, not a production-ready gameplay statistics system.
