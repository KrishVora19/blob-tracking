# blob-tracking

A lightweight Python project for detecting and tracking "blobs" (moving objects) in video streams or image sequences. This repository contains code to detect foreground blobs, compute bounding boxes and centroids, and optionally record and visualize trajectories over time. I have designed this project to be used as a unique video effect.

## Features
- Detect moving objects (blobs) in video or webcam streams.
- Filter blobs by size and shape to reduce noise.
- Track blobs across frames using simple centroid matching (and hooks for more advanced trackers).
- Output overlays on video and optional CSV logs for trajectories.
- CLI options to run on webcam, video files, or image sequences.

## How the tracker works (high-level)
1. Read frames from the chosen source (webcam, video file, or folder of images).
2. Convert frames to grayscale and apply smoothing to reduce noise.
3. Use a foreground detection method (frame differencing or background subtraction) to build a binary mask of moving areas.
4. Find contours on the binary mask and compute bounding boxes and centroids for each contour.
5. Filter contours using configurable thresholds (minimum area, maximum area, aspect ratio, etc.).
6. Assign detected centroids to existing tracked objects using nearest-centroid matching (with a configurable maximum distance). New blobs become new tracks; lost blobs can be forgotten after a timeout.
7. Draw debug overlays (bounding boxes, IDs, trails) on frames and optionally write output video and CSV logs.

## Prerequisites

- Python 3.8+
- Recommended virtual environment (venv, conda, etc.)

Suggested Python libraries (install below):
- opencv-python
- numpy
- pandas (optional, for CSV logging)
- matplotlib (optional, for plotting)
