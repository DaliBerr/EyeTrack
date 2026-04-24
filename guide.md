# EyeTrack Technical Report

## Based on the `QAT` Branch

## 1. Introduction

EyeTrack is a computer vision project designed to estimate where a person is looking by analyzing images of the eye. In simple terms, the system first learns how to recognize important parts of the eye, then measures their geometry, and finally converts those measurements into a gaze direction or a gaze point.

This repository is not only a machine learning training project. It is a complete technical pipeline that includes:

* dataset preparation,
* neural network training,
* model quantization,
* prediction on eye images,
* extraction of iris and pupil geometry,
* sequence analysis and smoothing,
* a desktop real-time demo,
* and a Raspberry Pi real-time deployment pipeline.

The `QAT` branch is the correct branch to study because it contains the newer and more structured implementation. Compared with an older experimental branch, this branch organizes the code into reusable modules and introduces a clearer deployment workflow.

This report is written for non-technical readers. The goal is not only to describe what files exist, but to explain what the system does, why each part exists, and how the full real-time pipeline works.

---

## 2. Main Goal of the Project

The main purpose of EyeTrack is to estimate gaze from eye images.

The system works in three major stages:

1. **Eye segmentation**
   A neural network analyzes an eye image and classifies each pixel into one of several classes, such as background, outer eye boundary, iris, and pupil.

2. **Geometric feature extraction**
   After segmentation, the software measures shapes and positions, especially the center and ellipse of the iris and pupil.

3. **Gaze estimation and mapping**
   The measured eye features are transformed into a direction feature, and after a calibration step, this feature is mapped to a screen point.

The project supports both offline analysis and real-time use:

* **Offline mode** is used for training, evaluation, quantization, and sequence analysis.
* **Real-time mode** is used to process camera frames live on a PC or on a Raspberry Pi system.

---

## 3. High-Level System Overview

At a high level, the repository can be understood as four connected subsystems.

### 3.1 Training subsystem

This part trains a lightweight U-Net model on an OpenEDS-style dataset.

### 3.2 Geometry subsystem

This part converts segmentation maps into meaningful eye measurements such as pupil center, iris center, ellipse axes, normalized offsets, and area ratios.

### 3.3 Deployment subsystem

This part exports trained models to ONNX, quantizes them, evaluates their quality, and benchmarks runtime latency.

### 3.4 Real-time subsystem

This part runs the model live on camera frames. It exists in two versions:

* a **desktop PC real-time demo**,
* and a **Raspberry Pi dual-camera real-time pipeline**.

The Raspberry Pi version is the most important deployment path in this repository.

---

## 4. Main Tools, Frameworks, and Libraries

This project relies on a combination of machine learning libraries, image processing libraries, data analysis tools, and device-side multimedia tools.

### 4.1 Core Python and ML stack

#### PyTorch

PyTorch is used to define, train, and fine-tune the neural network model. It provides:

* tensor computation,
* model definition,
* training with backpropagation,
* checkpoint saving and loading,
* and Quantization-Aware Training preparation.

#### ONNX

ONNX is used as a neutral model format for deployment. A trained PyTorch model can be exported to ONNX so that it can be used in a more portable runtime environment.

#### ONNX Runtime

ONNX Runtime is used for deployment inference and quantization. It is especially important in this project because the Raspberry Pi deployment does not run the PyTorch model directly. Instead, it runs a quantized ONNX model with ONNX Runtime.

### 4.2 Computer vision and image processing

#### OpenCV

OpenCV is used for:

* image resizing,
* grayscale conversion,
* contour extraction,
* ellipse fitting,
* morphology operations,
* visualization,
* and camera access on desktop.

It is also used to draw live overlays such as ellipses, calibration targets, arrows, and screen preview panels.

#### Pillow

Pillow is used mainly for reading image files in the dataset and inference scripts.

### 4.3 Data analysis and visualization

#### NumPy

NumPy is the numerical base of the project. It is used for image arrays, geometry calculations, interpolation, signal smoothing, and calibration math.

#### Pandas

Pandas is used for sequence-level CSV processing and filtering workflows.

#### Matplotlib

Matplotlib is used to generate analysis plots showing raw versus cleaned signals across time.

### 4.4 Utility and workflow tools

#### tqdm

Used for progress bars during training and validation.

#### argparse

Used in almost all scripts to define command-line interfaces.

### 4.5 Raspberry Pi specific tools

#### Picamera2

Picamera2 is used on Raspberry Pi to access CSI cameras in a modern, synchronized, metadata-aware way.

#### libcamera

Used together with Picamera2 for camera configuration and frame metadata.

#### GStreamer

GStreamer is used on Raspberry Pi for two output tasks:

* RTSP video streaming,
* local H.264 video recording.

#### python3-gi / GstRtspServer

Used to build the RTSP server process on Raspberry Pi.

### 4.6 Why this tool combination matters

This is an important design choice of the project:

* **PyTorch** is best for training.
* **ONNX Runtime** is better for lightweight deployment.
* **OpenCV** handles real-time image processing and visualization.
* **Picamera2 + GStreamer** make the Raspberry Pi pipeline practical for live camera use.

So the repository is not only a model repository. It is a bridge between machine learning development and real embedded deployment.

---

## 5. Repository Organization

The repository is structured around a central Python package named `eyetrack`, while many top-level scripts act as command-line entry points.

### 5.1 Top-level entry scripts

Important top-level scripts include:

* `Train.py` — training entry point,
* `predict_segmentation_to_npy.py` — image prediction,
* `extract_geometry_from_segmentation.py` — geometry extraction,
* `batch_predict_and_extract_sequences.py` — batch sequence processing,
* `baseline_filter_sequences_v3.py` — sequence cleaning and smoothing,
* `analyze_sequence_geometry.py` — visualization and anomaly analysis,
* `export_unet_to_onnx.py` — ONNX export,
* `quantize_onnx_model.py` — static ONNX quantization,
* `evaluate_onnx_model.py` — ONNX evaluation,
* `benchmark_onnx_model.py` — latency benchmark,
* `realtime_eye_direction.py` — desktop real-time demo,
* `realtime_eye_direction_pi.py` — Raspberry Pi real-time system.

### 5.2 Internal package layout

The `eyetrack` package is the real core of the project. It contains:

* `config.py` — global defaults and model metadata helpers,
* `data/` — dataset loading and preprocessing,
* `models/` — the U-Net model,
* `training/` — checkpoints, training engine, QAT utilities,
* `metrics/` — segmentation metrics,
* `deployment/` — ONNX-related helpers,
* `gaze.py` — geometry extraction and calibration mathematics,
* `realtime_gaze.py` — calibration and preview drawing helpers,
* `raspi/` — Raspberry Pi runtime modules.

This layout is much cleaner than a single-file prototype. It shows that the `QAT` branch is designed as a maintainable system.

---

## 6. Model and Learning Pipeline

### 6.1 Model architecture

The model used in this repository is a lightweight **U-Net**.

U-Net is a popular architecture for semantic segmentation because it combines:

* a **downsampling path** that captures context,
* and an **upsampling path** that recovers spatial detail.

In this implementation, the model is intentionally lightweight. Its width can be changed through the `base_channels` parameter, allowing smaller models such as `b8` or `b6` for mobile and embedded deployment.

### 6.2 Input and output

The model expects a grayscale eye image and outputs per-pixel class logits.

The class convention used in the project is:

* class 0: background,
* class 1: outer boundary or outer eye region,
* class 2: iris,
* class 3: pupil.

### 6.3 Dataset format

Training uses an OpenEDS-style directory structure:

* `train/images`, `train/labels`, `train/masks`
* `validation/images`, `validation/labels`, `validation/masks`

This is important because many scripts assume this structure.

### 6.4 Training process

During training, each image is:

1. loaded as grayscale,
2. normalized,
3. resized to the model input size,
4. paired with its label map,
5. optionally paired with a validity mask.

The training loop computes:

* cross-entropy loss,
* pixel accuracy,
* masked pixel accuracy,
* Dice score for foreground classes.

### 6.5 Checkpoints and metadata

A strong feature of the `QAT` branch is that checkpoints do not only store network weights. They also store metadata such as:

* number of input channels,
* number of output classes,
* base channel width,
* input width and height,
* preprocessing mode,
* AMP setting,
* mask usage,
* quantization mode,
* QAT backend.

This matters because later scripts can automatically recover the correct model configuration from the checkpoint itself, instead of requiring the user to manually re-enter all settings.

---

## 7. Quantization and Deployment Strategy

Quantization is a major focus of this branch.

### 7.1 Why quantization is needed

For embedded deployment, especially on Raspberry Pi, a full floating-point model can be too heavy. Quantization reduces model size and can improve practical CPU inference speed.

### 7.2 PTQ and QAT in this repository

The project supports two related ideas:

* **PTQ (Post-Training Quantization)** — quantize an already trained model,
* **QAT (Quantization-Aware Training)** — continue training while simulating quantization effects.

### 7.3 Important practical note

Although the branch is named `QAT`, the repository does not treat direct export of an eager QAT graph to ONNX QDQ as the main stable deployment path.

Instead, the recommended deployment route is:

1. train or fine-tune a PyTorch model,
2. if desired, run QAT fine-tuning,
3. strip the prepared QAT model back to a fused floating model,
4. export a floating ONNX model,
5. apply ONNX Runtime static quantization,
6. deploy the resulting quantized ONNX model.

This is a very important engineering choice. It shows that the authors prioritize deployment reliability over theoretical elegance.

---

## 8. Offline Geometry Extraction

After segmentation, the project does not directly claim a gaze point. It first converts the segmentation map into geometric information.

### 8.1 What is extracted

From the predicted label map, the code extracts:

* iris mask,
* pupil mask,
* outer boundary mask,
* connected components,
* cleaned morphology,
* centroids,
* fitted ellipses,
* area values,
* normalized offsets between pupil and iris,
* pupil-to-iris area ratio.

### 8.2 Why this matters

This intermediate geometric representation is the bridge between segmentation and gaze estimation.

Instead of using raw pixels directly for gaze mapping, the project uses interpretable features such as:

* where the pupil center is relative to the iris center,
* how far the pupil is from the center of the eye,
* and how stable the geometry is.

That makes the system easier to debug, smoother to calibrate, and more explainable.

---

## 9. Sequence Processing and Cleaning

The repository also supports batch processing of eye image sequences stored in folders such as `S_0`, `S_1`, and so on.

### 9.1 Batch sequence workflow

For each sequence, the system can:

1. run segmentation on all frames,
2. save `.npy` label maps,
3. extract geometry into `geometry.csv`,
4. detect abnormal frames,
5. interpolate short bad segments,
6. preserve long gaps as missing values,
7. smooth valid segments,
8. generate plots and summary CSV files.

### 9.2 Why sequence cleaning is needed

Real eye tracking is noisy. Some frames can be corrupted by:

* eyelid occlusion,
* blur,
* segmentation mistakes,
* unstable pupil shape,
* or sudden jumps in the extracted features.

The sequence filtering code is therefore not cosmetic. It is an important stabilization step for downstream use.

---

## 10. Desktop PC Real-Time Pipeline

The PC-side real-time system is implemented in `realtime_eye_direction.py`.

This script is primarily a live demonstration and debugging environment.

### 10.1 Purpose of the PC pipeline

The desktop version is useful for:

* testing a trained model with a webcam,
* selecting an eye region manually,
* validating segmentation quality,
* testing calibration behavior,
* visualizing the extracted geometry,
* and comparing `.pth` or `.onnx` runtime paths.

It is therefore both a demo and a developer validation tool.

### 10.2 Input source

The script opens a desktop camera with OpenCV and captures full frames.

Because the whole webcam frame contains much more than the eye, the user must first select a **Region of Interest (ROI)** with the mouse. This ROI is the part of the image that should contain the eye.

### 10.3 PC real-time chain step by step

The PC pipeline can be described in the following sequence.

#### Step 1: camera capture

A webcam frame is captured through OpenCV.

#### Step 2: ROI selection

The user draws a rectangle around the eye. This selected region becomes the working image for all later steps.

#### Step 3: preprocessing

The ROI is converted to grayscale, resized to the model input size, and converted to a tensor.

#### Step 4: segmentation inference

The system runs either:

* a PyTorch checkpoint, or
* an ONNX model.

This produces a segmentation label map.

#### Step 5: geometry extraction

The label map is converted into iris and pupil geometry using the shared geometry code.

#### Step 6: feature selection

The system derives a gaze-related feature from the geometry. It supports two modes:

* `pupil_iris` — based on the normalized pupil-to-iris offset,
* `iris_only` — based on iris position and boundary information.

#### Step 7: confidence estimation

A feature quality tracker estimates how trustworthy the current frame is.

This confidence depends on factors such as:

* iris area stability,
* motion stability,
* shape quality,
* and whether required features are valid.

#### Step 8: temporal smoothing

An adaptive Kalman filter smooths the feature over time. This reduces jitter.

#### Step 9: calibration

The user can launch a nine-point calibration routine. During calibration, the system displays targets and records eye features associated with known screen points.

#### Step 10: screen point prediction

After calibration, the smoothed feature is mapped to a normalized screen coordinate.

#### Step 11: visualization

The script draws:

* iris and pupil ellipses,
* centers,
* direction arrows,
* a small screen preview panel,
* segmentation preview,
* ROI preview,
* debug text.

### 10.4 Interpretation of the PC pipeline

The desktop real-time script is best understood as an interactive laboratory for the eye tracker. It lets the developer see every important intermediate result, which is extremely useful for debugging.

---

## 11. Raspberry Pi Real-Time Pipeline

The Raspberry Pi system is the most important real-time deployment in this repository.

It is implemented mainly in:

* `realtime_eye_direction_pi.py`,
* and the `eyetrack/raspi/` package.

Unlike the desktop demo, this is a more serious deployment-oriented architecture.

### 11.1 Main design idea

The Raspberry Pi version is built around **two different camera streams**:

1. an **eye camera** for close-up infrared eye tracking,
2. an **FPV camera** for the user’s forward-facing scene.

The project does not simply display a gaze arrow on the eye image. Instead, it uses the eye camera to estimate gaze, and then publishes that gaze as metadata that can be associated with the FPV stream.

This means the Raspberry Pi system combines:

* sensing,
* inference,
* synchronization,
* calibration,
* metadata output,
* and video streaming or recording.

### 11.2 Why the Pi pipeline is separate from the PC pipeline

The Raspberry Pi script does not reuse the desktop script directly because the deployment constraints are different.

The Pi side must handle:

* CSI cameras rather than generic USB webcams,
* fixed ONNX deployment rather than flexible PyTorch testing,
* low-latency CPU inference,
* synchronized timestamps across two cameras,
* optional RTSP serving,
* optional local recording,
* and machine-readable metadata output.

So this is not just a port. It is a distinct system architecture.

---

## 12. Raspberry Pi Pipeline Modules

The Raspberry Pi implementation is split into several modules, each responsible for one part of the runtime.

### 12.1 `runtime.py`

This module loads the quantized ONNX model and runs inference.

Its responsibilities are:

* verify that the model is an ONNX file,
* verify that it is a quantized QDQ-style model,
* build an ONNX Runtime session,
* read model metadata,
* enforce single-channel grayscale input,
* run segmentation inference and output label maps.

This is important because it clearly defines the Pi runtime as an ONNX-only deployment path.

### 12.2 `camera.py`

This module manages Picamera2-based camera acquisition.

Its responsibilities are:

* discover available cameras,
* configure each camera stream,
* capture frames in background worker threads,
* preserve metadata timestamps,
* convert eye YUV420 frames into model input tensors,
* convert FPV frames into BGR images when needed.

The eye camera uses YUV420 and directly extracts the luminance plane, which is a smart choice because the model only needs grayscale input.

### 12.3 `sync.py`

This module handles synchronization and freshness checks.

Its responsibilities are:

* resolve timestamps from camera metadata,
* pair eye and FPV frames,
* compute time skew between the two streams,
* detect stale frames.

This is important because in a dual-camera system, a predicted gaze point is only meaningful if it is associated with the correct FPV frame.

### 12.4 `metadata.py`

This module builds and publishes gaze metadata.

The metadata packet includes:

* eye and FPV timestamps,
* predicted screen coordinates,
* calibration state,
* tracking validity,
* feature validity,
* FPS,
* inference latency,
* synchronization status,
* stale frame status,
* and a status message.

The metadata can be published:

* to standard output as JSON lines,
* or over UDP.

This design is very strong because it separates **video transport** from **gaze information transport**.

### 12.5 `rtsp.py`

This module creates a GStreamer RTSP server.

Its job is to take the FPV video frames and stream them over the network as H.264 video.

It supports hardware encoder candidates first and can fall back to software encoding if necessary.

### 12.6 `recorder.py`

This module records FPV video locally into MKV files.

Importantly, the recorded version is not the same as the RTSP stream:

* RTSP mode sends raw FPV video without burning gaze graphics into it,
* record mode writes video locally and overlays a lightweight gaze marker.

This is a very sensible design. It allows live remote visualization without permanently altering the streamed video, while still offering locally stored annotated recordings for later review.

### 12.7 `overlay.py`

This module draws visual elements on FPV frames.

It can draw:

* a gaze point,
* a recording gaze circle,
* a calibration target,
* and a debug panel.

Different overlay functions are used depending on the output mode.

### 12.8 `terminal.py`

This module provides non-blocking keyboard input, allowing the user to press commands while the pipeline is running.

---

## 13. Raspberry Pi Real-Time Chain Step by Step

This is the most important section of the report.

The Raspberry Pi real-time chain can be described as follows.

### Step 1: model loading

The Pi runtime loads a **quantized ONNX model** and checks that its input shape exactly matches the configured eye camera resolution.

### Step 2: camera discovery and startup

The script discovers available Picamera2 devices and starts:

* one near-eye infrared stream,
* and optionally one FPV stream.

Each stream runs in its own worker thread.

### Step 3: eye frame acquisition

The eye camera continuously captures frames in YUV420 format. The software takes the luminance plane as the grayscale eye image.

This is efficient because no unnecessary color conversion is required.

### Step 4: FPV frame acquisition

The FPV camera captures scene video frames separately.

### Step 5: timestamp pairing and health checks

The system reads timestamps from frame metadata and compares the eye and FPV frame times. It also checks whether either stream has become stale.

This helps prevent the system from associating a gaze estimate with an old or mismatched scene frame.

### Step 6: segmentation inference

Each new eye frame is normalized and sent to the quantized ONNX model. The model returns a segmentation label map.

### Step 7: geometry extraction

The label map is converted into iris and pupil geometry. This step is shared with the offline and desktop pipeline, which is good because it avoids different geometry logic between environments.

### Step 8: feature extraction and confidence

A tracking feature is selected from the geometry, using either `pupil_iris` or `iris_only` mode. The system then computes a confidence score to estimate whether the feature is trustworthy.

### Step 9: temporal filtering

A Kalman filter smooths the feature over time to reduce instability and jitter.

### Step 10: calibration

When the user presses the calibration command, the system starts a nine-point calibration session.

For each calibration target:

* the user looks at a known point,
* the system waits for a settle period,
* valid feature samples are collected,
* the samples are filtered and averaged,
* and finally all collected samples are used to fit a mapping model.

The project can fit an affine model or a polynomial model depending on the amount of calibration information available.

### Step 11: gaze prediction

Once calibration is complete, the smoothed feature is mapped to a normalized screen position `(u, v)`.

### Step 12: metadata packaging

The current gaze result, status information, synchronization state, and timing information are packed into a metadata packet.

### Step 13: output path selection

The system supports two output modes.

#### RTSP mode

In RTSP mode:

* the FPV video is streamed,
* the gaze result is not permanently burned into the streamed image,
* metadata is emitted separately.

This is useful for remote systems that want to overlay gaze externally.

#### Record mode

In record mode:

* the FPV stream is recorded locally,
* a lightweight gaze circle is drawn into the recorded video,
* recording is only allowed after calibration is complete.

### Step 14: optional local eye preview

A local eye preview window can be enabled. It shows:

* the eye image,
* fitted geometry,
* calibration state,
* screen preview,
* confidence,
* and runtime status.

This acts as a debugging aid, not as the main output channel.

---

## 14. Why the Raspberry Pi Design Is Strong

The Raspberry Pi pipeline demonstrates several good engineering ideas.

### 14.1 Clear separation of responsibilities

The system separates:

* camera capture,
* inference,
* synchronization,
* calibration,
* video transport,
* and metadata transport.

This makes the code easier to maintain and easier to adapt.

### 14.2 Efficient grayscale input path

The eye camera uses the Y plane directly from YUV420. This avoids unnecessary preprocessing overhead.

### 14.3 Quantized model requirement

The Pi runtime only accepts quantized ONNX models. This is a good deployment rule because it prevents users from accidentally running an overly heavy development model on the device.

### 14.4 Dual output strategy

The system distinguishes between:

* **streaming**, where gaze is external metadata,
* **recording**, where a small gaze marker is embedded.

That is a thoughtful design for real operational use.

### 14.5 Synchronization awareness

The code explicitly tracks skew and stale states. This is very important in any multi-camera real-time system.

### 14.6 Calibration built into deployment

Calibration is not treated as an offline research tool. It is integrated into the real-time runtime, which makes the system more practical.

---

## 15. Comparison Between PC and Raspberry Pi Real-Time Pipelines

Although both pipelines perform real-time eye tracking, their roles are different.

### PC pipeline

The PC version is mainly for:

* experimentation,
* validation,
* debugging,
* visual inspection,
* and flexible testing with `.pth` or `.onnx` models.

It is more interactive and developer-oriented.

### Raspberry Pi pipeline

The Pi version is mainly for:

* deployment,
* dual-camera runtime operation,
* live metadata generation,
* RTSP streaming,
* local recording,
* and embedded execution.

It is more system-oriented and operational.

In short:

* the **PC version** is a real-time testbench,
* the **Pi version** is a real-time embedded product prototype.

---

## 16. Supporting Offline Scripts That Should Still Be Mentioned

Even though this report focuses on real-time pipelines, several offline scripts are still important because the real-time system depends on them.

### 16.1 `Train.py`

Used to train the segmentation model.

### 16.2 `predict_segmentation_to_npy.py`

Used to generate prediction label maps on image folders.

### 16.3 `extract_geometry_from_segmentation.py`

Used to convert segmentation output into geometry CSV files.

### 16.4 `batch_predict_and_extract_sequences.py`

Used to process entire image sequences automatically.

### 16.5 `baseline_filter_sequences_v3.py`

Used to clean noisy geometry signals by detecting bad frames, interpolating short bad intervals, preserving long gaps, and smoothing valid regions.

### 16.6 `analyze_sequence_geometry.py`

Used to generate analysis plots and sequence-level reports.

### 16.7 `export_unet_to_onnx.py`

Used to convert a trained PyTorch model into ONNX.

### 16.8 `quantize_onnx_model.py`

Used to perform static ONNX quantization, which is necessary for the Raspberry Pi deployment.

### 16.9 `evaluate_onnx_model.py`

Used to evaluate quantized ONNX performance and optionally compare it to the PyTorch model.

### 16.10 `benchmark_onnx_model.py`

Used to measure inference latency under ONNX Runtime.

These tools support the real-time system even if they are not part of the runtime loop itself.

---

## 17. Limitations and Important Practical Notes

Several important practical limitations should be mentioned clearly.

### 17.1 The system depends on segmentation quality

If the model cannot segment iris and pupil correctly, the entire gaze estimation chain becomes unstable.

### 17.2 The real-time gaze estimate depends on calibration

Without calibration, the system can still compute internal eye features, but it cannot reliably map them to a screen point.

### 17.3 The Raspberry Pi runtime requires a quantized ONNX model

This is not optional in the current design.

### 17.4 Desktop and Pi serve different purposes

The desktop script is not a drop-in replacement for the Pi deployment. The Pi runtime is more specialized and constrained.

### 17.5 Sequence cleaning remains important

Even with a good model, real eye data contains outliers and unstable frames. That is why the repository includes dedicated sequence filtering tools.

---

## 18. Conclusion

The `QAT` branch of EyeTrack is a well-structured eye tracking project that combines machine learning, geometry-based feature extraction, quantization, and real-time deployment.

Its most important achievement is not only the segmentation model itself, but the complete system built around it.

The repository supports a full chain:

* training a lightweight eye segmentation model,
* preparing it for efficient deployment,
* extracting interpretable eye geometry,
* smoothing and analyzing temporal sequences,
* running live eye tracking on a desktop for debugging,
* and deploying a dual-camera real-time system on Raspberry Pi.

The Raspberry Pi pipeline is the strongest and most deployment-oriented part of the project. It is designed as a modular embedded runtime that combines:

* a quantized ONNX eye model,
* dual CSI camera capture,
* synchronization logic,
* confidence-aware feature tracking,
* nine-point calibration,
* metadata publication,
* RTSP streaming,
* and local annotated recording.

For a non-technical reader, the simplest way to understand EyeTrack is this:

**the project watches the eye, identifies the iris and pupil, measures how they move, learns how those movements correspond to looking directions, and then sends the result into a real-time video system.**

That makes EyeTrack not just a neural network experiment, but a complete prototype for real-time gaze-aware vision systems.

---

## 19. Suggested Reading Order for Newcomers

For someone reading the repository for the first time, the most useful order is:

1. `README.md`
2. `realtime_eye_direction_pi.py`
3. `eyetrack/raspi/runtime.py`
4. `eyetrack/raspi/camera.py`
5. `eyetrack/raspi/metadata.py`
6. `eyetrack/gaze.py`
7. `realtime_eye_direction.py`
8. `eyetrack/models/unet.py`
9. `eyetrack/workflows/train.py`
10. ONNX and quantization scripts

This order helps the reader understand the final deployed system first, then move backward toward the model and training details.

---

## 20. Final Summary in One Paragraph

EyeTrack is a modular eye-tracking system built around a lightweight U-Net segmentation model. It detects iris and pupil regions, extracts geometric features, calibrates them to screen coordinates, and delivers real-time gaze information. The desktop pipeline is mainly a testing and visualization environment, while the Raspberry Pi pipeline is a dedicated embedded deployment system that uses a quantized ONNX model, dual CSI cameras, synchronization logic, metadata output, RTSP streaming, and optional local recording. The `QAT` branch therefore represents a complete engineering workflow from model development to deployable real-time gaze tracking.
