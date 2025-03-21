# Cloud-Edge Collaboration Benchmarks

A collection of benchmark applications designed to evaluate the performance of different cloud-edge collaborative frameworks. These benchmarks were developed as part of academic research to standardize performance testing across various cloud-edge computing scenarios.

## Overview

This repository contains benchmarks for common computer vision and IoT tasks that benefit from cloud-edge collaboration:

- **Object Detection**: Fast R-CNN based detection workflow for evaluating framework parallelism
- **Human Pose Estimation**: YOLOv5-based pose estimation for mobile edge scenarios
- **Image Classification**: YOLOX-cls implementation for scenarios like intelligent vehicles and drones
- **Appliance State Recognition**: LSTM-based appliance state recognition for smart home applications

## Benchmark Details

### Object Detection (Detection)

A widely used application in cloud-edge scenarios with the following characteristics:
- **Input**: Images (600×600)
- **Output**: Detected objects with corresponding labels
- **Architecture**:
  - **Cloud Nodes**: Data pre-processing, model loading, and result aggregation
  - **Edge Nodes**: Detection process based on Fast R-CNN (PyTorch framework)
- **Purpose**: Tests the parallelism capabilities of frameworks on cloud-edge testbeds
  
### Human Pose Estimation (Estimation)

Often used in mobile edge scenes, such as mobile phones and VR glasses:
- **Input**: Video data
- **Output**: Video with annotated human poses
- **Architecture**:
  - **Cloud Nodes**: Data pre-processing and result aggregation
  - **Edge Nodes**: Pose estimation process based on yolov5s6_pose (ONNX framework)

### Image Classification (Classification)

Widely used in multiple cloud-edge scenarios, including intelligent vehicles and drones:
- **Input**: Images (224×224)
- **Output**: Category labels
- **Architecture**:
  - **Cloud Nodes**: Image pre-processing and result aggregation
  - **Edge Nodes**: Classification algorithm based on YOLOX-cls (ONNX framework)

### Appliance State Recognition (Recognition)

Designed to recognize the state of household appliances, which is a typical scenario in smart homes:
- **Input**: Sequential current signal
- **Output**: Category tag of appliances
- **Architecture**:
  - **Cloud Nodes**: Data loading and aggregation
  - **Edge Nodes**: Recognition inference using LSTM model

## Getting Started

### Prerequisites

To run these benchmarks, you'll need:
- Python 3.8+
- PyTorch
- ONNX Runtime
- Edge computing devices
- Cloud computing resources

### Installation

```bash
git clone https://github.com/MrlixiangWE/cloud-edge-collaboration-benchmarks.git
cd cloud-edge-collaboration-benchmarks
# See individual benchmark directories for specific setup instructions
```

## Usage

Each benchmark is structured to evaluate different aspects of cloud-edge collaboration frameworks:

1. **Setup your cloud-edge environment** according to the specifications in each benchmark directory
2. **Configure the workload distribution** between cloud and edge nodes
3. **Run the benchmark** using the provided scripts
4. **Collect performance metrics** including latency, throughput, and resource utilization

See individual benchmark directories for specific usage instructions and configuration options.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributions

Contributions to improve or extend these benchmarks are welcome. Please feel free to submit pull requests or open issues to discuss potential improvements.
