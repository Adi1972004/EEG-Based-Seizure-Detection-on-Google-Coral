EEG-Based Seizure Detection on Google Coral
Project Description
This project implements a low-power, hardware-accelerated seizure detection system using Deep Convolutional Neural Networks (DCNNs) tailored for deployment on embedded edge AI platforms, specifically the Google Coral Dev Board. The system processes Electroencephalogram (EEG) signals in real-time, aiming for accurate classification of complex, high-dimensional signals while addressing latency, energy consumption, and privacy concerns associated with cloud-based solutions.

The core model, a lightweight DCNN, is trained on real-world EEG datasets structured as 1-second windows (64x18 matrices). Post-training quantization is applied to optimize the model for edge deployment, significantly reducing memory footprint and computational load. Practical implementation and benchmarking were conducted on the Google Coral Dev Board, leveraging its Edge TPU for hardware-accelerated inference.

Features
Real-time Seizure Detection: Processes EEG signals for immediate detection of seizure events.

Deep Convolutional Neural Network (DCNN): Utilizes a compact and lightweight DCNN model optimized for edge devices.

Quantized Models: Employs INT8 quantization for efficient inference on the Google Coral Edge TPU.

Low Latency: Achieves an average inference time of 1.82 ms per sample.

Energy Efficient: Operates with an average power consumption of 1.37 W.

High Accuracy: Demonstrates a classification accuracy of 90.80% on test data, with a sensitivity of 82.84% and specificity of 93.72%.

Edge AI Deployment: Designed for on-device inference, ensuring data privacy and autonomy without cloud dependence.

Hardware Requirements
Google Coral Dev Board: Or any Google Coral device with an Edge TPU (e.g., Coral USB Accelerator).

Here's an image of the Google Coral Dev Board:

And a block diagram of its architecture, highlighting the Edge TPU:

Software Requirements
Python 3.x

TensorFlow

TensorFlow Lite (TFLite)

Edge TPU Compiler

tflite_runtime (for Python inference on Coral)

NumPy

Pandas (for data handling, if applicable)

Scikit-learn (for evaluation metrics, if applicable)

Installation Guide
1. Clone the Repository
git clone https://github.com/your-username/EEG-Based-Seizure-Detection-on-Google-Coral.git
cd EEG-Based-Seizure-Detection-on-Google-Coral

2. Install Python Dependencies
It is recommended to use a virtual environment.

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install tensorflow numpy pandas scikit-learn

3. Install TensorFlow Lite Runtime for Coral
Follow the official Google Coral documentation for installing tflite_runtime on your specific Coral device. This typically involves:

# For Mendel Linux (Coral Dev Board)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime

For other setups (e.g., USB Accelerator on a host PC), refer to: https://coral.ai/docs/accelerator/get-started/

4. Install Edge TPU Compiler
The Edge TPU Compiler is needed to compile your TensorFlow Lite model for the Edge TPU.

# On your development machine (not necessarily the Coral device)
sudo apt-get install edgetpu-compiler

Refer to: https://coral.ai/docs/edgetpu/compiler/

Usage Instructions
1. Prepare Your Model
Ensure your trained DCNN model is in TensorFlow Lite format. If you have a .tflite model, you need to quantize it to INT8 and compile it for the Edge TPU.

# Example command to compile a TFLite model for Edge TPU
edgetpu_compiler your_model.tflite

This will generate your_model_edgetpu.tflite. Place this compiled model in the models/ directory.

2. Prepare EEG Data
Your EEG data should be preprocessed and formatted as 64x18 matrices, representing 1-second windows. You can use NumPy arrays or other suitable formats. Place your test data in the data/ directory.

3. Run Inference
Navigate to the project root and execute the main inference script.

python3 src/main_inference.py --model_path models/your_model_edgetpu.tflite --data_path data/your_eeg_test_data.npy

(Note: main_inference.py and your_eeg_test_data.npy are placeholders. You will need to create these files based on your specific implementation.)

The script will output real-time predictions, confidence scores, inference times, and TPU utilization, similar to the results shown in the seminar report. An example of the real-time inference output on the Coral Dev Board is shown below:

Results
The Edge TPU implementation demonstrated strong performance for real-time seizure detection.
Here's a summary of the key performance metrics:

Classification Metrics
Accuracy: 90.80%

Sensitivity (Recall): 82.84%

False Positive Rate (FPR): 6.28%

Here's the confusion matrix from the evaluation:

Temporal Performance
Average Inference Time: 1.82 ms

Maximum Inference Time: 3.06 ms

Minimum Inference Time: 0.94 ms

The distribution of inference times is shown below, with the majority completing below 2ms:

Power Characteristics
Average Power Consumption: 1.37 W

Maximum Power Consumption: 1.96 W

Minimum Power Consumption: 0.96 W

The power consumption distribution reveals consistent low-power operation:

Project Structure (Suggested)
.
├── README.md
├── .gitignore
├── LICENSE
├── images/                   # New folder for images
│   ├── confusion_matrix.jpeg
│   ├── coral_board.jpg
│   ├── coral_dev_board_block_diagram.PNG
│   ├── inference_on_coral_board.jpg
│   ├── inference_time_distribution.jpeg
│   ├── power_usage_distribution.jpeg
│   └── seizure_model_coral_analysis.jpeg
├── data/
│   └── raw_eeg_data/
│   └── preprocessed_eeg_data/
│       └── your_eeg_test_data.npy  # Example preprocessed data
├── models/
│   └── trained_model.h5            # Original Keras/TF model
│   └── trained_model.tflite        # Converted TFLite model
│   └── trained_model_edgetpu.tflite # Compiled for Edge TPU
├── src/
│   ├── main_inference.py           # Main script for running inference on Coral
│   ├── model.py                    # DCNN model definition (training part)
│   ├── preprocessing.py            # EEG data preprocessing functions
│   ├── train.py                    # Script for model training (if applicable)
│   └── utils.py                    # Utility functions
├── notebooks/
│   └── model_training_and_quantization.ipynb # Jupyter notebook for development
└── requirements.txt                # Python dependencies

Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
This project is based on the seminar report "Low Power Hardware Accelerator for Biomedical Signal Processing" by Aditya Chandel, guided by Dr. Anand D. Darji, Department of Electronics Engineering, Sardar Vallabhbhai National Institute of Technology, May-2025.