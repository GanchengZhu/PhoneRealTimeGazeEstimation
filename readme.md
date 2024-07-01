# Linear Adaptive Batch Normalization Based Neural Networks and Swarm Intelligence Calibration for Real-time Gaze Estimation on Smartphones

Eye-tracking has emerged as a valuable tool for both research and clinical applications. However, traditional eye-tracking systems are often bulky and expensive, limiting their widespread adoption in various fields. With the advancement of deep learning and edge computing technologies, eye-tracking on smartphones becomes feasible. Nonetheless, the field still faces practical challenges related to large-scale datasets, model inference speed, and gaze estimation accuracy. 

The present study created a new dataset that contains over 3.2 million face images collected with recent phone models and proposed a comprehensive smartphone eye-tracking pipeline, comprising a deep neural network framework (MGazeNet), a personalized model calibration method, and a heuristic gaze signal filter. The MGazeNet model introduced a Linear Adaptive Batch Normalization module to efficiently combine eye and face features, achieving state-of-the-art gaze estimation accuracy: 1.59 cm on the GazeCapture dataset and 1.48 cm on our custom dataset. Additionally, an algorithm utilizing Multi-Verse Optimization to optimize the hyperparameters of support vector regression (MVO-SVR) was proposed to improve eye tracking calibration accuracy with 13 or fewer ground-truth gaze points, further achieving gaze estimation accuracy to 0.89 cm compared to traditional calibration methods. This integrated approach allows for eye-tracking accuracy that is comparable to that of research-grade eye-trackers, offering new application possibilities for smartphone eye-tracking.


# Gaze Estimation

Train three models (iTracker, AFFNet, and MGazeNet) for gaze estimation and Deploy them on phones or PCs.
The experiment was performed on a workstation equipped with dual NVIDIA RTX 3090 graphics cards, an Intel Xeon Silver 4210 processor, and 256 GB of RAM. The software environment consisted of Ubuntu 18.0, CUDA 11.0, and Python 3.9.12. T

1.  Dataset preparation

    Download the preprocessed GazeCapture dataset using https://pan.baidu.com/s/1OUHRS_ZGWZ8J-YXIrasRIA?pwd=gaze .
    As per the IRB approved by the research ethics community governing the present study, the authors are not allowed to share any of the face images contained in the ZJUGaze dataset.
    
    Once downloaded, move to the dataset directory and unzip the dataset:
    
    ```bash
    cd GazeEstimation/dataset
    cp path/to/gazecapture.zip ./
    unzip gazecapture.zip
    ```

2. Train model

    Start by unzipping the dataset if not done already:
    ```bash
    unzip gazecapture.zip
    ```
   
    Install Pythonic dependency library:
    ```bash
    python -m pip install -r requirements.txt
    ```
    
    Create a configuration for a gaze estimation experiment according to a template `GazeEstimation/config/config_itracker.yaml`, 
    then run model training:
    ```bash
    # if you train iTracker
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=29900 train.py --world_size 2 --config_file config/config_itracker.yaml 
    # train your config file
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=29900 train.py --world_size 2 --config_file {path to configuration file} 
    ```
  
    Run model evaluation:
    ```bash
    python test.py --config_file {path to the configuration file}
    ```

3. Convert model
    
   Convert the PyTorch checkpoint model into an ONNX model, and then convert the ONNX model into an MNN model.

4. Deploy
    
    Deploy your MNN model to either a smartphone or a PC for real-time gaze estimation. Please refer to https://github.com/GanchengZhu/MediaPipe-MMN-Android.

5. Finetuning

    You can run the following code to run the finetuning experiments with the ZJUGaze dataset.
   
     ```bash
    cd GazeEstimation/finetuing
    python finetuning_freezen.py --model_path {model_path} --data_source {data_source}
    ```

6. How to get feature for calibration

     ```bash
    cd GazeEstimation
    python predict_features.py --model_path {model_path} --npy_save_dir {--npy_save_dir}
    ```
# Personal Calibration 

Calibration is essential for mapping the relationship between ocular features and gaze coordinates in both appearance- and geometry-based video eye tracking. We utilize three swarm intelligence algorithms to optimize the hyperparameters of the support vector regressor. Below are the steps to run the experiments.

Before running the following code, please download the calibration feature dataset 
using https://pan.baidu.com/s/1hdX_ntAam6cCwV5Za1qH0A?pwd=gaze . Then, unzipping the dataset and moving all files and directories to `SwarmIntelligentCalibration/calibration_data`.

1. Hyperparameter search for support vector regressor

    Navigate to the hyperparameter search directory and run the experiment:

    ```bash
    cd SwarmIntelligentCalibration
    python -m pip install -r requirements.txt
    cd hyperparameter_search
    python run.py
   ```
   
    Swarm intelligence algorithms can be time-consuming, so please be patient during this process.


2. Validation searched hyperparameter
    
   We assume that you have installed the Python package in requirements.txt. If not, see above.

   ```bash
   # For example, run smooth pursuit calibration 
   cd SwarmIntelligentCalibration/validation
   python run.py
   ```

# Eye movement filters

## Heuristic Filter
The heuristic filter is designed to reduce noise in gaze signals before detecting eye movements such as saccades and fixations. It relies on heuristics derived from an examination of noise patterns in raw gaze data, resembling expert systems' rules of thumb.

## One Euro Filter
The One Euro filter is a low-latency filtering algorithm, functioning as a first-order low-pass filter with an adaptive cutoff frequency. It stabilizes the signal at low velocities by using a low cutoff and minimizes delay at higher velocities with a higher cutoff. This filter is computationally efficient and suitable for real-time applications.

## how to run eye movement filters

   Download feature files from https://pan.baidu.com/s/1GkYKdjz1FHEuad4CyWiXOg?pwd=gaze. Then, unzipping the dataset and moving all files and directories to `Filter/feature`.
    
   ```bash
   cd Filter
   python -m pip install -r requirements.txt
   python filter_expriment.py
   python filter_exp_plotting.py
   ```

# Bibtex

Not applicable. The manuscript is currently under review.
