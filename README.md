

# **Time Series Forecasting with LSTM**

This repository contains the code and documentation for a time series forecasting project using an LSTM (Long Short-Term Memory) model. The goal of the project is to predict the number of monthly airline passengers based on historical data from the **Air Passengers dataset**.


## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)


## **Project Overview**
This project demonstrates how to adapt an off-the-shelf LSTM model from a PyTorch tutorial to solve a time series forecasting problem. The dataset used is the **Air Passengers dataset**, which contains monthly totals of international airline passengers from 1949 to 1960. The project includes:
- Data preprocessing and normalization.
- Implementation of an LSTM model with PyTorch.
- Hyperparameter tuning and model evaluation.
- Visualization of predictions vs. actual values.
  
![image](https://github.com/user-attachments/assets/2b599157-07e1-4abb-b6ce-173a0110fbc1)


## **Dataset**
The dataset used in this project is the **Air Passengers dataset**, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/rakannimer/air-passengers). It contains 144 rows of data, with each row representing the number of passengers (in thousands) for a specific month.

### **Dataset Format**
- **Month**: The month and year (e.g., "1949-01").
- **Passengers**: The number of passengers (in thousands).


## **Requirements**
To run this project, you need the following Python libraries:
- Python 3.8 or higher
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the required libraries using the following command:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```


## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/time-series-forecasting-lstm.git
   ```
2. Navigate to the project directory:
   ```bash
   cd time-series-forecasting-lstm
   ```
3. Install the required libraries (see [Requirements](#requirements)).



## **Usage**
1. **Data Preprocessing**:
   - The dataset is preprocessed to make it stationary and normalized for better model performance.
   - Run the `data_preprocessing.py` script to preprocess the data:
     ```bash
     python data_preprocessing.py
     ```

2. **Training the Model**:
   - Train the LSTM model using the `train.py` script:
     ```bash
     python train.py
     ```
   - The script will save the trained model to a file (`model.pth`).

3. **Evaluating the Model**:
   - Evaluate the model's performance on the test set using the `evaluate.py` script:
     ```bash
     python evaluate.py
     ```
   - The script will generate a plot comparing the predicted vs. actual values.

4. **Making Predictions**:
   - Use the trained model to make predictions on new data using the `predict.py` script:
     ```bash
     python predict.py
     ```


## **Results**
The following table summarizes the performance of the three models implemented in this project:

| Model                                | Test Loss (MSE) |
|--------------------------------------|-----------------|
| Baseline LSTM                        | 0.082146        |
| LSTM with Larger Hidden Size         | 0.031879        |
| Final LSTM with Dropout and Early Stopping | 0.060095        |

The **LSTM with Larger Hidden Size** achieved the best performance with a Test Loss (MSE) of **0.031879**. Below is a visualization of the predictions vs. actual values:

![image](https://github.com/user-attachments/assets/f9253b5d-8d5c-4875-90c5-bb2475595eb0)


## **Contributing**
Contributions to this project are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with a detailed description of your changes.




## **Acknowledgments**
- The **Air Passengers dataset** was sourced from [Kaggle](https://www.kaggle.com/datasets/rakannimer/air-passengers).
- The PyTorch tutorials provided valuable insights for implementing the LSTM model.


