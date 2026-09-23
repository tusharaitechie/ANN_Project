# 🧠 Customer Churn Prediction using Artificial Neural Network

> An end-to-end Deep Learning project that predicts the probability of
> customer churn using an Artificial Neural Network (ANN), with a fully
> deployed Streamlit application.

[![Live
Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/)
[![GitHub](https://img.shields.io/badge/💻%20GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tusharaitechie/ANN_Project)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

------------------------------------------------------------------------

## 🚀 Live Demo

**Try the application:**\
[Open Customer Churn Prediction
App](https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/)

The deployed application allows a user to enter customer details and
receive an ANN-based churn prediction without running the training
notebooks locally.

------------------------------------------------------------------------

## 📌 Project Overview

Customer retention is an important business problem in the banking
industry. This project uses an **Artificial Neural Network for binary
classification** to estimate whether a bank customer is likely to leave
the bank.

The project covers the complete workflow from data preprocessing and
model development to hyperparameter tuning, model serialization and
cloud deployment.

### End-to-End Workflow

``` text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
ANN Model Development
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Prediction Pipeline
       ↓
Streamlit Application
       ↓
Cloud Deployment
```

------------------------------------------------------------------------

## 🎯 Problem Statement

Given customer demographic, financial and account-related information,
predict whether the customer is likely to churn.

### Target Variable

  Value   Meaning
  ------- ------------------------
  `0`     Customer did not churn
  `1`     Customer churned

The ANN produces a probability between `0` and `1`, which is then
converted into a binary prediction using a `0.50` threshold in the
application.

------------------------------------------------------------------------

## 📊 Features Used

  Feature             Description
  ------------------- ------------------------------------------
  `CreditScore`       Customer credit score
  `Geography`         Customer's country/region
  `Gender`            Customer gender
  `Age`               Customer age
  `Tenure`            Number of years with the bank
  `Balance`           Customer account balance
  `NumOfProducts`     Number of bank products used
  `HasCrCard`         Whether the customer has a credit card
  `IsActiveMember`    Whether the customer is an active member
  `EstimatedSalary`   Estimated customer salary
  `Exited`            Target variable

### Identifier columns excluded

The following columns are not used as predictive features:

-   `RowNumber`
-   `CustomerId`
-   `Surname`

------------------------------------------------------------------------

## 🧹 Data Preprocessing

The project uses a consistent preprocessing workflow so that the
transformations applied during training can also be reproduced during
inference.

### 1. Gender Encoding

`Gender` is converted from categorical values into numerical values
using `LabelEncoder`.

``` text
Male / Female
      ↓
LabelEncoder
      ↓
Numerical Representation
```

### 2. Geography Encoding

`Geography` is transformed using `OneHotEncoder`.

``` text
France / Germany / Spain
          ↓
   OneHotEncoder
          ↓
Multiple Numerical Features
```

The encoder is configured to handle previously unseen categories safely
during inference.

### 3. Numerical Feature Scaling

Numerical features are standardized using `StandardScaler`.

``` text
Raw Numerical Features
          ↓
    StandardScaler
          ↓
    Scaled Features
          ↓
         ANN
```

### 4. Saved Preprocessing Artifacts

The preprocessing objects are saved and reused by the prediction
application:

``` text
label_encoder_gender.pkl
onehot_encoder_geo.pkl
scaler.pkl
```

This helps maintain consistency between training and production
inference.

------------------------------------------------------------------------

## 🧠 ANN Model Architecture

The main classification model uses the following architecture:

``` text
Input Features
      │
      ▼
Dense Layer
64 Neurons
ReLU Activation
      │
      ▼
Dense Layer
32 Neurons
ReLU Activation
      │
      ▼
Output Layer
1 Neuron
Sigmoid Activation
      │
      ▼
Churn Probability
```

### Model Configuration

  Parameter           Configuration
  ------------------- ---------------------------
  Model               Artificial Neural Network
  Problem             Binary Classification
  Framework           TensorFlow / Keras
  Hidden Layer 1      64 neurons
  Hidden Layer 2      32 neurons
  Hidden Activation   ReLU
  Output Layer        1 neuron
  Output Activation   Sigmoid
  Optimizer           Adam
  Loss Function       Binary Cross-Entropy
  Metric              Accuracy
  Maximum Epochs      100
  Training Control    Early Stopping

### Why Sigmoid?

The output layer uses the sigmoid activation function because this is a
binary classification problem. It maps the model output to a value
between `0` and `1`, which can be used as a probability estimate.

------------------------------------------------------------------------

## 🔬 Hyperparameter Tuning

The project also includes a dedicated hyperparameter-tuning experiment
using:

-   `GridSearchCV`
-   `SciKeras`
-   3-fold cross-validation

### Hyperparameters explored

  Hyperparameter   Values
  ---------------- -----------------
  Neurons          16, 32, 64, 128
  Hidden Layers    1, 2
  Epochs           50, 100

This experiment evaluates multiple ANN configurations instead of relying
on a single manually selected architecture.

### Recorded Best Configuration

``` text
Epochs  : 100
Layers  : 1
Neurons : 16
CV Score: ~0.8584
```

> **Important:** `0.8584` is the recorded cross-validation score from
> the hyperparameter-tuning experiment. It is not presented as a
> held-out test-set accuracy.

------------------------------------------------------------------------

## ⏱️ Early Stopping

Early stopping is used during ANN training to stop training when
validation performance stops improving.

### Benefits

-   Reduces unnecessary training
-   Helps control overfitting
-   Improves training efficiency
-   Can improve generalization

------------------------------------------------------------------------

## 📈 TensorBoard

TensorBoard is integrated into the training workflow for monitoring
model training.

The training logs can be used to inspect:

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy
-   Training progress

This makes it easier to understand how the ANN behaves during training.

------------------------------------------------------------------------

## 💾 Model Serialization

The trained model and preprocessing components are stored as reusable
artifacts.

``` text
model.h5
label_encoder_gender.pkl
onehot_encoder_geo.pkl
scaler.pkl
```

The Streamlit application loads these artifacts at runtime instead of
retraining the ANN for every prediction.

------------------------------------------------------------------------

## 🌐 Streamlit Application

The project includes an interactive Streamlit application for real-time
inference.

### User Inputs

The application accepts customer information including:

-   Geography
-   Gender
-   Age
-   Balance
-   Credit Score
-   Estimated Salary
-   Tenure
-   Number of Products
-   Credit Card status
-   Active Member status

### Prediction Pipeline

``` text
User Input
    ↓
Create Input DataFrame
    ↓
Encode Gender
    ↓
One-Hot Encode Geography
    ↓
Combine Features
    ↓
Apply StandardScaler
    ↓
Load Trained ANN
    ↓
Generate Probability
    ↓
Apply Prediction Threshold
    ↓
Churn / Not Churn
```

### Prediction Logic

``` text
Probability > 0.50
        ↓
Likely to Churn

Probability ≤ 0.50
        ↓
Not Likely to Churn
```

------------------------------------------------------------------------

## 🏗️ Project Architecture

``` text
                         ┌──────────────────────┐
                         │   Churn Modelling    │
                         │       Dataset       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Preprocessing   │
                         │ Encoding + Scaling  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Artificial Neural    │
                         │ Network              │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Churn Probability    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Interface  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                              Cloud Deployment
```

------------------------------------------------------------------------

## 📁 Repository Structure

``` text
ANN_Project/
│
├── app.py
├── Churn_Modelling.csv
├── requirements.txt
│
├── model.h5
├── regression_model.h5
│
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── scaler.pkl
│
├── experiments.ipynb
├── hyperparametertuningann.ipynb
├── prediction.ipynb
├── salaryregression.ipynb
│
├── logs/
│   └── fit/
│
└── regressionlogs/
    └── fit/
```

### Notebook Overview

  -----------------------------------------------------------------------
  Notebook                            Purpose
  ----------------------------------- -----------------------------------
  `experiments.ipynb`                 ANN experimentation, preprocessing,
                                      training, early stopping and
                                      TensorBoard

  `hyperparametertuningann.ipynb`     ANN hyperparameter tuning using
                                      GridSearchCV and SciKeras

  `prediction.ipynb`                  Loading saved preprocessing objects
                                      and generating predictions

  `salaryregression.ipynb`            ANN-based regression experiment
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Technology Stack

### Programming & Data

-   Python
-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn
-   SciKeras
-   GridSearchCV
-   Cross-validation
-   Feature encoding
-   Feature scaling

### Deep Learning

-   TensorFlow
-   Keras
-   Artificial Neural Networks
-   ReLU
-   Sigmoid
-   Adam
-   Binary Cross-Entropy
-   Early Stopping

### Visualization & Monitoring

-   Matplotlib
-   TensorBoard

### Deployment

-   Streamlit

### Development

-   Jupyter Notebook
-   Git
-   GitHub

------------------------------------------------------------------------

## ⚙️ Run the Project Locally

### 1. Clone the repository

``` bash
git clone https://github.com/tusharaitechie/ANN_Project.git
cd ANN_Project
```

### 2. Create a virtual environment

#### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

``` bash
streamlit run app.py
```

The application will open at the local Streamlit address displayed in
the terminal.

------------------------------------------------------------------------

## 🧪 Example Prediction Flow

A simplified version of the inference workflow:

``` python
# Load trained ANN
model = tf.keras.models.load_model("model.h5")

# Load preprocessing artifacts
label_encoder_gender = pickle.load(...)
onehot_encoder_geo = pickle.load(...)
scaler = pickle.load(...)

# Transform incoming customer data
# Encode → One-Hot Encode → Scale

# Generate prediction
prediction = model.predict(input_data_scaled)

churn_probability = prediction[0][0]
```

The key principle is that the same preprocessing logic used during
training is reused during inference.

------------------------------------------------------------------------

## 💡 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

-   Binary classification
-   Artificial Neural Networks
-   Feature preprocessing
-   Label encoding
-   One-hot encoding
-   Feature scaling
-   Model training
-   Validation
-   Cross-validation
-   Hyperparameter tuning
-   Early stopping
-   TensorBoard monitoring
-   Model serialization
-   Production inference
-   Streamlit deployment

------------------------------------------------------------------------

## 📈 From Notebook to Deployable Application

One of the main goals of this project is to demonstrate the difference
between simply training a model and building a usable ML application.

``` text
                 MODEL DEVELOPMENT
                       │
                       ▼
              ┌─────────────────┐
              │ Data Processing │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ ANN Development │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Model Tuning    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Model Saving    │
              └────────┬────────┘
                       │
                       ▼
                 DEPLOYMENT
                       │
                       ▼
              ┌─────────────────┐
              │ Load Artifacts  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ User Input      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Preprocessing   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ ANN Prediction  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Streamlit App   │
              └─────────────────┘
```

------------------------------------------------------------------------

## 🔭 Future Improvements

The following improvements can make the project more
production-oriented:

-   Add Precision, Recall and F1-Score
-   Add Confusion Matrix
-   Add ROC-AUC analysis
-   Tune the classification threshold based on business requirements
-   Add SHAP-based model explainability
-   Add stronger input validation
-   Add automated tests
-   Add GitHub Actions for CI/CD
-   Containerize the application using Docker
-   Add experiment tracking and model versioning
-   Add model/data drift monitoring
-   Migrate legacy `.h5` model artifacts to the modern `.keras` format
-   Improve Streamlit UI and prediction visualization

------------------------------------------------------------------------

## 🎓 Learning Outcomes

By building this project, the following end-to-end concepts were
practiced:

``` text
Data Understanding
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
ANN Development
        ↓
Hyperparameter Tuning
        ↓
Cross-Validation
        ↓
Model Training
        ↓
Model Monitoring
        ↓
Model Serialization
        ↓
Inference Pipeline
        ↓
Streamlit Development
        ↓
Cloud Deployment
```

------------------------------------------------------------------------

## 👨‍💻 Author

### Tushar

Aspiring **Data Scientist \| Machine Learning Engineer \| AI Engineer**

Focused on building practical Machine Learning, Deep Learning and AI
applications.

### Project Links

-   🚀 [Live Streamlit
    Application](https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/)
-   💻 [GitHub
    Repository](https://github.com/tusharaitechie/ANN_Project)

------------------------------------------------------------------------

## ⭐ If You Find This Project Useful

Feel free to explore the repository and try the live application.

**Built with Python • TensorFlow • Scikit-learn • SciKeras • Streamlit**
