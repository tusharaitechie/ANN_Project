# 🧠 Customer Churn Prediction using Artificial Neural Networks

```{=html}
<p align="center">
```
`<strong>`{=html}An end-to-end Deep Learning project that predicts
whether a bank customer is likely to churn.`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<a href="https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/">`{=html}
`<img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge" alt="Live Demo">`{=html}
`</a>`{=html}
`<a href="https://github.com/tusharaitechie/ANN_Project">`{=html}
`<img src="https://img.shields.io/badge/💻%20GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">`{=html}
`</a>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/TensorFlow-ANN-orange?style=flat-square&logo=tensorflow" alt="TensorFlow">`{=html}
`<img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn" alt="Scikit-Learn">`{=html}
`<img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Project Overview

Customer churn is a critical business problem for customer-centric
organizations, particularly in the banking domain. The objective of this
project is to build a **binary classification model using an Artificial
Neural Network (ANN)** that estimates whether a customer is likely to
leave a bank based on demographic, financial, and account-related
attributes.

This project goes beyond model training. It demonstrates an **end-to-end
Machine Learning / Deep Learning workflow**:

> **Raw Data → Preprocessing → Feature Engineering → Scaling → ANN →
> Hyperparameter Tuning → Model Serialization → Inference Pipeline →
> Streamlit Deployment**

The trained model is exposed through an interactive Streamlit
application where a user can enter customer information and receive a
churn probability and classification.

### 🎯 What this project demonstrates

-   Practical Artificial Neural Network implementation using
    TensorFlow/Keras
-   Categorical feature encoding and numerical feature scaling
-   Reproducible preprocessing using serialized encoders and scaler
-   ANN architecture experimentation
-   Hyperparameter optimization with `GridSearchCV` + `SciKeras`
-   Cross-validation
-   Early stopping and TensorBoard monitoring
-   Model serialization and production-style inference
-   Interactive ML application development with Streamlit
-   Deployment of a trained Deep Learning model as a web application

------------------------------------------------------------------------

## 🚀 Live Demo

### Try the deployed application

**Live App:**\
https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/

The application accepts customer attributes and runs the same
preprocessing pipeline used during model development before generating
the ANN prediction.

------------------------------------------------------------------------

## 🧩 Problem Statement

Given customer information such as:

-   Credit Score
-   Geography
-   Gender
-   Age
-   Tenure
-   Account Balance
-   Number of Products
-   Credit Card Status
-   Active Membership Status
-   Estimated Salary

predict the target variable:

**`Exited`**

Where:

-   `1` → Customer churned
-   `0` → Customer did not churn

The model also produces a **churn probability between 0 and 1**.

------------------------------------------------------------------------

## 🏗️ End-to-End Architecture

``` text
                         ┌─────────────────────────┐
                         │   Churn Modelling Data  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Data Cleaning &       │
                         │   Feature Selection     │
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
             Categorical Features              Numerical Features
                     │                                 │
                     ▼                                 ▼
              Label Encoding                    Standard Scaling
              One-Hot Encoding                         │
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Artificial Neural       │
                         │  Network (TensorFlow)   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Churn Probability      │
                         │       0.0 → 1.0         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Threshold Classification│
                         │       > 0.50 → Churn    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Streamlit Web App    │
                         └─────────────────────────┘
```

------------------------------------------------------------------------

## 📊 Dataset

The project uses the **Churn Modelling** dataset containing customer
demographic, banking, and financial information.

### Features used

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
  `Exited`            Target variable representing churn

### Features removed

The following columns were excluded because they are identifiers rather
than useful predictive signals:

-   `RowNumber`
-   `CustomerId`
-   `Surname`

------------------------------------------------------------------------

## 🧹 Data Preprocessing

A consistent preprocessing pipeline is essential because the exact same
transformations used during training must also be applied during
inference.

### 1. Remove non-predictive identifiers

``` text
RowNumber
CustomerId
Surname
```

### 2. Encode Gender

`Gender` is transformed using `LabelEncoder`.

``` text
Categorical Gender
       ↓
LabelEncoder
       ↓
Numerical Representation
```

### 3. Encode Geography

`Geography` is transformed using `OneHotEncoder`.

``` text
Geography
   ↓
OneHotEncoder
   ↓
Geography_France
Geography_Germany
Geography_Spain
```

The encoder is configured to handle unseen categories safely.

### 4. Scale numerical features

`StandardScaler` is used before feeding the features into the ANN.

``` text
Raw Numerical Features
          ↓
    StandardScaler
          ↓
     Scaled Features
          ↓
          ANN
```

### 🔒 Reusable preprocessing artifacts

The following objects are serialized and reused during prediction:

-   `label_encoder_gender.pkl`
-   `onehot_encoder_geo.pkl`
-   `scaler.pkl`

This prevents training-time and inference-time transformations from
becoming inconsistent.

------------------------------------------------------------------------

## 🧠 Artificial Neural Network

The primary ANN classification architecture is:

``` text
Input Features
      │
      ▼
Dense Layer — 64 Neurons
Activation — ReLU
      │
      ▼
Dense Layer — 32 Neurons
Activation — ReLU
      │
      ▼
Output Layer — 1 Neuron
Activation — Sigmoid
      │
      ▼
Churn Probability
```

### Model configuration

  Parameter           Value
  ------------------- ---------------------------
  Model Type          Artificial Neural Network
  Task                Binary Classification
  Framework           TensorFlow / Keras
  Hidden Layer 1      64 neurons
  Hidden Layer 2      32 neurons
  Hidden Activation   ReLU
  Output Layer        1 neuron
  Output Activation   Sigmoid
  Optimizer           Adam
  Loss                Binary Cross-Entropy
  Metric              Accuracy
  Maximum Epochs      100
  Training Control    Early Stopping

### Why Sigmoid?

For binary classification, the sigmoid activation maps the model output
to a value between `0` and `1`, which can be interpreted as the model's
estimated probability for the positive class.

------------------------------------------------------------------------

## 🔬 Hyperparameter Tuning

Instead of relying on a single ANN architecture, the project includes an
experiment using:

-   `GridSearchCV`
-   `SciKeras`
-   3-fold cross-validation

### Search space

  Hyperparameter   Values
  ---------------- -----------------
  Neurons          16, 32, 64, 128
  Hidden Layers    1, 2
  Epochs           50, 100

This resulted in:

``` text
16 configurations × 3 CV folds
= 48 model fits
```

### Recorded best configuration

``` text
Epochs : 100
Layers : 1
Neurons: 16
CV Score: ~0.8584
```

> **Note:** The CV score above is the recorded cross-validation score
> from the hyperparameter-tuning experiment. It should not be
> interpreted as a held-out test-set score.

This experiment demonstrates the use of systematic model selection
rather than choosing an ANN architecture arbitrarily.

------------------------------------------------------------------------

## ⏱️ Early Stopping

Early stopping is incorporated into the training workflow to avoid
unnecessary training once validation performance stops improving.

Benefits include:

-   Reduced unnecessary computation
-   Lower risk of overfitting
-   Better generalization
-   More efficient training

------------------------------------------------------------------------

## 📈 TensorBoard Monitoring

TensorBoard is integrated for monitoring the ANN training process.

The project can be used to inspect:

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy
-   Training progress

Training logs are maintained under the project logging directories.

------------------------------------------------------------------------

## 💾 Model Serialization

The trained model and preprocessing objects are saved as reusable
artifacts.

``` text
model.h5
label_encoder_gender.pkl
onehot_encoder_geo.pkl
scaler.pkl
```

This enables the deployed application to perform inference without
retraining the model.

------------------------------------------------------------------------

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface for
real-time predictions.

### Application inputs

Users can provide:

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

### Inference pipeline

``` text
User Input
    ↓
Create DataFrame
    ↓
Encode Gender
    ↓
One-Hot Encode Geography
    ↓
Combine Features
    ↓
StandardScaler
    ↓
Trained ANN
    ↓
Prediction Probability
    ↓
Threshold > 0.50
    ↓
Churn / Not Churn
```

### Prediction rule

``` text
Probability > 0.50
        ↓
Likely to Churn

Probability ≤ 0.50
        ↓
Not Likely to Churn
```

------------------------------------------------------------------------

## 🗂️ Project Structure

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

### Notebook responsibilities

  -----------------------------------------------------------------------
  Notebook                            Purpose
  ----------------------------------- -----------------------------------
  `experiments.ipynb`                 Main ANN development,
                                      preprocessing, training, early
                                      stopping, TensorBoard and model
                                      saving

  `hyperparametertuningann.ipynb`     ANN architecture search using
                                      GridSearchCV + SciKeras

  `prediction.ipynb`                  Loading serialized preprocessing
                                      objects and performing predictions

  `salaryregression.ipynb`            Additional ANN regression
                                      experiment
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Tech Stack

### Programming & Data

-   Python
-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn
-   SciKeras
-   Feature preprocessing
-   Cross-validation
-   GridSearchCV

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
-   Git / GitHub

------------------------------------------------------------------------

## ⚙️ Installation & Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/tusharaitechie/ANN_Project.git
```

### 2. Navigate to the project

``` bash
cd ANN_Project
```

### 3. Create a virtual environment

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

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

``` bash
streamlit run app.py
```

The application will be available locally through the Streamlit URL
shown in the terminal.

------------------------------------------------------------------------

## 🧪 Example Inference Flow

A simplified example of the deployed prediction process:

``` python
# Load trained model
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

The important design principle is that **preprocessing and inference use
the same saved transformations as training**.

------------------------------------------------------------------------

## 📌 Key Engineering Takeaways

This project demonstrates several concepts that are important when
moving from a notebook experiment toward a deployable ML solution:

### 1. Training ≠ Deployment

A trained model alone is not enough. The deployment pipeline must
reproduce the preprocessing steps used during training.

### 2. Preprocessing artifacts matter

Saving the encoder and scaler makes the inference pipeline reproducible.

### 3. Architecture should be validated experimentally

Grid search and cross-validation provide a systematic way to compare ANN
configurations.

### 4. Model monitoring improves development

TensorBoard makes it easier to understand training behavior and identify
potential overfitting or under-training.

### 5. A model becomes more useful when it is accessible

The Streamlit layer converts the trained ANN into an interactive
application that can be tested without opening a notebook.

------------------------------------------------------------------------

## 🔭 Future Improvements

Potential next steps for making the project more production-ready:

-   Add Precision, Recall and F1-Score
-   Add Confusion Matrix
-   Add ROC-AUC analysis
-   Add threshold tuning based on business cost
-   Add model explainability using SHAP
-   Add interactive model-performance dashboards
-   Add stronger input validation and exception handling
-   Add automated unit/integration tests
-   Add CI/CD using GitHub Actions
-   Containerize with Docker
-   Add experiment tracking and model versioning
-   Migrate legacy `.h5` artifacts to the modern `.keras` format
-   Improve Streamlit UI/UX
-   Add model/data drift monitoring

------------------------------------------------------------------------

## 🎓 Learning Outcomes

Building this project provided hands-on experience across the complete
ML lifecycle:

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
Model Evaluation
       ↓
Model Serialization
       ↓
Inference Pipeline
       ↓
Streamlit Application
       ↓
Cloud Deployment
```

The project therefore demonstrates not only **Deep Learning model
development**, but also the practical steps required to turn an ML
experiment into a usable application.

------------------------------------------------------------------------

## 👨‍💻 Author

**Tushar**

Aspiring **Data Scientist / Machine Learning Engineer** focused on
building practical Machine Learning, Deep Learning and AI applications.

### 🔗 Project Links

-   🚀 **Live Demo:**
    https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/
-   💻 **GitHub Repository:**
    https://github.com/tusharaitechie/ANN_Project

------------------------------------------------------------------------

## ⭐ Feedback

If you find this project useful, feel free to explore the repository,
try the live application, or share feedback.

```{=html}
<p align="center">
```
`<strong>`{=html}Built with Python • TensorFlow • Scikit-learn •
SciKeras • Streamlit`</strong>`{=html}
```{=html}
</p>
```
