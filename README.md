🧠 Customer Churn Prediction Using Artificial Neural Network
<p align="center">
🚀 End-to-End Machine Learning Project | ANN | Deep Learning | Streamlit Deployment
<br> <a href="https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/"> <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge" alt="Live Demo"> </a> <a href="https://github.com/tusharaitechie/ANN_Project"> <img src="https://img.shields.io/badge/💻%20GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub"> </a> </p>
📌 About The Project
Customer churn is one of the most important business problems for customer-centric organizations, especially in the banking sector.

This project uses an Artificial Neural Network (ANN) to predict whether a bank customer is likely to churn (leave the bank) based on their demographic, financial, and account-related information.

The project covers the complete Machine Learning lifecycle — from data preprocessing and feature engineering to model development, hyperparameter tuning, model serialization, and deployment using Streamlit.

The trained ANN model is integrated into an interactive web application where users can enter customer information and receive a churn prediction.

🌐 Live Demo
👉 Try the Live Customer Churn Prediction App
The deployed application allows users to enter customer details and get a real-time churn prediction from the trained Artificial Neural Network.

🎯 Project Objectives
The main objectives of this project are:

Build an Artificial Neural Network for customer churn prediction.

Perform complete data preprocessing and feature engineering.

Handle categorical variables using appropriate encoding techniques.

Scale numerical features before training the neural network.

Experiment with different ANN architectures.

Perform hyperparameter tuning using GridSearchCV.

Implement Early Stopping to improve model training.

Monitor training using TensorBoard.

Save the trained model and preprocessing objects.

Build an interactive prediction application using Streamlit.

Deploy the Machine Learning model as a web application.

🔄 End-to-End Machine Learning Workflow
                    ┌──────────────────────────┐
                    │    Customer Churn Data   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Data Preprocessing     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             Categorical Encoding      Numerical Features
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │    Feature Scaling       │
                    │     StandardScaler       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Artificial Neural Network│
                    │          (ANN)            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Model Prediction      │
                    │   Churn Probability      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Streamlit Web App     │
                    └──────────────────────────┘

📊 Dataset
The project uses the Churn Modelling Dataset.

The dataset contains customer information related to demographics, banking activity, and financial characteristics.

Important Features
Feature	Description
CreditScore	Customer's credit score
Geography	Customer's country
Gender	Customer gender
Age	Customer age
Tenure	Number of years the customer has been with the bank
Balance	Customer account balance
NumOfProducts	Number of products used by the customer
HasCrCard	Whether the customer has a credit card
IsActiveMember	Whether the customer is an active member
EstimatedSalary	Estimated customer salary
Exited	Target variable indicating customer churn

The following columns are removed because they do not provide useful predictive information:

RowNumber
CustomerId
Surname

🧹 Data Preprocessing
The project performs several preprocessing steps before feeding the data into the neural network.

1. Removing Unnecessary Features
The following columns are removed:

RowNumber
CustomerId
Surname

These fields are identifiers rather than meaningful predictive features.

2. Gender Encoding
The Gender column is converted from categorical values into numerical values using LabelEncoder.

Example:

Male   → Numerical Value
Female → Numerical Value

3. Geography Encoding
The Geography feature contains categorical values and is transformed using OneHotEncoder.

This allows the neural network to work with categorical geographical information numerically.

4. Feature Scaling
Numerical features are standardized using:

StandardScaler

Scaling is important for neural networks because it helps keep the input features within a comparable numerical range and can improve training stability.

🧠 Artificial Neural Network
The main classification model is built using TensorFlow / Keras.

The ANN architecture used in the primary experiment is:

Input Layer
     │
     ▼
Dense Layer — 64 Neurons
Activation: ReLU
     │
     ▼
Dense Layer — 32 Neurons
Activation: ReLU
     │
     ▼
Output Layer — 1 Neuron
Activation: Sigmoid

Model Configuration
Parameter	Value
Model Type	Artificial Neural Network
Problem Type	Binary Classification
Hidden Layer 1	64 neurons
Hidden Layer 2	32 neurons
Hidden Activation	ReLU
Output Neurons	1
Output Activation	Sigmoid
Optimizer	Adam
Loss Function	Binary Cross-Entropy
Metric	Accuracy
Maximum Epochs	100
Training Control	Early Stopping

🔬 Hyperparameter Tuning
To experiment with different neural network configurations, GridSearchCV with SciKeras was used.

The tuning process explored different combinations of:

Number of Neurons:
16
32
64
128

Number of Hidden Layers:
1
2

Epochs:
50
100

This resulted in multiple ANN configurations being evaluated using cross-validation.

The recorded best configuration from the hyperparameter tuning experiment was:

Epochs  : 100
Layers  : 1
Neurons : 16
CV Score: ~0.8584

This experiment helped evaluate how different network configurations affected model performance.

⏱️ Early Stopping
Early Stopping is used during ANN training to prevent unnecessary training once the validation performance stops improving.

This helps:

Reduce unnecessary computation.

Prevent over-training.

Improve generalization.

Automatically stop training when further epochs are not beneficial.

📈 TensorBoard
TensorBoard is integrated into the project to monitor the neural network training process.

It can be used to visualize:

Training loss

Validation loss

Training accuracy

Validation accuracy

Model training progress

Training logs are stored inside the project under the logs/ directory.

💾 Model & Preprocessing Artifacts
The trained model and preprocessing objects are saved so that the same transformation pipeline can be reused during prediction.

Saved Files
File	Purpose
model.h5	Trained ANN classification model
label_encoder_gender.pkl	Saved Gender LabelEncoder
onehot_encoder_geo.pkl	Saved Geography OneHotEncoder
scaler.pkl	Saved StandardScaler

Keeping these preprocessing objects is important because the production application must transform new customer data in exactly the same way as the training data.

🌐 Streamlit Application
The trained ANN model is deployed using Streamlit.

Application Workflow
User Input
    ↓
Customer Information
    ↓
Categorical Encoding
    ↓
Feature Transformation
    ↓
StandardScaler
    ↓
Trained ANN Model
    ↓
Prediction Probability
    ↓
Churn Prediction

The application loads the saved model and preprocessing objects and performs inference on new customer information.

🔮 Prediction
The ANN produces a probability value between:

0 → 1

The application uses a threshold-based classification approach.

Prediction Probability > 0.5
            ↓
       Likely to Churn


Prediction Probability ≤ 0.5
            ↓
    Not Likely to Churn

The application also displays the predicted churn probability.

🖥️ Application Inputs
The Streamlit application accepts customer information such as:

Credit Score

Geography

Gender

Age

Tenure

Balance

Number of Products

Credit Card status

Active Member status

Estimated Salary

The entered information is transformed using the saved preprocessing pipeline before being passed to the ANN model.

📁 Project Structure
ANN_Project/
│
├── 📄 app.py
├── 📄 Churn_Modelling.csv
├── 📄 requirements.txt
│
├── 🤖 model.h5
├── 🤖 regression_model.h5
│
├── ⚙️ label_encoder_gender.pkl
├── ⚙️ onehot_encoder_geo.pkl
├── ⚙️ scaler.pkl
│
├── 📓 experiments.ipynb
├── 📓 hyperparametertuningann.ipynb
├── 📓 prediction.ipynb
├── 📓 salaryregression.ipynb
│
├── 📂 logs/
│   └── fit/
│
└── 📂 regressionlogs/
    └── fit/

📓 Project Notebooks
experiments.ipynb
This notebook contains the main ANN development workflow including:

Data loading

Data preprocessing

Feature engineering

Train-test split

ANN model creation

Model training

Early stopping

TensorBoard logging

Model saving

hyperparametertuningann.ipynb
This notebook focuses on ANN hyperparameter tuning using:

GridSearchCV

SciKeras

Different neuron configurations

Different numbers of hidden layers

Different epoch values

prediction.ipynb
This notebook demonstrates how the saved:

ANN model

Label encoder

One-hot encoder

Standard scaler

are loaded and used for prediction on new customer data.

salaryregression.ipynb
This notebook contains an additional ANN regression experiment.

A separate regression model is generated as part of the experimentation process.

🛠️ Tech Stack
<p align="center"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"> <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"> </p>
Technologies Used
Python — Programming Language

TensorFlow / Keras — Deep Learning & ANN

Scikit-learn — Preprocessing & Hyperparameter Tuning

SciKeras — Keras and Scikit-learn Integration

Pandas — Data Manipulation

NumPy — Numerical Computing

Matplotlib — Visualization

TensorBoard — Model Training Monitoring

Streamlit — Web Application & Deployment

Jupyter Notebook — Experimentation

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/tusharaitechie/ANN_Project.git

2. Navigate to the Project Directory
cd ANN_Project

3. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

5. Run the Streamlit Application
streamlit run app.py

The application will open in your browser.

🚀 Deployment
The application is deployed using Streamlit.

Live Application
👉 https://annproject-zkhvuagmyq8godb5q73nzq.streamlit.app/

The deployed application provides an interactive interface for generating customer churn predictions using the trained ANN model.

💡 Key Machine Learning Concepts Demonstrated
This project demonstrates practical implementation of:

Artificial Neural Networks

Deep Learning

Binary Classification

Data Cleaning

Feature Engineering

Label Encoding

One-Hot Encoding

Feature Scaling

Train-Test Split

Cross-Validation

Hyperparameter Tuning

GridSearchCV

SciKeras

Early Stopping

TensorBoard

Model Serialization

Model Inference

Streamlit Deployment

🔭 Future Improvements
Some possible improvements for future versions include:

Add Precision, Recall and F1-Score evaluation.

Add ROC-AUC analysis.

Add a confusion matrix to the application.

Add interactive charts for model performance.

Add SHAP-based model explainability.

Improve the Streamlit UI/UX.

Add better input validation and error handling.

Add automated testing.

Add CI/CD using GitHub Actions.

Containerize the application using Docker.

Add model versioning and experiment tracking.

Migrate the model from legacy .h5 format to the modern .keras format.

📚 Learning Outcomes
By building this project, I gained practical experience in:

Data Preprocessing
       ↓
Feature Engineering
       ↓
Deep Learning
       ↓
Artificial Neural Networks
       ↓
Hyperparameter Optimization
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Prediction Pipeline
       ↓
Web Application Development
       ↓
Machine Learning Deployment

This project helped bridge the gap between building a Machine Learning model in a notebook and deploying that model as a usable application.

👨‍💻 Author
Tushar
Aspiring Data Scientist / Machine Learning Engineer passionate about building practical Machine Learning and Deep Learning applications.

Connect With Me
🐙 GitHub: tusharaitechie

📂 Project Repository: ANN_Project

🚀 Live Demo: Customer Churn Prediction App

⭐ Show Your Support
If you found this project useful or interesting, consider giving the repository a ⭐ Star on GitHub.

Your feedback and suggestions are always welcome!
