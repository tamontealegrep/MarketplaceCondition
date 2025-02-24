---

Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- This document outlines the project workflow project.
    ├── report.md                   <- Detailed report.
    ├── data
    │   ├── processed               <- The final, canonical data sets.
    │   ├── staging                 <- Intermediate data that has been transformed.
    │   └── raw                     <- The original dataset in format jsonlines.
    │
    ├── src                         <- Contains the all source code files.
    │   └── features                <- Contains source code files related with features of the code.
    │       ├── data_preprocess.py  <- Code related to preprocess of the data.
    │       ├── data_process.py     <- Code related to process of the data.
    │       ├── functions.py        <- Code related to general functions.
    │       ├── neural_networks.py  <- Code related to the cration, training and evaluation of neural networks.
    │       ├── new_or_used.py      <- Code related to read the .jsonlines dataset.
    │       └── plots.py            <- Code related to plot.
    │
    ├── notebooks                   <- Jupyter notebooks with steps for data preprocessing, data processing, data exploration analisys, training and evaluating models.
    │
    ├── images                      <- Generated images used in this document.
    │
    └── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.

--------

Predicting Item Condition in Marketplace
==============================
This project addresses the challenge of predicting whether an item listed on a marketplace is new or used. The dataset contains various product attributes, seller information, and listing details. The goal is to build a machine learning model that accurately classifies item conditions based on these features, enabling better decision-making for buyers and sellers.

## Model Selection 
Five models were considered for this task:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Neural Network
5. Soft Voting Ensemble (combining the previous four models) 

## Workflow
The project followed a structured workflow to ensure a methodical approach to data processing, modeling, and evaluation:

1. **Data Collection & Exploration**:
   - Loaded raw data and conducted an exploratory data analysis (EDA) to understand its structure and distributions.
   - Identified missing values, outliers, and class imbalances that could impact model performance.

2. **Data Preprocessing**:
   - Cleaned data by handling missing values, removing duplicates, and addressing inconsistencies.
   - Engineered relevant features such as seller category, listing type, and product pricing characteristics.
   - Encoded categorical variables and scaled numerical features for better model compatibility.

3. **Model Development**:
   - Implemented various classification models: Logistic Regression, Random Forest, XGBoost, Neural Network, and an Ensemble approach.
   - Performed hyperparameter tuning to optimize model performance.
   - Applied cross-validation to ensure robustness and generalization.

4. **Model Evaluation & Selection**:
   - Used Accuracy, F1-score, AUC, Precision, and Sensitivity to assess model effectiveness.
   - Determined the optimal classification threshold using the ROC curve.
   - Selected the best-performing model (Ensemble Model) based on overall performance metrics.

5. **Interpretability & Insights**:
   - Conducted feature importance analysis to understand the most influential predictors.
   - Used model explanations to characterize item conditions and guide potential business applications.
   - Planned future improvements.

6. **Deployment & Documentation**:
   - Organized the project repository with structured directories for data, notebooks, and source code.
   - Documented findings, insights, and future enhancements to facilitate reproducibility and scalability.

# Operational Limits
The model has been conceived and trained with operational limits in mind. Specifically, it has been tailored to handle sales values in Argentina for products priced up to 50,000 ARS and with an initial quantity of up to 150 products. These limits were not arbitrarily chosen but rather derived from the data, ensuring that at least 98% of the dataset falls within these ranges.
# Results
Based on the results, it can be observed that the neural network model exhibited the poorest performance, followed by logistic regression and random forest models. Conversely, XGBoost and the ensemble model demonstrated the best performance, with the ensemble model showing a slightly superior performance.

<table>
  <tr>
    <th></th>
    <th>Logistic Regression</th>
    <th>Random Forest</th>
    <th>XGBoost</th>
    <th>Neural Network</th>
    <th>Ensembled</th>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>0.831</td>
    <td>0.843</td>
    <td>0.851</td>
    <td>0.806</td>
    <td>0.854</td>
  </tr>
  <tr>
    <td>F1 score</td>
    <td>0.841</td>
    <td>0.853</td>
    <td>0.854</td>
    <td>0.813</td>
    <td>0.862</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>0.851</td>
    <td>0.863</td>
    <td>0.897</td>
    <td>0.843</td>
    <td>0.878</td>
  </tr>
  <tr>
    <td>NVP</td>
    <td>0.831</td>
    <td>0.845</td>
    <td>0.891</td>
    <td>0.831</td>
    <td>0.864</td>
  </tr>
  <tr>
    <td>Sensitivity</td>
    <td>0.831</td>
    <td>0.842</td>
    <td>0.815</td>
    <td>0.785</td>
    <td>0.847</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>0.831</td>
    <td>0.845</td>
    <td>0.891</td>
    <td>0.831</td>
    <td>0.864</td>
  </tr>
  <tr>
    <td>AUC</td>
    <td>0.91</td>
    <td>0.92</td>
    <td>0.93</td>
    <td>0.86</td>
    <td>0.93</td>
  </tr>
</table>

The ensemble model emerged as the preferred choice for deployment due to its consistently superior performance across various metrics. With values surpassing 86% for all auxiliary metrics, except sensitivity, the ensemble model showcases robustness and reliability in predicting item conditions accurately within the dataset.

Regarding its classification accuracy, the model effectively distinguishes between new and used items, ensuring their correct categorization. It demonstrates proficiency in identifying used products, capturing the majority of instances within the dataset. However, there's an opportunity for enhancement in correctly identifying new products, as evidenced by a lower sensitivity metric.

The AUC value of 0.93 achieved by the model signifies its exceptional ability to differentiate between positive and negative instances, reflecting a strong discriminatory power. This indicates that in 93% of cases, the model correctly ranks positive instances higher than negative ones. Such performance underscores the model's effectiveness in distinguishing between item conditions.

# Models Interpretability
In the context of model interpretability, it's essential to note that both logistic regression and random forest models exhibited similar behaviors in assigning importance to certain variables. Notably, features such as `price`, `listing_free`, `listing_bronze`, `listing_silver`, `available_quantity` and `num_pictures` received a realtive high importance scores in both models. The sign of the coefficient in logistic regression allows for the construction of interpretations regarding how these variables influence the prediction of whether a product is new or used.

## Variables
1. **price:** High prices tend to favor the classification of products as new.
2. **listing_free:** Sellers categorized as "free" are less likely to offer new products.
3. **listing_bronze:** Sellers classified as "bronze" are more likely to sell new products.
4. **listing_silver:** Sellers classified as "silver" are more likely to sell new products.
5. **available_quantity:** Higher available quantities suggest a higher likelihood of the product being new.
5. **num_pictures:** An increase in the number of published photos decreases the probability of the product being new.

## Item Condition Characterization
Based on the outcomes of the models, it becomes feasible to delineate the characteristics of products.
### New Products
New products are typically characterized by higher prices and are sold in larger quantities. They are often listed by sellers categorized as "bronze" or "silver," or higher seller levels. Additionally, listings for new products may include fewer photos compared to used products. 
### Used Products
Used products are often associated with lower prices and are sold in smaller quantities. They are typically listed by sellers categorized as "free" or with lower seller levels. Listings for used products tend to include a higher number of photos, likely to provide more detail about the condition of the item.

# Areas for Future Development
Within our project, we have identified several areas for improvement and alternative approaches to address the underlying challenge. These opportunities encompass potential enhancements in methodologies, explorations of additional data sources, and the consideration of alternative modeling techniques. Regrettably, due to the constraints imposed by time and resources, these avenues were not pursued. Nevertheless, recognizing their potential significance, we acknowledge them as avenues for future exploration and refinement.

- Exploring the application of Natural Language Processing (NLP) or transformer architectures to analyze the `title` column is an intriguing possibility, aiming to derive embeddings that can be utilized by other models. The `title` column frequently contains valuable information directly indicating whether a product is new or used, serving as a potentially rich source for further analysis.

- Investigating the interpretability of the ensemble model via SHAP (SHapley Additive exPlanations) values presents an intriguing opportunity to gain deeper insights into the inner workings of the model. By leveraging SHAP values, we can elucidate the contribution of each feature to the model's predictions, providing a comprehensive understanding of how different variables influence the classification outcomes. This approach not only enhances our confidence in the model's decisions but also empowers stakeholders to make more informed decisions based on the underlying factors driving the predictions.

# **How to Run the Project**

## **Prerequisites**
- Python >= 3.12
- PyTorch (with CUDA support for GPU usage)
- Required dependencies:
  - NumPy
  - Pandas
  - Scikit-learn
  - Scipy
  - Matplotlib
  - Seaborn
  - Xgboost

## **Installation Instructions**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tamontealegrep/MarketplaceCondition.git
    cd MarketplaceCondition
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Jupyter Notebooks:**

   To run the notebooks, make sure you have **Jupyter Notebook** or **JupyterLab** installed:

   ```bash
   pip install jupyter
   ```

   ```bash
   pip install jupyterlab
   ```

## **License**

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You are free to share, copy, and distribute the material in any medium or format for non-commercial purposes, with proper attribution to the author. However, you cannot alter, transform, or build upon the material, nor can you use it for commercial purposes.

For more details, please refer to the full license at:
https://creativecommons.org/licenses/by-nc-nd/4.0/. 