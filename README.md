# Lead-prioritization-tool
A useful feature that can enhance the performance of the SaaSquatch leadgen model
---
## 🚀 Lead Prioritization Tool for Caprae Capital

This repository contains an AI-powered Lead Prioritization Tool designed to significantly improve Caprae Capital's SaaSquatch lead generation tool. Building upon an existing lead generation model, this project introduces machine learning techniques to further refine lead scoring and assign intuitive "Hot," "Warm," or "Cold" priority levels.  

---

### 🌟 Introduction

Caprae Capital, like many thriving businesses, relies on effective lead generation. However, simply generating leads isn't enough; the ability to quickly identify and focus on the most promising prospects is paramount for maximizing sales efficiency and conversion rates. This tool addresses that need by acting as an intelligent layer on top of Caprae Capital's existing lead generation efforts.  

By leveraging **unsupervised machine learning (K-Means clustering)**, the tool analyzes various lead attributes to group similar leads together. These statistically defined groups are then interpreted and assigned a priority, enabling Caprae Capital's sales and marketing teams to strategically allocate resources and engage with high-potential leads first. This enhancement helps streamline the sales pipeline, reduce wasted effort on low-value leads, and ultimately drive better business outcomes.

---

### ✨ Features

- **Enhanced Lead Prioritization**: Integrates machine learning to provide more nuanced and data-driven lead scoring.  
- **Intuitive Priority Levels**: Clearly labels leads as "Hot," "Warm," or "Cold" for immediate actionability by sales teams.  
- **CSV Upload Functionality**: Easily upload new lead data for rapid AI analysis.  
- **Interactive Results Display**: View prioritized leads within the app, complete with filtering options.  
- **Downloadable Output**: Export the AI-prioritized leads as a CSV for seamless integration into other workflows.

---

### 🛠️ Technologies Used

- **Python**: The foundational programming language.  
- **Pandas**: For robust data manipulation and analysis.  
- **Scikit-learn**: Utilized for core machine learning components (K-Means Clustering, PCA, ColumnTransformer, StandardScaler, OneHotEncoder).  
- **Streamlit**: For creating the interactive and user-friendly web application interface.  
- **Joblib**: Employed for efficiently saving and loading the trained machine learning models and data transformers.

---

### 🚀 Getting Started

Follow these steps to set up and run the Lead Prioritization AI on your local machine:

1.  **Clone the Repository:** 
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended):** 
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:** * **On macOS/Linux:** 
        ```bash
        source .venv/bin/activate
        ```
    * **On Windows (Command Prompt/PowerShell):** 
       ```bash
        .venv\Scripts\activate
        ```

4.  **Install Dependencies:** 
    ```bash
    pip install -r requirements.txt
    ```
  

5.  **Run the Streamlit Application:** 
    ```bash
    streamlit run app.py
    ```
   

---

### 📊 How to Use the App

1.  Once the Streamlit application is loaded in your browser, locate the "Choose a CSV file" uploader.  
2.  **Upload the `sample_leads.csv` file from this repository.** This file serves as a crucial sample, as the underlying AI model has been specifically trained on its column structure.  
3.  The application will automatically process the uploaded data, predict lead clusters, and assign the corresponding priority levels.  
4.  You can then review the summarized lead metrics and an interactive table of prioritized leads.  
5.  Utilize the "Filter by Priority" option to quickly narrow down the displayed leads to "Hot," "Warm," or "Cold" categories.  
6.  Finally, click "Download Prioritized Leads CSV" to save the enhanced lead list for use in your CRM or other systems.

---

### ⚠️ Key Considerations & Constraints

-   **Input CSV File Schema:** This tool is designed to work with lead data that adheres to a specific format. The uploaded CSV file **must have the exact same column names and data types** as the `sample_leads.csv` file used for training (found within this repository).  
    * **Missing or Mismatched Columns:** Uploading a CSV with missing critical columns will lead to processing errors.  
    * **Extra Columns:** Any additional columns not present in the original training data will be ignored by the model.  
    * **Column Order:** The sequence of columns in your uploaded CSV should ideally match that of the training data for seamless processing.  
    
-   **Model Specificity & Retraining:** The AI model and its preprocessing pipeline have been rigorously trained on the provided dataset. While this provides a powerful enhancement, any significant changes to Caprae Capital's lead data schema or the introduction of new, influential features would necessitate **retraining and redeploying the model** for optimal performance.  
-   **Priority Assignment as a Business Decision:** The "Hot," "Warm," and "Cold" labels are assigned to the statistically derived clusters based on a careful interpretation of their characteristics (e.g., high funding, larger employee count, specific industries). In a real-world application, these priority definitions would be continuously refined and validated in close collaboration with Caprae Capital's sales and marketing domain experts to ensure maximum business impact.  
-   **Unsupervised Learning Nature:** As an unsupervised learning model, the system identifies patterns and segments leads without being explicitly told what a "Hot" or "Cold" lead looks like beforehand. The value comes from the post-clustering interpretation and strategic assignment of priorities.

---
