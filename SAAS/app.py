import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import joblib
import numpy as np

# --- 1. Initialize variables and Load Model Components ---

preprocessor = None
kmeans = None
cluster_priorities = None
models_loaded_successfully = False

try:
    preprocessor = joblib.load('preprocessor.joblib')
    kmeans = joblib.load('kmeans_model.joblib')
    cluster_priorities = joblib.load('cluster_priorities.joblib')
    models_loaded_successfully = True
except FileNotFoundError:
    
    pass

# --- 2. App Title and Introduction ---
st.title("🚀 Saasquatch AI Lead Prioritization Tool")
st.write("Upload your lead data to let our AI identify your hottest prospects!")

# --- 3. Display Model Loading Status ---

if models_loaded_successfully:
    st.sidebar.success("ML Model and Preprocessor loaded successfully!")
else:
    st.error("Error: ML model components not found. Please ensure 'preprocessor.joblib', 'kmeans_model.joblib', and 'cluster_priorities.joblib' are in the same directory as app.py.")
    st.stop() # Stop the app if crucial files are missing

# --- 4. File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        leads_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully! Running AI analysis...")

        # --- 5. Data Preprocessing for New Data  ---
        
        expected_input_cols = list(preprocessor.feature_names_in_)

        # Create a copy to work with, avoiding modifying the original uploaded DataFrame
        leads_df_for_processing = leads_df.copy()

        # Check for missing columns in the uploaded file compared to what's expected
        missing_cols = [col for col in expected_input_cols if col not in leads_df_for_processing.columns]
        if missing_cols:
            st.error(f"Error: Your uploaded CSV is missing the following crucial columns required for processing: {', '.join(missing_cols)}. Please ensure your CSV matches the format of the training data.")
            st.stop() # Stop if essential columns are missing

        # Check for extra columns in the uploaded file not used during training
        extra_cols = [col for col in leads_df_for_processing.columns if col not in expected_input_cols]
        if extra_cols:
            st.warning(f"Warning: Your uploaded CSV contains extra columns that will be ignored: {', '.join(extra_cols)}. Only columns used in training will be processed.")
            # Drop extra columns to ensure consistency for the preprocessor
            leads_df_for_processing = leads_df_for_processing.drop(columns=extra_cols)

        # Ensure column order is consistent with the training data
        leads_df_for_processing = leads_df_for_processing[expected_input_cols]

        # Handle potential boolean/object types and missing numerical values
        for col in leads_df_for_processing.columns:
            if leads_df_for_processing[col].dtype == 'object':
                # Convert 'true'/'false' strings to actual booleans if they exist
                if leads_df_for_processing[col].astype(str).str.lower().isin(['true', 'false']).any():
                    leads_df_for_processing[col] = leads_df_for_processing[col].astype(str).str.lower().map({'true': True, 'false': False, 'nan': np.nan}).fillna(False)
            # Impute missing numerical values (e.g., fill with 0) before transformation
            elif leads_df_for_processing[col].dtype in [np.number, 'int64', 'float64'] and leads_df_for_processing[col].isnull().any():
                leads_df_for_processing[col] = leads_df_for_processing[col].fillna(0) # Using 0 for simplicity

        # Apply the preprocessor (transform the new raw data)
        try:
            processed_data_new = preprocessor.transform(leads_df_for_processing)
        except ValueError as e:
            st.error(f"Error during data preprocessing. This often means your uploaded CSV columns or data types do not exactly match the format the model was trained on. Details: {e}")
            st.stop()

        # Convert to dense array if the preprocessor outputted a sparse matrix
        processed_data_new_dense = processed_data_new.toarray() if hasattr(processed_data_new, 'toarray') else processed_data_new

        # --- 6. Predict Clusters and Assign Priorities ---
        st.spinner("Predicting lead clusters...")
        leads_df['Cluster'] = kmeans.predict(processed_data_new_dense)
        leads_df['Priority'] = leads_df['Cluster'].map(cluster_priorities)

        # --- 7. Display Prioritized Leads ---
        # Sort leads for better visualization (Hot leads first)
        prioritized_leads_df = leads_df.sort_values(
            by='Priority',
            key=lambda x: x.map({'Hot':0, 'Warm':1, 'Cold':2}), # Define sorting order for priorities
            ascending=True
        ).reset_index(drop=True)

        st.subheader("📊 AI-Prioritized Leads")

        # Key Metrics Summary
        hot_leads = prioritized_leads_df[prioritized_leads_df['Priority'] == 'Hot'].shape[0]
        warm_leads = prioritized_leads_df[prioritized_leads_df['Priority'] == 'Warm'].shape[0]
        cold_leads = prioritized_leads_df[prioritized_leads_df['Priority'] == 'Cold'].shape[0]

        # Use columns for a neat display of metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Leads", len(prioritized_leads_df))
        with col2:
            st.metric("Hot Leads", hot_leads)
        with col3:
            st.metric("Warm Leads", warm_leads)
        with col4:
            st.metric("Cold Leads", cold_leads)

        st.write("---")

        # Filtering options for the displayed DataFrame
        st.subheader("Filter Leads")
        selected_priority = st.multiselect(
            "Filter by Priority",
            options=['Hot', 'Warm', 'Cold'],
            default=['Hot', 'Warm', 'Cold'] 
        )

        filtered_df = prioritized_leads_df[prioritized_leads_df['Priority'].isin(selected_priority)]

        st.dataframe(filtered_df) # Display the filtered DataFrame

        # Download button for prioritized leads
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prioritized Leads CSV",
            data=csv,
            file_name='ai_prioritized_leads.csv',
            mime='text/csv',
        )

    except Exception as e:
        # Catch any other unexpected errors during the processing pipeline
        st.error(f"An unexpected error occurred during processing. Please ensure your CSV is correctly formatted and contains the expected data. Details: {e}")
        st.warning("Please re-check your CSV file for correct column names, data types, and any special characters.")

st.write("---")
st.info("💡 This tool uses K-Means clustering to segment your leads and assign priorities based on learned patterns.")