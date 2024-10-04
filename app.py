import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib  # For loading saved models

# Function to load different file types
def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format! Please upload a CSV or Excel file.")
        return None

# Function to normalize all numeric columns
def normalize_all_data(df, scaler):
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df

# Streamlit app
def main():
    st.title("Tax Evasion CLustering Using  Kmeans Algorithm")

    # File upload: Accept .csv, .xlsx, and .xls files
    uploaded_file = st.file_uploader("Upload a CSV or Excel file for clustering", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Load the uploaded file based on its format
        df = load_file(uploaded_file)

        if df is not None:
            st.write("Uploaded Data:")
            st.dataframe(df.head())

            # Convert column names to uppercase to match training
            df.columns = df.columns.str.upper()

            # Load the pre-trained model and scaler
            model_file = 'vat_tax_evasion_model.pkl'  # Path to your saved KMeans model
            scaler_file = 'scaler.pkl'       # Path to your saved scaler

            try:
                kmeans_model = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
            except:
                st.error("Error loading the saved model or scaler. Ensure they are available.")
                return

            # Normalize the entire dataset (scale all numeric columns)
            df_cleaned = df.copy()
            numeric_columns = list(df_cleaned.select_dtypes(include=['number']).columns)
            numeric_columns.remove("YEAR")

            try:
                df_normalized = normalize_all_data(df_cleaned[numeric_columns], scaler)
            except Exception as e:
                st.error(f"Error in scaling: {e}")
                return

            # Ensure 'PROFIT' is created after normalization
            if 'profit' not in df_normalized.columns and 'TOTAL_SALES' in df_normalized.columns and 'TOTAL_PURCHASES' in df_normalized.columns:
                df_normalized['profit'] = df_normalized['TOTAL_SALES'] - df_normalized['TOTAL_PURCHASES']

            # Select only the columns that were used during training
            required_features = ['TOTAL_SALES', 'TOTAL_PURCHASES', 'NET_TAX_PAYABLE', 'profit']

            # Check if the necessary features are in the dataframe
            missing_features = [feature for feature in required_features if feature not in df_normalized.columns]
            if missing_features:
                st.error(f"Missing required columns: {', '.join(missing_features)}")
                return

            # Extract only the columns that the model was trained on
            X_new = df_normalized[required_features]

            # Apply the pre-trained model to the new data
            try:
                cluster_labels = kmeans_model.predict(X_new)
                df['cluster_label'] = cluster_labels
            except Exception as e:
                st.error(f"Error in prediction: {e}")
                return

            # Display the resulting data with clusters
            st.write("Data with Clusters:")
            st.dataframe(df.head())

            # Download the result as an Excel file
            output_file = 'clustered_new_data.xlsx'
            df.to_excel(output_file, index=False)

            # Provide download link for the clustered data
            with open(output_file, "rb") as file:
                st.download_button(label="Download Clustered Data", data=file, file_name="clustered_new_data.xlsx")

if __name__ == "__main__":
    main()
