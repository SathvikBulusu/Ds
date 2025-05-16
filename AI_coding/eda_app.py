import streamlit as st
import pandas as pd
from utils import convert_df_to_csv # For potential download from main page

st.set_page_config(
    page_title="Automated EDA & Prediction Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Automated EDA & Prediction Tool")
st.markdown("""
Welcome to the Automated EDA and Prediction Tool!
Navigate through the pages using the sidebar to explore your data and build quick predictive models.

**👈 Start by uploading your dataset using the sidebar!**
""")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state: # For the Prediction Studio
    st.session_state.df_processed = None
if 'numeric_cols' not in st.session_state:
    st.session_state.numeric_cols = []
if 'categorical_cols' not in st.session_state:
    st.session_state.categorical_cols = []
if 'datetime_cols' not in st.session_state:
    st.session_state.datetime_cols = []
if 'preprocessing_log' not in st.session_state: # To track applied steps
    st.session_state.preprocessing_log = []


# --- File Uploader in Sidebar ---
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df_upload = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Reset states if a new file is uploaded
            if st.session_state.df is None or not st.session_state.df.equals(df_upload):
                st.session_state.df = df_upload
                st.session_state.df_processed = df_upload.copy() # Initialize processed df
                from utils import get_column_types # Local import to avoid circular if utils imports st
                st.session_state.numeric_cols, st.session_state.categorical_cols, st.session_state.datetime_cols = get_column_types(st.session_state.df)
                st.session_state.preprocessing_log = ["Initial data loaded."]
                st.success("File Uploaded Successfully! Navigate to other pages to explore.")
                # st.experimental_rerun() # Force rerun to update page content based on new df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
            st.session_state.df_processed = None


    if st.session_state.df is not None:
        st.sidebar.success("Dataset loaded successfully!")
        st.sidebar.metric("Rows", st.session_state.df.shape[0])
        st.sidebar.metric("Columns", st.session_state.df.shape[1])

        if st.sidebar.button("Download Original Data as CSV"):
            csv = convert_df_to_csv(st.session_state.df)
            st.sidebar.download_button(
                label="Click to Download Original CSV",
                data=csv,
                file_name="original_data.csv",
                mime="text/csv",
            )
    else:
        st.sidebar.warning("Please upload a dataset.")

st.sidebar.markdown("---")
st.sidebar.info("""
Navigation:
- **Exploratory Data Analysis**: Automated EDA, visualizations.
- **Prediction Studio**: Preprocess data, select features, train simple models.
""")

if st.session_state.df is None:
    st.warning("Please upload a dataset using the sidebar to get started.")
    