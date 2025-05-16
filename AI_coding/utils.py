import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats # For Z-score
import io
import streamlit as st

# --- Data Type Identification ---
def get_column_types(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = df.select_dtypes(include=numerics).columns.tolist()
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Attempt to convert to datetime
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except (ValueError, TypeError, OverflowError):
                # If not datetime, consider it categorical
                categorical_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            if col in numeric_cols: numeric_cols.remove(col)

    # Ensure numeric_cols doesn't overlap with datetime_cols
    numeric_cols = [col for col in numeric_cols if col not in datetime_cols]
    # Ensure categorical_cols are truly distinct
    categorical_cols = list(set(categorical_cols) - set(datetime_cols))
    
    return numeric_cols, categorical_cols, datetime_cols


# --- Preprocessing ---
def handle_missing_values(df, column, strategy, constant_value=None):
    df_copy = df.copy()
    if strategy == "Drop Row":
        df_copy.dropna(subset=[column], inplace=True)
    elif strategy == "Drop Column":
        df_copy.drop(columns=[column], inplace=True)
    elif strategy == "Mean":
        df_copy[column].fillna(df_copy[column].mean(), inplace=True)
    elif strategy == "Median":
        df_copy[column].fillna(df_copy[column].median(), inplace=True)
    elif strategy == "Mode":
        df_copy[column].fillna(df_copy[column].mode()[0] if not df_copy[column].mode().empty else "Unknown", inplace=True)
    elif strategy == "Constant" and constant_value is not None:
        df_copy[column].fillna(constant_value, inplace=True)
    return df_copy

def treat_outliers_iqr(df, column, treatment='Cap', lower_quantile=0.25, upper_quantile=0.75, factor=1.5):
    df_copy = df.copy()
    Q1 = df_copy[column].quantile(lower_quantile)
    Q3 = df_copy[column].quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    if treatment == 'Cap':
        df_copy[column] = np.where(df_copy[column] < lower_bound, lower_bound, df_copy[column])
        df_copy[column] = np.where(df_copy[column] > upper_bound, upper_bound, df_copy[column])
    elif treatment == 'Remove':
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    # Can add 'Transform' (log, sqrt) later
    return df_copy

def encode_categorical_column(df, column, method='Label Encoding'):
    df_copy = df.copy()
    if method == 'Label Encoding':
        le = LabelEncoder()
        df_copy[column] = le.fit_transform(df_copy[column].astype(str)) # Handle NaNs as a string category
    elif method == 'One-Hot Encoding':
        df_copy = pd.get_dummies(df_copy, columns=[column], prefix=column, dummy_na=False) # dummy_na=False to avoid NaN columns after imputation
    return df_copy

def scale_numerical_column(df, column, method='StandardScaler'):
    df_copy = df.copy()
    if method == 'StandardScaler':
        scaler = StandardScaler()
        df_copy[column] = scaler.fit_transform(df_copy[[column]])
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
        df_copy[column] = scaler.fit_transform(df_copy[[column]])
    return df_copy

def extract_datetime_features(df, column):
    df_copy = df.copy()
    dt_col = pd.to_datetime(df_copy[column], errors='coerce')
    if not dt_col.isnull().all(): # Proceed if conversion is successful for at least some
        df_copy[f'{column}_year'] = dt_col.dt.year
        df_copy[f'{column}_month'] = dt_col.dt.month
        df_copy[f'{column}_day'] = dt_col.dt.day
        df_copy[f'{column}_weekday'] = dt_col.dt.weekday
        df_copy[f'{column}_hour'] = dt_col.dt.hour
        df_copy.drop(columns=[column], inplace=True) # Drop original datetime column
    return df_copy

# --- Plotting (Examples) ---
def plot_histogram(df, column):
    fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
    return fig

def plot_scatter(df, x_col, y_col, color_col=None):
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline="ols" if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]) else None,
                     title=f"{y_col} vs. {x_col}" + (f" by {color_col}" if color_col else ""))
    return fig

def plot_boxplot(df, x_col, y_col): # x_col categorical, y_col numeric
    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    return fig

def plot_correlation_heatmap(df, numeric_cols):
    if not numeric_cols: return None
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
    return fig

# --- Download Helpers ---
@st.cache_data # Use st.cache_data for dataframes
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def download_plotly_fig(fig, filename="plot.html"):
    buffer = io.StringIO()
    fig.write_html(buffer, full_html=False, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    return html_bytes