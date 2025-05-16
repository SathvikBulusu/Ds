import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_column_types, plot_histogram, plot_scatter, plot_boxplot, plot_correlation_heatmap, convert_df_to_csv
import io # For plot download

st.set_page_config(layout="wide") # Already set in main app, but good practice
st.title("📊 Exploratory Data Analysis")

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a dataset from the main page first.")
    st.stop()

df = st.session_state.df
numeric_cols = st.session_state.numeric_cols
categorical_cols = st.session_state.categorical_cols
datetime_cols = st.session_state.datetime_cols

# --- Data Filtering (Interactive) ---
st.sidebar.header("Filter Data (for EDA)")
filter_cols = st.sidebar.multiselect("Select columns to filter by", df.columns)
df_filtered = df.copy()

for col in filter_cols:
    if col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected_range = st.sidebar.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
        df_filtered = df_filtered[(df_filtered[col] >= selected_range[0]) & (df_filtered[col] <= selected_range[1])]
    elif col in categorical_cols:
        unique_vals = df[col].unique()
        selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
        df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
    elif col in datetime_cols:
        try:
            # Ensure the column is datetime
            dt_series = pd.to_datetime(df[col], errors='coerce')
            if not dt_series.isnull().all():
                min_date, max_date = dt_series.min(), dt_series.max()
                if pd.NaT not in [min_date, max_date]: # Check if min/max are valid dates
                    selected_date_range = st.sidebar.date_input(
                        f"Filter {col} (Date Range)",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                    )
                    if len(selected_date_range) == 2:
                         df_filtered = df_filtered[(dt_series >= pd.to_datetime(selected_date_range[0])) & (dt_series <= pd.to_datetime(selected_date_range[1]))]
                else:
                    st.sidebar.warning(f"Cannot create date range filter for {col} due to NaT values or conversion issues.")
            else:
                st.sidebar.warning(f"Column {col} could not be fully converted to datetime for filtering.")
        except Exception as e:
            st.sidebar.error(f"Error filtering {col}: {e}")


st.header("Filtered Dataset Overview")
st.dataframe(df_filtered.head())
st.write(f"Displaying {df_filtered.shape[0]} rows out of {df.shape[0]} total rows after filtering.")

# --- Initial EDA (Automated) ---
tab1, tab2, tab3 = st.tabs(["Basic Info", "Descriptive Stats", "Missing Values"])
with tab1:
    st.subheader("Dataset Shape:")
    st.write(f"Rows: {df_filtered.shape[0]}, Columns: {df_filtered.shape[1]}")
    st.subheader("Column Data Types:")
    st.dataframe(df_filtered.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

with tab2:
    st.subheader("Descriptive Statistics:")
    st.dataframe(df_filtered.describe(include='all').T)

with tab3:
    st.subheader("Missing Values Count:")
    missing_values = df_filtered.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Count']
    st.dataframe(missing_values[missing_values['Missing Count'] > 0])


# --- Interactive Visualization ---
st.header("Interactive Visualizations")
plot_type = st.selectbox("Select Plot Type",
                         ["Univariate", "Bivariate", "Multivariate (Correlation/Pair Plot)"])

# Function to download plotly figures
def download_plotly_figure(fig, filename="plot.html"):
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    return html_bytes

if plot_type == "Univariate":
    uni_col = st.selectbox("Select a column for Univariate Analysis", df_filtered.columns)
    if uni_col:
        if uni_col in numeric_cols:
            fig_hist = plot_histogram(df_filtered, uni_col)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.download_button("Download Plot (HTML)", download_plotly_figure(fig_hist, f"{uni_col}_hist.html"), f"{uni_col}_hist.html", "text/html")
        elif uni_col in categorical_cols:
            val_counts = df_filtered[uni_col].value_counts().reset_index()
            val_counts.columns = [uni_col, 'count']
            fig_bar = px.bar(val_counts, x=uni_col, y='count', title=f"Bar Chart of {uni_col}")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.download_button("Download Plot (HTML)", download_plotly_figure(fig_bar, f"{uni_col}_bar.html"), f"{uni_col}_bar.html", "text/html")
        elif uni_col in datetime_cols:
            st.write(f"Time series analysis for {uni_col}:")
            try:
                # Ensure it's datetime
                dt_series = pd.to_datetime(df_filtered[uni_col], errors='coerce').dropna()
                if not dt_series.empty:
                    # Plot count over time (e.g., monthly)
                    counts_over_time = dt_series.groupby(dt_series.dt.to_period("M")).count()
                    if not counts_over_time.empty:
                        fig_ts = px.line(x=counts_over_time.index.to_timestamp(), y=counts_over_time.values,
                                         labels={'x': 'Time Period', 'y': 'Count'}, title=f"Counts over Time for {uni_col}")
                        st.plotly_chart(fig_ts, use_container_width=True)
                        st.download_button("Download Plot (HTML)", download_plotly_figure(fig_ts, f"{uni_col}_ts.html"), f"{uni_col}_ts.html", "text/html")
                    else:
                        st.info(f"Not enough data points to plot time series for {uni_col} after aggregation.")
                else:
                    st.warning(f"Could not plot time series for {uni_col} as it contains no valid dates after filtering.")
            except Exception as e:
                st.error(f"Error plotting time series for {uni_col}: {e}")


elif plot_type == "Bivariate":
    col1 = st.selectbox("Select X-axis column", df_filtered.columns, key="col1_bi_eda")
    col2 = st.selectbox("Select Y-axis column / Grouping column", df_filtered.columns, key="col2_bi_eda")
    color_by_col_bi = st.selectbox("Optional: Color by (Categorical Column)", [None] + categorical_cols, key="color_bi_eda")

    if col1 and col2 and col1 != col2:
        fig_bi = None
        if col1 in numeric_cols and col2 in numeric_cols:
            fig_bi = plot_scatter(df_filtered, col1, col2, color_col=color_by_col_bi if color_by_col_bi in categorical_cols else None)
        elif col1 in numeric_cols and col2 in categorical_cols:
            fig_bi = plot_boxplot(df_filtered, x_col=col2, y_col=col1) # x=categorical, y=numeric
        elif col1 in categorical_cols and col2 in numeric_cols:
            fig_bi = plot_boxplot(df_filtered, x_col=col1, y_col=col2)
        elif col1 in categorical_cols and col2 in categorical_cols:
            pivot_df = pd.crosstab(df_filtered[col1], df_filtered[col2])
            fig_bi = px.imshow(pivot_df, text_auto=True, aspect="auto", title=f"Heatmap: {col1} vs {col2}")
        
        if fig_bi:
            st.plotly_chart(fig_bi, use_container_width=True)
            st.download_button("Download Plot (HTML)", download_plotly_figure(fig_bi, f"{col1}_vs_{col2}.html"), f"{col1}_vs_{col2}.html", "text/html")
        else:
            st.warning("Select compatible column types or ensure columns are different.")

elif plot_type == "Multivariate (Correlation/Pair Plot)":
    st.subheader("Correlation Heatmap (Numeric Columns)")
    # Use numeric_cols identified from the original df to ensure consistency for correlation
    # but apply it to the filtered dataframe.
    numeric_cols_in_filtered = [col for col in numeric_cols if col in df_filtered.columns]
    if len(numeric_cols_in_filtered) > 1:
        fig_corr = plot_correlation_heatmap(df_filtered, numeric_cols_in_filtered)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
            st.download_button("Download Plot (HTML)", download_plotly_figure(fig_corr, "corr_heatmap.html"), "corr_heatmap.html", "text/html")
    else:
        st.info("Not enough numeric columns in the filtered data for correlation heatmap.")

    if len(numeric_cols_in_filtered) > 1:
        st.subheader("Pair Plot (Numeric Columns)")
        max_cols_pairplot = 5 # Performance consideration
        default_pair_plot_cols = numeric_cols_in_filtered[:min(max_cols_pairplot, len(numeric_cols_in_filtered))]
        pair_plot_cols = st.multiselect("Select columns for Pair Plot", numeric_cols_in_filtered, default=default_pair_plot_cols)
        
        if pair_plot_cols and len(pair_plot_cols) > 1:
            if len(pair_plot_cols) > max_cols_pairplot:
                st.warning(f"Pair plots with more than {max_cols_pairplot} columns can be slow. Selected: {len(pair_plot_cols)}")
            
            color_by_col_pair = st.selectbox("Optional: Color by (Categorical Column)", [None] + [c for c in categorical_cols if c in df_filtered.columns], key="color_pair")
            
            fig_pair = px.scatter_matrix(df_filtered[pair_plot_cols], 
                                         color=df_filtered[color_by_col_pair] if color_by_col_pair else None,
                                         title="Pair Plot")
            st.plotly_chart(fig_pair, use_container_width=True)
            st.download_button("Download Plot (HTML)", download_plotly_figure(fig_pair, "pair_plot.html"), "pair_plot.html", "text/html")
        elif pair_plot_cols and len(pair_plot_cols) <=1:
            st.info("Please select at least two numeric columns for a pair plot.")
    else:
        st.info("Not enough numeric columns for pair plot.")

st.sidebar.markdown("---")
st.sidebar.info("Explore your data visually. Filters apply only to this EDA page.")