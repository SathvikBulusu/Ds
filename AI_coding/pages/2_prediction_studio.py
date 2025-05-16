import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from utils import (get_column_types, handle_missing_values, treat_outliers_iqr,
                   encode_categorical_column, scale_numerical_column, extract_datetime_features,
                   plot_histogram, plot_scatter, plot_boxplot, convert_df_to_csv)
import io # For plot download

# try:
#     import shap # Optional for explainability
#     SHAP_AVAILABLE = True
# except ImportError:
#     SHAP_AVAILABLE = False
SHAP_AVAILABLE = False # Keep it simple for now

st.set_page_config(layout="wide")
st.title("🔮 Prediction Studio")

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a dataset from the main page first.")
    st.stop()

# Use df_processed for all operations on this page
if 'df_processed' not in st.session_state or st.session_state.df_processed is None:
    st.session_state.df_processed = st.session_state.df.copy()
    st.session_state.preprocessing_log = ["Initial data loaded for processing."]


# --- Preprocessing Section ---
st.header("STEP 1: Data Preprocessing")
st.markdown("Apply preprocessing steps to your dataset. Changes will be reflected in the 'Processed Data Preview'.")

# Display preprocessing log
with st.expander("Show Preprocessing Log", expanded=False):
    for log_entry in st.session_state.preprocessing_log:
        st.text(f"- {log_entry}")
    if st.button("Reset Processed Data to Original"):
        st.session_state.df_processed = st.session_state.df.copy()
        st.session_state.preprocessing_log = ["Data reset to original."]
        st.experimental_rerun()

# Refresh column types based on df_processed
proc_numeric_cols, proc_categorical_cols, proc_datetime_cols = get_column_types(st.session_state.df_processed)

# Preprocessing Accordion/Tabs
prep_tabs = st.tabs([
    "🧹 Data Cleaning", "💧 Missing Values", "📈 Outlier Treatment", "📅 Datetime Features",
    "🏷️ Categorical Encoding", "⚖️ Feature Scaling"
])

# --- NEW DATA CLEANING TAB ---
with prep_tabs[0]: # Data Cleaning
    st.subheader("General Data Cleaning Operations")

    # 1. Drop Columns
    st.markdown("**1. Drop Unnecessary Columns**")
    cols_to_drop_options = st.session_state.df_processed.columns.tolist()
    if cols_to_drop_options:
        cols_to_drop = st.multiselect("Select columns to drop:", cols_to_drop_options, key="dc_cols_drop")
        if st.button("Drop Selected Columns", key="apply_dc_drop_cols"):
            if cols_to_drop:
                st.session_state.df_processed.drop(columns=cols_to_drop, inplace=True)
                st.session_state.preprocessing_log.append(f"Dropped columns: {', '.join(cols_to_drop)}.")
                st.experimental_rerun()
            else:
                st.warning("No columns selected to drop.")
    else:
        st.info("No columns available to drop (dataset might be empty).")
    
    st.markdown("---")

    # 2. Remove Duplicate Rows
    st.markdown("**2. Remove Duplicate Rows**")
    if not st.session_state.df_processed.empty:
        num_duplicates = st.session_state.df_processed.duplicated().sum()
        st.write(f"Found {num_duplicates} duplicate rows.")
        if num_duplicates > 0:
            if st.button("Remove Duplicate Rows", key="apply_dc_remove_duplicates"):
                st.session_state.df_processed.drop_duplicates(keep='first', inplace=True)
                st.session_state.preprocessing_log.append(f"Removed {num_duplicates} duplicate rows.")
                st.experimental_rerun()
        else:
            st.info("No duplicate rows found.")
    else:
        st.info("Dataset is empty, cannot check for duplicates.")


    st.markdown("---")

    # 3. Drop Rows with High Percentage of Missing Values
    st.markdown("**3. Drop Rows with High Missing Value Percentage**")
    if not st.session_state.df_processed.empty:
        missing_row_threshold = st.slider("Drop rows if more than X% of their values are missing:", 0, 100, 70, 5, key="dc_missing_row_thresh")
        if st.button("Apply Row Drop based on Missing %", key="apply_dc_drop_rows_na_perc"):
            min_non_na_count = int(st.session_state.df_processed.shape[1] * (1 - missing_row_threshold / 100.0))
            original_rows = st.session_state.df_processed.shape[0]
            st.session_state.df_processed.dropna(axis=0, thresh=min_non_na_count, inplace=True)
            rows_dropped = original_rows - st.session_state.df_processed.shape[0]
            st.session_state.preprocessing_log.append(f"Dropped {rows_dropped} rows with > {missing_row_threshold}% missing values.")
            st.experimental_rerun()
    else:
        st.info("Dataset is empty, cannot drop rows.")
    
    st.markdown("---")

    # 4. Drop Columns with High Percentage of Missing Values
    st.markdown("**4. Drop Columns with High Missing Value Percentage**")
    if not st.session_state.df_processed.empty:
        missing_col_threshold = st.slider("Drop columns if more than X% of their values are missing:", 0, 100, 70, 5, key="dc_missing_col_thresh")
        if st.button("Apply Column Drop based on Missing %", key="apply_dc_drop_cols_na_perc"):
            min_valid_count_col = int(st.session_state.df_processed.shape[0] * (1 - missing_col_threshold / 100.0))
            cols_before_drop = st.session_state.df_processed.columns.tolist()
            st.session_state.df_processed.dropna(axis=1, thresh=min_valid_count_col, inplace=True)
            cols_after_drop = st.session_state.df_processed.columns.tolist()
            cols_dropped_list = [col for col in cols_before_drop if col not in cols_after_drop]
            if cols_dropped_list:
                st.session_state.preprocessing_log.append(f"Dropped {len(cols_dropped_list)} columns with > {missing_col_threshold}% missing values: {', '.join(cols_dropped_list)}.")
            else:
                st.session_state.preprocessing_log.append(f"No columns dropped (all had <= {missing_col_threshold}% missing values).")
            st.experimental_rerun()
    else:
        st.info("Dataset is empty, cannot drop columns.")


    st.markdown("---")
    
    # 5. Basic Data Type Conversion (Optional - Can be complex)
    st.markdown("**5. Attempt Basic Data Type Conversion**")
    st.caption("This attempts to convert object columns that look like numbers into numeric types. Use with caution.")
    if not st.session_state.df_processed.empty:
        object_cols_for_conversion = st.session_state.df_processed.select_dtypes(include='object').columns.tolist()
        if object_cols_for_conversion:
            col_to_convert_type = st.selectbox("Select object column to attempt numeric conversion:", 
                                               ["None"] + object_cols_for_conversion, key="dc_type_convert_col")
            
            if col_to_convert_type and col_to_convert_type != "None":
                if st.button(f"Attempt to Convert '{col_to_convert_type}' to Numeric", key="apply_dc_type_convert"):
                    try:
                        original_column_data = st.session_state.df_processed[col_to_convert_type].copy() # Keep a copy
                        original_type = st.session_state.df_processed[col_to_convert_type].dtype
                        
                        # Attempt conversion, coercing errors to NaN
                        converted_series = pd.to_numeric(st.session_state.df_processed[col_to_convert_type], errors='coerce')
                        
                        # Only update if successful conversion to numeric happened for at least some values
                        if pd.api.types.is_numeric_dtype(converted_series.dtype) and not converted_series.isnull().all():
                            st.session_state.df_processed[col_to_convert_type] = converted_series
                            new_type = st.session_state.df_processed[col_to_convert_type].dtype
                            st.session_state.preprocessing_log.append(f"Converted column '{col_to_convert_type}' from {original_type} to {new_type}. NaNs introduced for unconvertible values.")
                            st.success(f"Column '{col_to_convert_type}' converted to {new_type}. Values that couldn't be converted are now NaN (handle in Missing Values tab).")
                        else:
                            # Revert if conversion failed or resulted in all NaNs
                            st.session_state.df_processed[col_to_convert_type] = original_column_data
                            st.warning(f"Conversion of '{col_to_convert_type}' to numeric did not succeed or resulted in all NaNs. Column reverted to original type/data. Please check column content.")
                        
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error during type conversion for '{col_to_convert_type}': {e}")
                        # Optionally revert on error too
                        # st.session_state.df_processed[col_to_convert_type] = original_column_data
        else:
            st.info("No object columns available for basic numeric conversion attempt.")
    else:
        st.info("Dataset is empty, cannot perform type conversion.")


with prep_tabs[1]: # Missing Values
    st.subheader("Handle Missing Values")
    if not st.session_state.df_processed.empty:
        missing_summary = st.session_state.df_processed.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if not missing_summary.empty:
            st.dataframe(missing_summary.reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'}))
            col_to_handle_na = st.selectbox("Select column to handle missing values", missing_summary.index, key="na_col")
            
            strat_options = ["Drop Row", "Drop Column"] # Drop Column was missing
            current_proc_numeric_cols, current_proc_categorical_cols, _ = get_column_types(st.session_state.df_processed)


            if col_to_handle_na in current_proc_numeric_cols:
                strat_options.extend(["Mean", "Median", "Constant"])
            elif col_to_handle_na in current_proc_categorical_cols: # Only object/category
                strat_options.extend(["Mode", "Constant"])
            # Datetime missing values are tricky, often best handled by domain knowledge or converting to categorical then mode.
            # For simplicity, we'll allow mode/constant if it falls into categorical after type check.
            
            na_strategy = st.selectbox("Select strategy", strat_options, key="na_strat")
            
            constant_val_na = None
            if na_strategy == "Constant":
                if col_to_handle_na in current_proc_numeric_cols:
                     constant_val_na = st.number_input("Enter constant value", value=0.0, key="na_const_num")
                else: # Categorical or other
                     constant_val_na = st.text_input("Enter constant value", value="Unknown", key="na_const_cat")

            if st.button("Apply Missing Value Treatment", key="apply_na"):
                st.session_state.df_processed = handle_missing_values(st.session_state.df_processed, col_to_handle_na, na_strategy, constant_val_na)
                st.session_state.preprocessing_log.append(f"Handled NA in '{col_to_handle_na}' using '{na_strategy}'" + (f" with '{constant_val_na}'" if na_strategy=="Constant" else ""))
                st.experimental_rerun()
        else:
            st.success("No missing values found in the processed dataset.")
    else:
        st.info("Dataset is empty, cannot handle missing values.")


with prep_tabs[2]: # Outlier Treatment
    st.subheader("Treat Outliers (IQR Method for Numeric Columns)")
    current_proc_numeric_cols, _, _ = get_column_types(st.session_state.df_processed) # Refresh numeric cols
    if current_proc_numeric_cols:
        col_to_treat_outlier = st.selectbox("Select numeric column for outlier treatment", current_proc_numeric_cols, key="outlier_col")
        if col_to_treat_outlier and not st.session_state.df_processed.empty:
            if st.session_state.df_processed[col_to_treat_outlier].isnull().all():
                 st.warning(f"Column '{col_to_treat_outlier}' is all NaN. Cannot treat outliers.")
            else:
                st.plotly_chart(plot_histogram(st.session_state.df_processed, col_to_treat_outlier), use_container_width=True)
                
                outlier_treatment_method = st.selectbox("Select treatment method", ["Cap", "Remove"], key="outlier_treat")
                # Advanced: factor, quantiles
                # factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1, key="outlier_factor")

                if st.button("Apply Outlier Treatment", key="apply_outlier"):
                    original_rows = st.session_state.df_processed.shape[0]
                    st.session_state.df_processed = treat_outliers_iqr(st.session_state.df_processed, col_to_treat_outlier, treatment=outlier_treatment_method)
                    rows_after = st.session_state.df_processed.shape[0]
                    st.session_state.preprocessing_log.append(f"Treated outliers in '{col_to_treat_outlier}' using IQR '{outlier_treatment_method}'. Rows changed from {original_rows} to {rows_after}.")
                    st.experimental_rerun()
        elif st.session_state.df_processed.empty:
            st.info("Dataset is empty, cannot treat outliers.")
    else:
        st.info("No numeric columns available for outlier treatment in the current processed data.")


with prep_tabs[3]: # Datetime Features
    st.subheader("Extract Features from Datetime Columns")
    _, _, current_proc_datetime_cols = get_column_types(st.session_state.df_processed) # Refresh datetime cols
    if current_proc_datetime_cols:
        col_to_extract_dt = st.selectbox("Select datetime column", current_proc_datetime_cols, key="dt_col_extract")
        if col_to_extract_dt and not st.session_state.df_processed.empty:
            if st.button(f"Extract Features from {col_to_extract_dt}", key="apply_dt_extract"):
                st.session_state.df_processed = extract_datetime_features(st.session_state.df_processed, col_to_extract_dt)
                st.session_state.preprocessing_log.append(f"Extracted date/time features from '{col_to_extract_dt}' and dropped original.")
                st.experimental_rerun()
        elif st.session_state.df_processed.empty:
             st.info("Dataset is empty, cannot extract datetime features.")
    else:
        st.info("No datetime columns identified for feature extraction in the current processed data.")


with prep_tabs[4]: # Categorical Encoding
    st.subheader("Encode Categorical Columns")
    _, current_proc_categorical_cols, _ = get_column_types(st.session_state.df_processed) # Refresh cat cols

    if current_proc_categorical_cols:
        col_to_encode = st.selectbox("Select categorical column to encode", current_proc_categorical_cols, key="cat_col_encode")
        encoding_method = st.selectbox("Select encoding method", ["Label Encoding", "One-Hot Encoding"], key="cat_encode_method")
        
        if col_to_encode and not st.session_state.df_processed.empty:
            if st.button(f"Encode {col_to_encode}", key="apply_cat_encode"):
                st.session_state.df_processed = encode_categorical_column(st.session_state.df_processed, col_to_encode, encoding_method)
                st.session_state.preprocessing_log.append(f"Encoded '{col_to_encode}' using '{encoding_method}'.")
                st.experimental_rerun()
        elif st.session_state.df_processed.empty:
            st.info("Dataset is empty, cannot encode categorical columns.")

    else:
        st.info("No categorical columns identified for encoding in the current processed data.")


with prep_tabs[5]: # Feature Scaling
    st.subheader("Scale Numerical Columns")
    current_proc_numeric_cols, _, _ = get_column_types(st.session_state.df_processed) # Refresh numeric cols

    if current_proc_numeric_cols:
        col_to_scale = st.selectbox("Select numeric column to scale", current_proc_numeric_cols, key="num_col_scale")
        scaling_method = st.selectbox("Select scaling method", ["StandardScaler", "MinMaxScaler"], key="num_scale_method")

        if col_to_scale and not st.session_state.df_processed.empty:
            if st.session_state.df_processed[col_to_scale].isnull().all():
                 st.warning(f"Column '{col_to_scale}' is all NaN. Cannot scale.")
            elif not pd.api.types.is_numeric_dtype(st.session_state.df_processed[col_to_scale]):
                 st.warning(f"Column '{col_to_scale}' is not numeric. Cannot scale.")
            else:
                if st.button(f"Scale {col_to_scale}", key="apply_num_scale"):
                    st.session_state.df_processed = scale_numerical_column(st.session_state.df_processed, col_to_scale, scaling_method)
                    st.session_state.preprocessing_log.append(f"Scaled '{col_to_scale}' using '{scaling_method}'.")
                    st.experimental_rerun()
        elif st.session_state.df_processed.empty:
            st.info("Dataset is empty, cannot scale numeric columns.")

    else:
        st.info("No numeric columns identified for scaling in the current processed data.")


st.subheader("Processed Data Preview (First 5 Rows)")
if not st.session_state.df_processed.empty:
    st.dataframe(st.session_state.df_processed.head())
    st.write(f"Current shape of processed data: {st.session_state.df_processed.shape}")

    if st.button("Download Processed Data as CSV"):
        csv_processed = convert_df_to_csv(st.session_state.df_processed)
        st.download_button(
            label="Click to Download Processed CSV",
            data=csv_processed,
            file_name="processed_data.csv",
            mime="text/csv",
        )
else:
    st.warning("Processed data is currently empty. Please check your cleaning/preprocessing steps or reset data.")


st.markdown("---")

# --- Variable Selection & EDA for Modeling ---
st.header("STEP 2: Variable Selection & Targeted EDA")

if st.session_state.df_processed.empty:
    st.warning("Processed data is empty. Cannot proceed to variable selection. Please check Step 1 or upload a new dataset.")
    st.stop()

df_model = st.session_state.df_processed.copy()

model_numeric_cols, model_categorical_cols, _ = get_column_types(df_model)

problem_type = st.radio("Select Problem Type:", ["Regression (Predict Numeric)", "Classification (Predict Category)"], horizontal=True, key="problem_type_radio")

target_column = None # Initialize
selected_features = [] # Initialize

if problem_type == "Regression (Predict Numeric)":
    target_col_options = model_numeric_cols
else: # Classification
    # Filter for categorical columns with a reasonable number of unique values for classification
    target_col_options = [
        col for col in model_categorical_cols 
        if 1 < df_model[col].nunique() < 20 #  Suitable for classification (e.g. 2 to 19 classes)
    ]
    if not target_col_options: # Fallback if no ideal categorical columns
        target_col_options = model_categorical_cols 
        if model_categorical_cols:
             st.info("No categorical columns with 2-19 unique values found. Showing all categorical columns. High cardinality targets might be challenging.")


if not target_col_options:
    st.warning(f"No suitable columns for {problem_type.split('(')[0]}. Please check your data or preprocessing steps in Step 1.")
else:
    target_column = st.selectbox("Select Target Variable:", target_col_options, key="target_select")

    if target_column:
        feature_columns_options = [col for col in df_model.columns if col != target_column]
        if not feature_columns_options:
            st.warning("No feature columns available after selecting the target.")
        else:
            st.markdown("**Select Feature Variables:**")
            selected_features = st.multiselect(
                "Choose features:", 
                feature_columns_options, 
                default=feature_columns_options[:min(5, len(feature_columns_options))], 
                key="feature_select"
            )

            if selected_features:
                st.subheader(f"Quick EDA for Target '{target_column}' and Selected Features")
                
                st.markdown(f"**Distribution of Target Variable: {target_column}**")
                if target_column in model_numeric_cols:
                    if not df_model[target_column].isnull().all():
                        st.plotly_chart(plot_histogram(df_model, target_column), use_container_width=True)
                    else: st.warning(f"Target column '{target_column}' is all NaN.")
                elif target_column in model_categorical_cols:
                    if not df_model[target_column].isnull().all():
                        val_counts_target = df_model[target_column].value_counts().reset_index()
                        val_counts_target.columns = [target_column, 'count']
                        fig_bar_target = px.bar(val_counts_target, x=target_column, y='count', title=f"Distribution of {target_column}")
                        st.plotly_chart(fig_bar_target, use_container_width=True)
                    else: st.warning(f"Target column '{target_column}' is all NaN.")


                st.markdown("**Feature vs. Target Analysis**")
                for feature in selected_features:
                    if feature == target_column: continue
                    if df_model[feature].isnull().all() or df_model[target_column].isnull().all():
                        st.caption(f"Skipping plot for '{feature}' vs '{target_column}' as one or both columns are all NaN.")
                        continue

                    plot_title = f"{target_column} vs. {feature}"
                    fig_feature_target = None

                    try:
                        if target_column in model_numeric_cols:
                            if feature in model_numeric_cols:
                                fig_feature_target = plot_scatter(df_model, x_col=feature, y_col=target_column)
                            elif feature in model_categorical_cols:
                                fig_feature_target = plot_boxplot(df_model, x_col=feature, y_col=target_column)
                        
                        elif target_column in model_categorical_cols:
                            if feature in model_numeric_cols:
                                fig_feature_target = plot_boxplot(df_model, x_col=target_column, y_col=feature)
                                plot_title = f"{feature} by {target_column}"
                            elif feature in model_categorical_cols:
                                crosstab_df = pd.crosstab(df_model[feature], df_model[target_column])
                                fig_feature_target = px.imshow(crosstab_df, text_auto=True, aspect="auto", title=f"Crosstab: {feature} vs {target_column}")
                        
                        if fig_feature_target:
                            fig_feature_target.update_layout(title=plot_title)
                            st.plotly_chart(fig_feature_target, use_container_width=True)
                        # else: # This condition might not be needed if checks above are robust
                        #     st.write(f"Cannot auto-generate plot for {feature} vs {target_column} with current settings.")
                    except Exception as e:
                        st.warning(f"Could not generate plot for '{feature}' vs '{target_column}': {e}")
                
                numeric_selected_features = [f for f in selected_features if f in model_numeric_cols and not df_model[f].isnull().all()]
                if problem_type == "Regression (Predict Numeric)" and numeric_selected_features and target_column in model_numeric_cols and not df_model[target_column].isnull().all():
                    corr_data_cols = numeric_selected_features + [target_column]
                    corr_data = df_model[corr_data_cols].corr()
                    if not corr_data.empty and target_column in corr_data:
                        st.subheader(f"Correlation of Numeric Features with Target '{target_column}'")
                        corr_target = corr_data[[target_column]].drop(target_column, errors='ignore').sort_values(by=target_column, ascending=False)
                        if not corr_target.empty:
                            st.dataframe(corr_target)
                            fig_corr_bar = px.bar(corr_target, x=corr_target.index, y=target_column, title=f"Feature Correlation with {target_column}")
                            st.plotly_chart(fig_corr_bar, use_container_width=True)
                        else: st.info("No numeric features to correlate after dropping target or if only target was selected.")

st.markdown("---")

# --- Model Training & Evaluation ---
st.header("STEP 3: Model Training & Evaluation")

if target_column and selected_features:
    X = df_model[selected_features].copy() # Use copy to avoid SettingWithCopyWarning on potential later modifications
    y = df_model[target_column].copy()

    non_numeric_features = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_features) > 0:
        st.error(f"The following selected features are still non-numeric: {list(non_numeric_features)}. Please ensure they are encoded in Step 1.")
        st.stop()
    
    # Final check for NaNs that might have been introduced or missed
    combined_check = pd.concat([X, y], axis=1)
    if combined_check.isnull().values.any():
        rows_before_na_drop = combined_check.shape[0]
        combined_check.dropna(inplace=True)
        rows_after_na_drop = combined_check.shape[0]
        if rows_before_na_drop > rows_after_na_drop:
            st.warning(f"NaNs detected in final features/target for modeling. {rows_before_na_drop - rows_after_na_drop} rows dropped.")
        
        if combined_check.empty:
            st.error("Dataset is empty after final NaN drop for modeling. Cannot proceed.")
            st.stop()
        X = combined_check[selected_features]
        y = combined_check[target_column]

    if X.empty or y.empty:
        st.error("No data available for modeling after selections and final NaN handling. Please check your data and preprocessing steps.")
        st.stop()
    
    # Ensure y for classification is int/str if it's categorical
    if problem_type == "Classification (Predict Category)" and y.dtype == 'object':
        try: # Attempt label encoding if not already numeric (though it should be from preprocessing)
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            st.info("Target variable for classification was label encoded for model training.")
        except Exception as e:
            st.error(f"Could not encode target variable for classification: {e}")
            st.stop()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y if problem_type == "Classification (Predict Category)" and y.nunique() > 1 and len(y) >= y.nunique()*2 else None
    ) # Stratify only if enough samples per class

    model_options = {}
    if problem_type == "Regression (Predict Numeric)":
        model_options = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
            "K-Neighbors Regressor": KNeighborsRegressor(n_jobs=-1),
            "SVR (Support Vector Regressor)": SVR()
        }
    else: # Classification
        model_options = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
            "Random Forest Classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
            "K-Neighbors Classifier": KNeighborsClassifier(n_jobs=-1),
            "SVC (Support Vector Classifier)": SVC(probability=True, random_state=42),
            "Gaussian Naive Bayes": GaussianNB()
        }

    chosen_model_name = st.selectbox("Choose a Model:", list(model_options.keys()), key="model_choice")
    model = model_options[chosen_model_name]

    if st.button(f"Train {chosen_model_name} and Evaluate", key="train_eval"):
        if X_train.empty or y_train.empty:
             st.error("Training data is empty. Cannot train model. Check data and preprocessing.")
             st.stop()
        try:
            with st.spinner(f"Training {chosen_model_name}..."):
                model.fit(X_train, y_train)
            
            if X_test.empty:
                st.warning("Test set is empty. Model trained, but no evaluation possible.")
                y_pred = [] # or handle as appropriate
            else:
                 y_pred = model.predict(X_test)

            st.subheader(f"Performance of {chosen_model_name}")

            if problem_type == "Regression (Predict Numeric)":
                if not X_test.empty:
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R-squared", f"{r2:.3f}")
                    col2.metric("MAE", f"{mae:.3f}")
                    col3.metric("MSE", f"{mse:.3f}")
                    col4.metric("RMSE", f"{rmse:.3f}")

                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions',
                                                 marker=dict(color='blue', opacity=0.7)))
                    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                                 mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')))
                    fig_pred.update_layout(title="Actual vs. Predicted Values", xaxis_title="Actual Values", yaxis_title="Predicted Values")
                    st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    st.success("Model trained successfully. No test data to evaluate.")


            else: # Classification
                if not X_test.empty:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.3f}")

                    st.subheader("Classification Report")
                    report_classes = [str(c) for c in np.unique(np.concatenate((y_test.astype(str), y_pred.astype(str))))]

                    try:
                        report_dict = classification_report(y_test, y_pred, target_names=report_classes, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report_dict).transpose())
                    except ValueError as e_report: # If target_names cause issues
                        st.warning(f"Could not generate full classification report with names ({e_report}). Showing basic report.")
                        st.text(classification_report(y_test, y_pred, zero_division=0))

                    st.subheader("Confusion Matrix")
                    cm_labels = model.classes_ if hasattr(model, 'classes_') else np.unique(y_test)
                    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                       labels=dict(x="Predicted Label", y="True Label"),
                                       x=[str(c) for c in cm_labels],
                                       y=[str(c) for c in cm_labels],
                                       title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.success("Model trained successfully. No test data to evaluate.")


            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importances")
                importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                fig_imp = px.bar(importances, x=importances.values, y=importances.index, orientation='h', title="Feature Importances")
                st.plotly_chart(fig_imp, use_container_width=True)
            elif hasattr(model, 'coef_'):
                coeffs = None
                if model.coef_.ndim == 1: 
                    st.subheader("Coefficients")
                    coeffs = pd.Series(model.coef_, index=X_train.columns).sort_values(ascending=False)
                elif model.coef_.ndim == 2 and model.coef_.shape[0] > 0 :
                     st.subheader("Coefficients (per class)")
                     model_classes_names = [f"Class_{c}" for c in (model.classes_ if hasattr(model,'classes_') else range(model.coef_.shape[0]))]
                     if len(model_classes_names) == model.coef_.shape[0]: # Ensure correct number of class names
                        coeffs_df = pd.DataFrame(model.coef_, columns=X_train.columns, index=model_classes_names)
                        st.dataframe(coeffs_df)
                        avg_abs_coeffs = coeffs_df.abs().mean().sort_values(ascending=False)
                        coeffs = avg_abs_coeffs 
                        st.info("Displaying average absolute coefficient magnitude across classes for bar chart.")
                     else:
                        st.warning("Could not determine class names for coefficients display. Showing raw coefficients array.")
                        st.text(model.coef_)


                if coeffs is not None and not coeffs.empty:
                    fig_coeffs = px.bar(coeffs, x=coeffs.values, y=coeffs.index, orientation='h', title="Feature Coefficients/Importances")
                    st.plotly_chart(fig_coeffs, use_container_width=True)
        except Exception as e_train:
            st.error(f"Error during model training or evaluation: {e_train}")


else:
    st.info("Please select a target variable and at least one feature variable in Step 2 to enable modeling.")

st.sidebar.markdown("---")
st.sidebar.info("Prediction Studio: Preprocess, select features, train, and evaluate simple models.")