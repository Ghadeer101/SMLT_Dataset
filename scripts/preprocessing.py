
import pandas as pd

# ------------------------------
# Function: Reshape Raw Data
# ------------------------------
def reshape(df):
    '''
    Transpose the DataFrame and set the first row as the new header.

    Parameters:
    df: Raw DataFrame (e.g., loaded from CSV with header=None)

    Returns:
    Transformed DataFrame with corrected orientation and headers.
    '''
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)
    return df

# ------------------------------
# Function: Clean Data (Remove Outliers)
# ------------------------------
def clean_df(df):
    '''
    Remove outliers from the DataFrame using the IQR method and replace them with the median.

    Parameters:
    df: DataFrame of numerical LMP data.

    Returns:
    Cleaned DataFrame with outliers mitigated.
    '''
    df_cleaned = pd.DataFrame()
    multiplier = 1.5  # IQR threshold for outlier definition

    for column_name in df.columns:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        upper_threshold = Q3 + multiplier * IQR
        lower_threshold = Q1 - multiplier * IQR

        # Apply median replacement for outliers
        df_cleaned[column_name] = df.apply(
            lambda row: df[column_name].median() if row[column_name] > upper_threshold or row[column_name] < lower_threshold else row[column_name],
            axis=1
        )
    return df_cleaned

# ------------------------------
# Main Processing Pipeline
# ------------------------------
if __name__ == "__main__":
    # Load raw CSV file (no header)
    df = pd.read_csv('case1_raw.csv', header=None)

    # Step 1: Reshape the raw data
    new_df = reshape(df)

    # Step 2: Separate metadata columns (timestamp and label)
    metadata_df = new_df[['timestamp', 'Label']]

    # Step 3: Remove 'Week' and 'Label' columns for data cleaning
    data_df = new_df.drop(columns=['Week', 'Label'])

    # Step 4: Clean the data
    cleaned_df = clean_df(data_df)

    # Step 5: Reattach metadata columns to cleaned data
    final_df = pd.concat([metadata_df, cleaned_df], axis=1)

    # Step 6: Export cleaned dataset
    final_df.to_csv('case1', index=False)
    print("Cleaned dataset saved as 'case1.csv'")
