import numpy as np
import os

def split_dataframe(df, fraction=None, sample_size=None, stratify_column=None, 
                    save_directory=None, seed=None, file_format='csv'):
    """
    Split a DataFrame into stratified subsets based on a fraction or sample size,
    with an optional feature to save the subsets to a directory.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - fraction (float, optional): Fraction of the data to include in each split. 
                                  Defaults to None.
    - sample_size (int, optional): Number of rows to include in each split. 
                                   Defaults to None.
    - stratify_column (str, optional): Column name to stratify the splits by. 
                                       Must be present in the DataFrame. Defaults to None.
    - save_directory (str, optional): Directory path to save the subsets. 
                                      Defaults to None.
    - seed (int, optional): Random seed for reproducibility. Defaults to None.
    - file_format (str, optional): File format for saving (e.g., 'csv', 'pickle', 'excel'). 
                                   Defaults to 'csv'.

    Returns:
    - list of pd.DataFrames: A list of stratified subsets of the original DataFrame.
    """

    # Check if both fraction and sample_size are provided
    if fraction is not None and sample_size is not None:
        raise ValueError("Cannot provide both 'fraction' and 'sample_size'. Choose one.")

    # Check if neither fraction nor sample_size is provided
    if fraction is None and sample_size is None:
        raise ValueError("Must provide either 'fraction' or 'sample_size'.")

    # Check if stratify_column is provided but not present in the DataFrame
    if stratify_column is not None and stratify_column not in df.columns:
        raise ValueError(f"'{stratify_column}' is not a column in the DataFrame.")

    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Calculate the number of splits based on the fraction or sample size
    if fraction is not None:
        sample_size = int(len(df) * fraction)
        num_splits = int(np.floor(len(df) / sample_size))
        remaining_rows = len(df) % sample_size
    else:
        num_splits = int(np.floor(len(df) / sample_size))
        remaining_rows = len(df) % sample_size

    # Initialize an empty list to store the subsets
    subsets = []

    # Stratified Split
    if stratify_column is not None:
        remaining_df = df.copy()  # Create a copy to avoid modifying the original df
        subsets = []
        while len(remaining_df) >= sample_size:
            # Group by stratify column and sample proportionally
            stratified_sample = remaining_df.groupby(stratify_column, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(remaining_df)))))
            )
            
            # Ensure we don't exceed sample size
            if len(stratified_sample) > sample_size:
                stratified_sample = stratified_sample.sample(sample_size)
            
            subsets.append(stratified_sample)
            
            # Remove sampled rows using index
            remaining_df = remaining_df.drop(stratified_sample.index)
            
        # Handle remaining rows if any
        if len(remaining_df) > 0:
            subsets.append(remaining_df)
    else:
        # Non-stratified split
        df_shuffled = df.sample(frac=1, random_state=seed)
        subsets = []
        for i in range(num_splits):
            start_idx = i * sample_size
            end_idx = start_idx + sample_size
            if start_idx < len(df_shuffled):
                subset = df_shuffled.iloc[start_idx:min(end_idx, len(df_shuffled))]
                subsets.append(subset)

    # Save subsets to directory if specified
    if save_directory:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        for i, subset in enumerate(subsets):
            filename = f"dataset_{i+1}_subset"
            if file_format == 'csv':
                subset.to_csv(os.path.join(save_directory, f"{filename}.csv"), index=False)
            elif file_format == 'pickle':
                subset.to_pickle(os.path.join(save_directory, f"{filename}.pickle"))
            elif file_format == 'excel':
                subset.to_excel(os.path.join(save_directory, f"{filename}.xlsx"), index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

    return subsets