def winnow_down_dataframe_on_cat(df, column, target_rows):
    """
    Iteratively reduce the number of rows in a DataFrame based on the 
    minimum value in a specified column until a target number of rows 
    remains.

    This function drops rows with NaN in the target column first. Then, 
    it repeatedly removes rows with the lowest value in the specified 
    column. If removing all rows with the minimum value would drop the 
    DataFrame below the target number of rows, the function will 
    interactively prompt the user to select which rows to remove.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to reduce.
    column : str
        The name of the column to use for determining the lowest value.
    target_rows : int
        The target number of rows to retain in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        The reduced DataFrame containing exactly `target_rows` rows, 
        unless the original DataFrame already had `target_rows` or fewer.

    Notes
    -----
    - This function is interactive if a tie occurs at the final selection 
      stage; the user will be prompted to manually choose rows to remove.
    - Rows removed during tie-breaking must be specified by their 
      DataFrame index in the prompt.
    """
    if len(df) <= target_rows:
        print(f"The DataFrame already has {len(df)} rows or fewer.")
        return df

    # Drop rows with NaN in the specified column
    df = df.dropna(subset=[column])

    while len(df) > target_rows:
        # Find the minimum value
        min_value = df[column].min()

        # Get rows with the minimum value
        min_rows = df[df[column] == min_value]

        if len(df) - len(min_rows) < target_rows:
            # Only interact for the final tie
            print(f"Tie detected for the lowest value ({min_value}) in '{column}'.")
            print("Here are the tied rows with selected values:")
            print(min_rows[['final_num', 'image_description', 'valence_mean', 'valence_std']])
            
            # Prompt user to select rows to remove
            remove_index = input(f"Enter the index(es) of row(s) to remove (comma-separated): ").strip()
            remove_index = [int(i) for i in remove_index.split(',')]

            # Validate indices
            if not set(remove_index).issubset(set(min_rows.index)):
                print("Invalid indices provided. Please try again.")
                continue

            # Drop the selected rows
            df = df.drop(remove_index)
            
        else:
            # Automatically drop the rows with the minimum value
            df = df.drop(min_rows.index)

    return df
def winnow_down_dataframe_on_cat(df, column, target_rows):
    if len(df) <= target_rows:
        print(f"The DataFrame already has {len(df)} rows or fewer.")
        return df

    # Drop rows with NaN in the specified column
    df = df.dropna(subset=[column])

    while len(df) > target_rows:
        # Find the minimum value
        min_value = df[column].min()

        # Get rows with the minimum value
        min_rows = df[df[column] == min_value]

        if len(df) - len(min_rows) < target_rows:
            # Only interact for the final tie
            print(f"Tie detected for the lowest value ({min_value}) in '{column}'.")
            print("Here are the tied rows with selected values:")
            print(min_rows[['final_num', 'image_description', 'valence_mean', 'valence_std']])
            
            # Prompt user to select rows to remove
            remove_index = input(f"Enter the index(es) of row(s) to remove (comma-separated): ").strip()
            remove_index = [int(i) for i in remove_index.split(',')]

            # Validate indices
            if not set(remove_index).issubset(set(min_rows.index)):
                print("Invalid indices provided. Please try again.")
                continue

            # Drop the selected rows
            df = df.drop(remove_index),
            
        else:
            # Automatically drop the rows with the minimum value
            df = df.drop(min_rows.index)

    return df
