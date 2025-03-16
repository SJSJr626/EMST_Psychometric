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