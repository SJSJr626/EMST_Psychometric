def align_distributions_dynamic_tests(
    df1, df2, metrics, categorical_vars=None, threshold=0.1, tol=0.2, max_iterations=10000, seed=random_seed, 
    enforce_category_limit=True, category_limit=0.60, balance_tuples=balance_images_custom, balance_column="final_num"
):
    np.random.seed(seed)

    # Ensure categorical_vars is a list
    if categorical_vars and not isinstance(categorical_vars, list):
        categorical_vars = [categorical_vars]

    # Initialize fixed_indices
    fixed_indices = set()

    # Balance tuples between DataFrames
    if balance_tuples and balance_column:
        for value in balance_tuples:
            df1_rows = df1[df1[balance_column].isin(value)]
            df2_rows = df2[df2[balance_column].isin(value)]

            # Balance rows between DataFrames
            while len(df1_rows) > 1:
                random_idx_df2 = np.random.choice(df2.index)
                random_row_df2 = df2.loc[random_idx_df2].copy()

                idx_to_swap = df1_rows.index[0]
                df2.loc[random_idx_df2] = df1.loc[idx_to_swap]
                df1.loc[idx_to_swap] = random_row_df2

                # Re-check rows
                df1_rows = df1[df1[balance_column].isin(value)]
                df2_rows = df2[df2[balance_column].isin(value)]

            while len(df2_rows) > 1:
                random_idx_df1 = np.random.choice(df1.index)
                random_row_df1 = df1.loc[random_idx_df1].copy()

                idx_to_swap = df2_rows.index[0]
                df1.loc[random_idx_df1] = df2.loc[idx_to_swap]
                df2.loc[idx_to_swap] = random_row_df1

                # Re-check rows
                df1_rows = df1[df1[balance_column].isin(value)]
                df2_rows = df2[df2[balance_column].isin(value)]

            # Add rows to fixed_indices
            if len(df1_rows) > 0:
                fixed_indices.update(df1_rows.index)
            if len(df2_rows) > 0:
                fixed_indices.update(df2_rows.index)

    # Iterative alignment
    for iteration in range(max_iterations):
        all_non_significant = True
        print(f"\nIteration {iteration + 1}: Checking all metrics and categorical variables...")

        # Check numerical metrics
        for metric in metrics:
            stat1, pval1 = shapiro(df1[metric].dropna())
            stat2, pval2 = shapiro(df2[metric].dropna())
            normality = pval1 > 0.05 and pval2 > 0.05

            if normality:
                stat, p_val = ttest_ind(df1[metric].dropna(), df2[metric].dropna(), equal_var=False)
                test_used = "T-Test"
            else:
                stat, p_val = mannwhitneyu(df1[metric].dropna(), df2[metric].dropna(), alternative='two-sided')
                test_used = "Mann-Whitney U Test"

            mean_diff = abs(df1[metric].mean() - df2[metric].mean())
            median_diff = abs(df1[metric].median() - df2[metric].median())

            print(f"{metric}: {test_used}, P-value = {p_val:.4f}, Mean Diff = {mean_diff:.4f}, Median Diff = {median_diff:.4f}")

            if p_val > threshold and mean_diff <= tol and median_diff <= tol:
                print(f"{metric} is aligned (P-value > {threshold}, Mean Diff <= {tol}, Median Diff <= {tol}).")
            else:
                all_non_significant = False
                print(f"{metric} is not aligned. Adjusting distributions...")

                # Randomly swap rows, excluding fixed indices
                idx1 = np.random.choice([idx for idx in df1.index if idx not in fixed_indices], 1)[0]
                idx2 = np.random.choice([idx for idx in df2.index if idx not in fixed_indices], 1)[0]

                temp_row = df1.loc[idx1].copy()
                df1.loc[idx1] = df2.loc[idx2]
                df2.loc[idx2] = temp_row

        # Check categorical variables
        if categorical_vars:
            for categorical_var in categorical_vars:
                freq1 = df1[categorical_var].value_counts().sort_index()
                freq2 = df2[categorical_var].value_counts().sort_index()

                all_categories = sorted(set(freq1.index).union(set(freq2.index)))
                freq1 = freq1.reindex(all_categories, fill_value=0)
                freq2 = freq2.reindex(all_categories, fill_value=0)

                contingency_table = pd.DataFrame({'T2': freq1, 'T3': freq2}).T
                print(contingency_table)
                
                # Compute Chi-Square test
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)

                if p_val < threshold:
                    print(f"Chi-square test (p = {p_val:.4f}) indicates a need for adjustment.")
                    # Continue with proportion enforcement and swapping logic
                else:
                    print(f"Chi-square test (p = {p_val:.4f}) shows no significant difference. Skipping adjustment.")
                                
                # Proportion enforcement with targeted swapping
                if enforce_category_limit:
                    total_counts = contingency_table.sum(axis=0)
                    proportions = contingency_table.div(total_counts, axis=1)
                    over_limit = (proportions > category_limit).any(axis=0)

                    if over_limit.any():
                        # Identify the most overrepresented category
                        max_exceeding_category = proportions.loc[:, over_limit].idxmax(axis=1).iloc[0]

                        # Identify the least represented category if possible
                        underrepresented_categories = proportions.loc[:, ~over_limit]

                        if not underrepresented_categories.empty:
                            min_underrepresented_category = underrepresented_categories.idxmin(axis=1).iloc[0]

                            # Find indices for these categories
                            t2_problem_idx = df1[df1[categorical_var] == max_exceeding_category].index
                            t3_problem_idx = df2[df2[categorical_var] == max_exceeding_category].index
                            t2_under_idx = df1[df1[categorical_var] == min_underrepresented_category].index
                            t3_under_idx = df2[df2[categorical_var] == min_underrepresented_category].index

                            # Ensure valid indices exist for swapping
                            if not t2_problem_idx.empty and not t3_under_idx.empty:
                                idx1 = np.random.choice(t2_problem_idx, 1)[0] 
                                idx2 = np.random.choice(t3_under_idx, 1)[0] 

                                # Swap rows
                                temp_row = df1.loc[idx1].copy()
                                df1.loc[idx1] = df2.loc[idx2]
                                df2.loc[idx2] = temp_row

                            elif not t3_problem_idx.empty and not t2_under_idx.empty:
                                idx1 = np.random.choice(t3_problem_idx, 1)[0]  
                                idx2 = np.random.choice(t2_under_idx, 1)[0]  

                                # Swap rows
                                temp_row = df2.loc[idx1].copy()
                                df2.loc[idx1] = df1.loc[idx2]
                                df1.loc[idx2] = temp_row

                            else:
                                print(f"Skipping category {max_exceeding_category} due to insufficient swap candidates.")

                        else:
                            print(f"No underrepresented categories available to balance {max_exceeding_category}. Continuing to next iteration.")

        if all_non_significant:
            print("\nAll metrics and categorical variables are aligned. Stopping.")
            break

    return df1, df2