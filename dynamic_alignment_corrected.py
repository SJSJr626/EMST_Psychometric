def align_distributions_dynamic_tests(
    df1,
    df2,
    metrics,
    categorical_vars=None,
    scalar_p_threshold=0.50,
    categorical_p_threshold=0.50,
    tol=0.10,
    central_tendency="both",
    ks_threshold=0.10,
    TOST_bounds=0.25,
    max_iterations=100000,
    seed=None,
    enforce_category_limit=True,
    category_limit=0.60,
    category_ceiling_mode="across_sets",
    categorical_swaps_per_iter=3,
    balance_tuples=None,
    balance_column='final_num',
    locked_indices=None,
    lock_prebalanced=False,
    residual_guidance=False,
    early_stop=False,
    stall_limit=200,
    verbose=True,
    debug_flags=False
):
    """
    Align df1 and df2 by swapping rows until scalar and/or categorical 
    features meet user-defined statistical equivalence criteria.

    PILOT LOGIC:
      1. Satisfy scalar tests (p-value, central tendency, KS) 
         then categorical tests (chi-square + category ceiling).
      2. Once all above are satisfied, run TOST equivalence tests 
         across scalar metrics as a final check.
      3. If any TOST fails, do not stop;  nudge with a single swap 
         and continue until TOST passes or max_iterations is reached.

    This script always logs full verbose/debug output to ./align_YYYYmmdd_HHMMSS.log
    (console printing still controlled by `verbose` / `debug_flags`).

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame to be aligned.
    df2 : pandas.DataFrame
        Second DataFrame to be aligned.
    metrics : list
        List of scalar (continuous) metrics to balance. 
        Compulsory, but can be an empty list [] for categorical-only alignment.
    categorical_vars : list, optional
        List of categorical metrics to balance. 
        Can be set to None for scalar-only alignment, by default None.
    scalar_p_threshold : float, optional
        Minimum p-value allowed for any between-group test of distributional similarity.
        Choose a value comfortably above significance (e.g., 0.50) to push distributions 
        further from significant difference, by default 0.50.
    categorical_p_threshold : float, optional
        Minimum p-value allowed for categorical balance tests (chi-square).
        Also should be set above typical significance thresholds, by default 0.50.
    tol : float, optional
        Maximum allowable difference in central tendency between sets 
        (see `central_tendency`), by default 0.10.
    central_tendency : str, optional
        Options: "mean", "median", or "both", by default "both".
    ks_threshold : float, optional
        Maximum allowable Kolmogorov–Smirnov statistic (D) for any scalar metric,
        by default 0.10.
    TOST_bounds : float, optional
        Equivalence bounds for Robust Two One-Sided Tests (TOST) on scalar metrics,
        by default 0.25.
    max_iterations : int, optional
        Maximum iterations before stopping. We recommend 100,000; if no solution 
        is found, try a different random seed, by default 100000.
    seed : int, optional
        Random seed for reproducibility. Different seeds can yield different solutions,
        by default None.
    enforce_category_limit : bool, optional
        If True, enforces a ceiling on the proportion of any single category,
        by default True.
    category_limit : float, optional
        Maximum allowable proportion of a labeled category (see `category_ceiling_mode`),
        by default 0.60.
    category_ceiling_mode : str, optional
        If "across_sets" the proportion is calculated across both sets combined.
        If "within_set" the proportion is calculated within each set alone,
        by default "across_sets".
    categorical_swaps_per_iter : int, optional
        Number of categorical swaps attempted per iteration; impacts alignment speed,
        by default 3.
    balance_tuples : list, optional
        List of tuples specifying pairs of stimuli that must be placed in opposing sets.
        Warning: locks these rows into place, by default None.
    balance_column : str, optional
        Column used to index and enforce pre-balance constraints, by default 'final_num'.
    locked_indices : list, optional
        List of row indices to lock in place (prevent swapping), by default None.
    lock_prebalanced : bool, optional
        If True, automatically locks any rows that already satisfy all criteria at start.
        WARNING: very restrictive, limits solutions, by default False.
    residual_guidance : bool, optional
        If False, swaps are chosen indiscriminately (i.e., completely at random).
        If True, swaps are biased toward stimuli furthest from each set's mean.
        IN BETA TESTING, by default False.
    early_stop : bool, optional
        If True, stops when no improvement is seen for `stall_limit` iterations.
        Falls back to best last solution. Avoids local minima but is currently experimental.
        IN BETA TESTING, by default False.
    stall_limit : int, optional
        Number of iterations with no improvement before early stopping triggers,
        by default 200.
    verbose : bool, optional
        If True, swap breakdowns print to console.
        All verbose/debug output is also logged to ./align_YYYYmmdd_HHMMSS.log,
        by default True.
    debug_flags : bool, optional
        If True, enables additional low-level debug information in the log,
        by default False.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        A tuple containing the aligned df1 and df2 DataFrames.

    Notes
    -----
    It is recommended to also provide categorical variables as scalar transformations 
    (e.g., percentages) when possible and list them in `metrics`. This assists in 
    biasing towards categorical alignment before these swaps and allows for TOST 
    testing for extra assurance.
    """

    import numpy as np
    import pandas as pd
    import logging
    from scipy.stats import shapiro, ttest_ind, mannwhitneyu, ks_2samp, chi2_contingency  # Statistical tests to be run
    from rpy2.robjects import r, FloatVector  # Access to R from Python via r() and conversion of Python lists to R numeric vectors
    import rpy2.robjects.packages as rpackages  # For importing R packages (TOSTER must be installed in R environment) for this pipeline
    from datetime import datetime  # For timestamped log file names
    import os  # For file path handling and directory operations

    # ----- Setup logging with both file and console handlers -----
    log_path = f"align_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger('align_distributions')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    if verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.CRITICAL)  # Suppress console output when not verbose
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"align_distributions_dynamic_tests run @ {datetime.now().isoformat()}")
    logger.info(f"Writing detailed output to ./{os.path.basename(log_path)}")
    
    # Update console handler level based on debug_flags
    if debug_flags:
        console_handler.setLevel(logging.DEBUG)

    # ----- random number generator (RNG) -----
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)  # Automatically pick a random seed
        logger.info(f"Randomly selected seed: {seed}")  # Print RNG seed
    else:
        logger.info(f"Using provided seed: {seed}")  # Print manual seed

    np.random.seed(seed)

    # ----- Make TOST available -----
    try:
        rpackages.importr('TOSTER')
        has_toster = True
    except Exception:
        has_toster = False
        logger.warning("TOSTER is not available; skipping FINAL TOST equivalence checks.")

    # Normalize inputs
    if categorical_vars and not isinstance(categorical_vars, list):
        categorical_vars = [categorical_vars]

    # Lock set
    fixed_indices = set()
    if locked_indices is not None:
        fixed_indices.update(list(locked_indices))

    # ---------- helpers ----------
    def _eligible(idx_iterable):
        return [idx for idx in idx_iterable if idx not in fixed_indices]

    def _swap_rows(dfA, idxA, dfB, idxB):
        tmp = dfA.loc[idxA].copy()
        dfA.loc[idxA] = dfB.loc[idxB]
        dfB.loc[idxB] = tmp

    def _finite_indices(df, metric, pool_indices):
        ser = pd.to_numeric(df.loc[pool_indices, metric], errors="coerce")
        return list(ser.index[np.isfinite(ser.values)])

    def _pick_by_residual(df, metric, target_mean, pool_indices):
        if not pool_indices:
            return None
        ser = pd.to_numeric(df.loc[pool_indices, metric], errors="coerce")
        finite_mask = np.isfinite(ser.values)
        if not finite_mask.any():
            return None
        ser = ser[finite_mask]
        if not np.isfinite(target_mean):
            return np.random.choice(ser.index.values, 1)[0]
        w = np.abs(ser.values - target_mean)
        w[~np.isfinite(w)] = 0.0
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return np.random.choice(ser.index.values, 1)[0]
        p = w / s
        return np.random.choice(ser.index.values, 1, p=p)[0]

    def _objective(dfa, dfb):
        total = 0.0
        count = 0
        for m in metrics:
            x, y = dfa[m].dropna(), dfb[m].dropna()
            ks, _ = ks_2samp(x, y)
            total += abs(x.mean() - y.mean())
            total += abs(x.median() - y.median())
            total += ks
            count += 3
        return total / max(count, 1)

    # Optional pre-pass
    if balance_tuples and balance_column:
        for value in balance_tuples:
            df1_rows = df1[df1[balance_column].isin(value)]
            df2_rows = df2[df2[balance_column].isin(value)]
            while len(df1_rows) > 1:
                j = np.random.choice(df2.index)
                i = df1_rows.index[0]
                _swap_rows(df2, j, df1, i)
                df1_rows = df1[df1[balance_column].isin(value)]
                df2_rows = df2[df2[balance_column].isin(value)]
            while len(df2_rows) > 1:
                i = np.random.choice(df1.index)
                j = df2_rows.index[0]
                _swap_rows(df1, i, df2, j)
                df1_rows = df1[df1[balance_column].isin(value)]
                df2_rows = df2[df2[balance_column].isin(value)]
            if lock_prebalanced:
                fixed_indices.update(df1_rows.index)
                fixed_indices.update(df2_rows.index)

    # Early-stop trackers
    if early_stop:
        best_state = (df1.copy(), df2.copy())
        best_obj = _objective(df1, df2)
        stall = 0

    # ---------- main loop ----------
    for iteration in range(max_iterations):
        logger.info(f"\nIteration {iteration + 1}: evaluating metrics and categorical constraints...")

        scalar_ok_all = True
        chi_square_ok = True
        ceiling_ok    = True

        # ---- Continuous metrics (NO per-metric TOST here) ----
        for metric in metrics:
            x = df1[metric].dropna()
            y = df2[metric].dropna()

            # Choose test by normality
            _, p1 = shapiro(x); _, p2 = shapiro(y)
            if (p1 > 0.05 and p2 > 0.05):
                _, p_val = ttest_ind(x, y, equal_var=False)
                test_used = "T-Test"
            else:
                _, p_val = mannwhitneyu(x, y, alternative='two-sided')
                test_used = "Mann-Whitney U"

            mean_diff   = abs(x.mean()   - y.mean())
            median_diff = abs(x.median() - y.median())
            ks_stat, _  = ks_2samp(x, y)

            if central_tendency == "mean":
                ct_ok = (mean_diff <= tol)
            elif central_tendency == "median":
                ct_ok = (median_diff <= tol)
            else:
                ct_ok = (mean_diff <= tol) and (median_diff <= tol)

            scalar_ok = (p_val > scalar_p_threshold) and ct_ok and (ks_stat <= ks_threshold)

            logger.info(
                f"{metric}: {test_used}, p={p_val:.4f}, |Δmean|={mean_diff:.4f}, "
                f"|Δmedian|={median_diff:.4f}, KS D={ks_stat:.4f}  → "
                f"{'PASS' if scalar_ok else 'FAIL'}"
            )

            if not scalar_ok:
                scalar_ok_all = False
                i_pool = _finite_indices(df1, metric, _eligible(df1.index))
                j_pool = _finite_indices(df2, metric, _eligible(df2.index))
                if residual_guidance:
                    i = _pick_by_residual(df1, metric, df2[metric].dropna().mean(), i_pool)
                    j = _pick_by_residual(df2, metric, df1[metric].dropna().mean(), j_pool)
                else:
                    i = np.random.choice(i_pool, 1)[0] if i_pool else None
                    j = np.random.choice(j_pool, 1)[0] if j_pool else None
                if i is not None and j is not None:
                    _swap_rows(df1, i, df2, j)

        # ---- Categorical variables ----
        if categorical_vars:
            for categorical_var in categorical_vars:
                f1 = df1[categorical_var].value_counts().sort_index()
                f2 = df2[categorical_var].value_counts().sort_index()
                cats = sorted(set(f1.index) | set(f2.index))
                f1 = f1.reindex(cats, fill_value=0)
                f2 = f2.reindex(cats, fill_value=0)
                ct = pd.DataFrame({'Set1': f1, 'Set2': f2}).T
                logger.info(f"\n{categorical_var} contingency:\n{ct}")

                _, p_cat, _, _ = chi2_contingency(ct)
                if p_cat < categorical_p_threshold:
                    chi_square_ok = False
                    logger.info(f"Chi-square p={p_cat:.4f} < {categorical_p_threshold} → needs adjustment.")
                else:
                    logger.info(f"Chi-square p={p_cat:.4f} ≥ {categorical_p_threshold} → OK.")

                if enforce_category_limit:
                    if category_ceiling_mode == "within_set":
                        row_totals = ct.sum(axis=1).replace(0, 1)
                        props = ct.div(row_totals, axis=0)
                        for row in props.index:
                            over_mask = props.loc[row] > category_limit
                            if over_mask.any():
                                ceiling_ok = False
                                over_cat = props.loc[row, over_mask].idxmax()
                                other = "Set2" if row == "Set1" else "Set1"
                                under_mask = ~over_mask
                                if under_mask.any():
                                    under_cat = props.loc[other, under_mask].idxmin()
                                    swaps_done = 0
                                    while swaps_done < categorical_swaps_per_iter:
                                        if row == "Set1":
                                            src = _eligible(df1[df1[categorical_var] == over_cat].index)
                                            dst = _eligible(df2[df2[categorical_var] == under_cat].index)
                                        else:
                                            src = _eligible(df2[df2[categorical_var] == over_cat].index)
                                            dst = _eligible(df1[df1[categorical_var] == under_cat].index)
                                        if not src or not dst:
                                            if swaps_done == 0:
                                                logger.info(f"No partner category available to balance {over_cat} in {row}.")
                                            break
                                        i = np.random.choice(src, 1)[0]
                                        j = np.random.choice(dst, 1)[0]
                                        if row == "Set1":
                                            _swap_rows(df1, i, df2, j)
                                        else:
                                            _swap_rows(df2, i, df1, j)
                                        swaps_done += 1
                                        logger.info(f"Swapped {categorical_var}: {row}[{over_cat}] ↔ {other}[{under_cat}] (≤{category_limit:.0%}).")

                    elif category_ceiling_mode == "across_sets":
                        col_totals = ct.sum(axis=0).replace(0, 1)
                        shares = ct.div(col_totals, axis=1)
                        for row in shares.index:
                            over_mask = shares.loc[row] > category_limit
                            if over_mask.any():
                                ceiling_ok = False
                                over_cat = shares.loc[row, over_mask].idxmax()
                                other = "Set2" if row == "Set1" else "Set1"
                                under_cat = shares.loc[other].idxmax()
                                if under_cat == over_cat and len(shares.columns) > 1:
                                    under_cat = shares.loc[other].drop(over_cat).idxmax()
                                swaps_done = 0
                                while swaps_done < categorical_swaps_per_iter:
                                    if row == "Set1":
                                        src = _eligible(df1[df1[categorical_var] == over_cat].index)
                                        dst = _eligible(df2[df2[categorical_var] == under_cat].index)
                                    else:
                                        src = _eligible(df2[df2[categorical_var] == over_cat].index)
                                        dst = _eligible(df1[df1[categorical_var] == under_cat].index)
                                    if not src or not dst:
                                        if swaps_done == 0:
                                            logger.info(f"No available swap partners for {categorical_var} in {row} (over={over_cat}, target={under_cat}).")
                                        break
                                    i = np.random.choice(src, 1)[0]
                                    j = np.random.choice(dst, 1)[0]
                                    if row == "Set1":
                                        _swap_rows(df1, i, df2, j)
                                    else:
                                        _swap_rows(df2, i, df1, j)
                                    swaps_done += 1
                                    logger.info(f"Swapped {categorical_var}: {row}[{over_cat}] ↔ {other}[{under_cat}] (no-hoarding ≤{category_limit:.0%}).")
                    else:
                        raise ValueError("category_ceiling_mode must be 'across_sets' or 'within_set'")

        # ---- progress / early-stop bookkeeping ----
        if early_stop:
            cur_obj = _objective(df1, df2)
            if cur_obj + 1e-12 < best_obj:
                best_obj = cur_obj
                best_state = (df1.copy(), df2.copy())
                stall = 0
            else:
                stall += 1

        # ---- FINAL CHECK ORDER (pilot): TOST last ----
        gates_ok = scalar_ok_all and chi_square_ok and ceiling_ok
        if gates_ok:
            if has_toster:
                logger.info("\nAll gates passed; running FINAL TOST checks...")
                all_eq = True
                for metric in metrics:
                    x = df1[metric].dropna().values
                    y = df2[metric].dropna().values
                    try:
                        r_res = r['wilcox_TOST'](
                            FloatVector(x), FloatVector(y),
                            low_eqbound=-TOST_bounds, high_eqbound=TOST_bounds,
                            paired=False, alpha=0.05
                        )
                        p_lo = float(r_res.rx2('TOST')[1][1])
                        p_hi = float(r_res.rx2('TOST')[1][2])
                        eq_ok = (p_lo < 0.05 and p_hi < 0.05)
                        logger.info(
                            f"  TOST {metric}: p_lower={p_lo:.4f}, p_upper={p_hi:.4f} → "
                            f"{'EQUIVALENT' if eq_ok else 'NOT EQUIVALENT'}"
                        )
                        if not eq_ok:
                            all_eq = False
                    except Exception as e:
                        logger.info(f"  TOST {metric} failed: {e} → treating as NOT EQUIVALENT")
                        all_eq = False

                if all_eq:
                    logger.info("\nAll metrics equivalent on FINAL TOST. Stopping.")
                    return df1, df2
                else:
                    i_pool = _eligible(df1.index); j_pool = _eligible(df2.index)
                    if i_pool and j_pool:
                        i = np.random.choice(i_pool, 1)[0]
                        j = np.random.choice(j_pool, 1)[0]
                        _swap_rows(df1, i, df2, j)
                    logger.info("Final TOST failed for at least one metric → continuing to next iteration.")
            else:
                logger.info("\nAll gates passed (TOST unavailable). Stopping.")
                return df1, df2

        # ---- early stop ----
        if early_stop and stall > stall_limit:
            logger.info(f"\nEarly stop: no progress in {stall_limit} iterations. Restoring best state.")
            df1, df2 = best_state
            return df1, df2

    # for-else: hit max_iterations without breaking
    logger.info("\nReached max_iterations without full alignment.")
    return df1, df2
    
