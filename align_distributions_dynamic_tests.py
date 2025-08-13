def align_distributions_dynamic_tests(
    df1, df2,  # DataFrames to be aligned.
    metrics,  # List of scalar (continuous) metrics to balance. 
              # Compulsory, but can be an empty list [] for categorical-only alignment.
    categorical_vars=None,  # List of categorical metrics to balance. 
                             # Can be set to None for scalar-only alignment.

    # ----- Thresholds for statistical alignment to reduce signficant diffrences  -----
    scalar_p_threshold=0.50,  # Minimum p-value allowed for any between-group test of distributional similarity.
                              # Choose a value comfortably above significance (e.g., 0.50) to push distributions further from significant difference.
    categorical_p_threshold=0.50,  # Minimum p-value allowed for categorical balance tests (chi-square).
                                   # Also should be set above typical significance thresholds.
    tol=0.10,  # Maximum allowable difference in central tendency between sets (see `central_tendency`).
    central_tendency="both",  # Options: "mean", "median", or "both".
    ks_threshold=0.10,  # Maximum allowable Kolmogorov–Smirnov statistic (D) for any scalar metric.
    TOST_bounds=0.25,  # Equivalence bounds for Robust Two One-Sided Tests (TOST) on scalar metrics.

    # ----- Loop control -----
    max_iterations=100000,  # Maximum iterations before stopping. 
                            # We recommend 100,000; if no solution is found, try a different random seed.
    seed=None,  # Random seed for reproducibility. Different seeds can yield different solutions.

    # ----- Categorical constraints -----
    # It is recommended to also provide categorical variables as scalar transformations (e.g., percentages) when possible
    # and list them in `metrics`. This assists in biasing towards categorical alignment before these swaps
    # and allows for TOST testing for extra assurance
    enforce_category_limit=True,  # If True, enforces a ceiling on the proportion of any single category.
    category_limit=0.60,  # Maximum allowable proportion of a labeled category (see `category_ceiling_mode`).
    category_ceiling_mode="across_sets",  # if "across_sets" the proportion is calculated across both sets combined.
                                          # if "within_set" the proportion is calculated within each set alone.
    categorical_swaps_per_iter=3,  # Number of categorical swaps attempted per iteration; impacts alignment speed.

    # ----- Optional pre-pass -----
    balance_tuples=None,  # List of tuples specifying pairs of stimuli that must be placed in opposing sets.
                          # Warning: locks these rows into place.
    balance_column='final_num',  # Column used to index and enforce pre-balance constraints.

    # ----- Locking -----
    locked_indices=None,  # List of row indices to lock in place (prevent swapping).
    lock_prebalanced=False,  # If True, automatically locks any rows that already satisfy all criteria at start.
                             # WARNING: very restrictive, limits solutions

    # ----- IN BETA TESTING: Swap guidance for scalar metrics-----
    residual_guidance=False,  # If False, swaps are chosen indiscriminately (i.e., completely at random).
                              # If True, swaps are biased toward stimuli furthest from each set's mean.


    # ----- IN BETA TESTING: Early-stop controls -----
    early_stop=False,  # If True, stops when no improvement is seen for `stall_limit` iterations.
                       # Falls back to best last solution. Avoids local minima but is currently experimental.
    stall_limit=200,  # Number of iterations with no improvement before early stopping triggers.

    # ----- Logging & debugging -----
    verbose=True,  # If True, swap breakdowns print to console.
                   # All verbose/debug output is also logged to ./align_YYYYmmdd_HHMMSS.log.
    debug_flags=False  # If True, enables additional low-level debug information in the log.
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
    """

    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro, ttest_ind, mannwhitneyu, ks_2samp, chi2_contingency  # Statistical tests to be run
    from rpy2.robjects import r, FloatVector  # Access to R from Python via r() and conversion of Python lists to R numeric vectors
    import rpy2.robjects.packages as rpackages  # For importing R packages (TOSTER must be installed in R environment) for this pipeline
    from datetime import datetime  # For timestamped log file names
    import os  # For file path handling and directory operations

    # ----- file logger (always on, current directory) -----
    log_path = f"align_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = open(log_path, "w", encoding="utf-8") #set file handeling (fh) variable
    fh.write(f"### align_distributions_dynamic_tests run @ {datetime.now().isoformat()}\n")
    fh.flush() #for immediate writing 

    def _log(msg: str):
        fh.write(msg + "\n")
        fh.flush()

    def _v(msg: str):
        if verbose:
            print(msg)
        _log(msg)

    def _dbg(msg: str):
        if debug_flags:
            print(msg)
        _log(msg)

    _v(f"[log] Writing detailed output to ./{os.path.basename(log_path)}")

    # ----- random number generator (RNG) -----
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)  # Automatically pick a random seed
        _v(f"[INFO] Randomly selected seed: {seed}") # Print RNG seed
    else:
        _v(f"[INFO] Using provided seed: {seed}") # Print manual seed

    np.random.seed(seed)


    try:
        # ----- Make TOST availabile -----
        try:
            rpackages.importr('TOSTER')
            has_toster = True
        except Exception:
            has_toster = False
            _v("Warning: TOSTER is not available; skipping FINAL TOST equivalence checks.")

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
            _v(f"\nIteration {iteration + 1}: evaluating metrics and categorical constraints...")

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

                _v(f"{metric}: {test_used}, p={p_val:.4f}, |Δmean|={mean_diff:.4f}, "
                   f"|Δmedian|={median_diff:.4f}, KS D={ks_stat:.4f}  → "
                   f"{'PASS' if scalar_ok else 'FAIL'}")

                if not scalar_ok:
                    scalar_ok_all = False
                    # Swap (residual-guided or uniform) to push toward alignment; still in testing
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
                    # proceed; stop condition won't be set this round

            # ---- Categorical variables ----
            if categorical_vars:
                for categorical_var in categorical_vars:
                    f1 = df1[categorical_var].value_counts().sort_index()
                    f2 = df2[categorical_var].value_counts().sort_index()
                    cats = sorted(set(f1.index) | set(f2.index))
                    f1 = f1.reindex(cats, fill_value=0)
                    f2 = f2.reindex(cats, fill_value=0)
                    ct = pd.DataFrame({'Set1': f1, 'Set2': f2}).T
                    _v(f"\n{categorical_var} contingency:\n{ct}")

                    _, p_cat, _, _ = chi2_contingency(ct)
                    if p_cat < categorical_p_threshold:
                        chi_square_ok = False
                        _v(f"Chi-square p={p_cat:.4f} < {categorical_p_threshold} → needs adjustment.")
                    else:
                        _v(f"Chi-square p={p_cat:.4f} ≥ {categorical_p_threshold} → OK.")

                    # Category ceiling
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
                                                    _v(f"No partner category available to balance {over_cat} in {row}.")
                                                break
                                            i = np.random.choice(src, 1)[0]
                                            j = np.random.choice(dst, 1)[0]
                                            if row == "Set1":
                                                _swap_rows(df1, i, df2, j)
                                            else:
                                                _swap_rows(df2, i, df1, j)
                                            swaps_done += 1
                                            _v(f"Swapped {categorical_var}: {row}[{over_cat}] ↔ {other}[{under_cat}] (≤{category_limit:.0%}).")

                        elif category_ceiling_mode == "across_sets":
                            col_totals = ct.sum(axis=0).replace(0, 1)
                            shares = ct.div(col_totals, axis=1)
                            for row in shares.index:
                                over_mask = shares.loc[row] > category_limit
                                if over_mask.any():
                                    ceiling_ok = False
                                    over_cat = shares.loc[row, over_mask].idxmax()
                                    other = "Set2" if row == "Set1" else "Set1"
                                    # pick a prominent category on the other side
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
                                                _v(f"No available swap partners for {categorical_var} in {row} (over={over_cat}, target={under_cat}).")
                                            break
                                        i = np.random.choice(src, 1)[0]
                                        j = np.random.choice(dst, 1)[0]
                                        if row == "Set1":
                                            _swap_rows(df1, i, df2, j)
                                        else:
                                            _swap_rows(df2, i, df1, j)
                                        swaps_done += 1
                                        _v(f"Swapped {categorical_var}: {row}[{over_cat}] ↔ {other}[{under_cat}] (no-hoarding ≤{category_limit:.0%}).")
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
                    _v("\nAll gates passed; running FINAL TOST checks...")
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
                            _v(f"  TOST {metric}: p_lower={p_lo:.4f}, p_upper={p_hi:.4f} → "
                               f"{'EQUIVALENT' if eq_ok else 'NOT EQUIVALENT'}")
                            if not eq_ok:
                                all_eq = False
                        except Exception as e:
                            _v(f"  TOST {metric} failed: {e} → treating as NOT EQUIVALENT")
                            all_eq = False

                    if all_eq:
                        _v("\nAll metrics equivalent on FINAL TOST. Stopping.")
                        break
                    else:
                        # one swap nudge to escape plateau; then continue loop
                        i_pool = _eligible(df1.index); j_pool = _eligible(df2.index)
                        if i_pool and j_pool:
                            i = np.random.choice(i_pool, 1)[0]
                            j = np.random.choice(j_pool, 1)[0]
                            _swap_rows(df1, i, df2, j)
                        _v("Final TOST failed for at least one metric → continuing to next iteration.")
                else:
                    _v("\nAll gates passed (TOST unavailable). Stopping.")
                    break

            # ---- early stop ----
            if early_stop and stall > stall_limit:
                _v(f"\nEarly stop: no progress in {stall_limit} iterations. Restoring best state.")
                df1, df2 = best_state
                break

        else:
            _v("\nReached max_iterations without full alignment.")

        return df1, df2

    finally:
        fh.close()
