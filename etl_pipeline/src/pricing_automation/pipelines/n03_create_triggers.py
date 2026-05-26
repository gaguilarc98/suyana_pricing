from .utils import *


########____GENERATE ACCUMULATED ANOMALIES____########
 
 
def generate_accumulated_anomalies(
    df_orig,
    window,
    time_dim='time',
    group_cols=['climate_area_id'],
    variable_name='swc_adjusted',
    set_variable='index_value'
):
    """Accumulate variable into seasonal windows. Returns one row per
    (group, window_year, window) with 'flag_complete_window' (1=full)."""
    df           = df_orig.copy()
    df[time_dim] = pd.to_datetime(df[time_dim])
    month_day    = df[time_dim].dt.strftime('%m-%d')
    year         = df[time_dim].dt.year
 
    df_total = []
 
    for code, lapse in window.items():
        start_month, start_day = map(int, lapse[0].split('-'))
        end_month,   end_day   = map(int, lapse[1].split('-'))
        crosses_year = (end_month < start_month) or (
            end_month == start_month and end_day < start_day
        )
 
        if crosses_year:
            in_window   = (month_day >= lapse[0]) | (month_day <= lapse[1])
            window_year = np.where(month_day <= lapse[1], year - 1, year)
        else:
            in_window   = (month_day >= lapse[0]) & (month_day <= lapse[1])
            window_year = year
 
        if not in_window.any():
            continue
 
        df_window                = df[in_window].copy()
        df_window['window']      = code
        df_window['window_year'] = window_year[in_window]
        df_window['start_date']  = pd.to_datetime(
            df_window['window_year'].astype(str) + '-' + lapse[0]
        )
        df_window['end_date'] = pd.to_datetime(
            (df_window['window_year'] + crosses_year).astype(str) + '-' + lapse[1]
        )
        df_total.append(df_window)
 
    df_total = pd.concat(df_total, axis=0, ignore_index=True)
 
    df_cumulated = df_total.groupby(
        group_cols + ['window_year', 'window', 'start_date', 'end_date'],
        as_index=False
    ).agg(**{
        'first_date':      (time_dim, 'min'),
        'last_date':       (time_dim, 'max'),
        f'{set_variable}': (variable_name, lambda x: np.round(np.sum(x), 4))
    })
 
    df_cumulated['index_desc'] = f'cum_{variable_name}'
    df_cumulated['first_date'] = pd.to_datetime(df_cumulated['first_date'].dt.date)
    df_cumulated['last_date']  = pd.to_datetime(df_cumulated['last_date'].dt.date)
 
    df_cumulated = df_cumulated[
        ~(df_cumulated['first_date'] > df_cumulated['start_date'])
    ].copy()
    df_cumulated = df_cumulated.drop(columns=['first_date']).reset_index(drop=True)
 
    df_cumulated['flag_complete_window'] = np.where(
        df_cumulated['last_date'] < df_cumulated['end_date'], 0, 1
    )
 
    return df_cumulated
 
 
########____STAGE 2: FITTED DISTRIBUTION CLASS____########
 
 
class FittedDistribution:
    """Per-group fitted distribution (parametric or empirical).
 
    Attributes: dist (str), df_fit (DataFrame, one row per group),
    group_cols (list). Serialise with pickle.
    """
 
    def __init__(self, dist: str, df_fit: pd.DataFrame, group_cols: list):
        self.dist       = dist
        self.df_fit     = df_fit
        self.group_cols = group_cols
 
    # ------------------------------------------------------------------ #
    #  Class method: fit from a df_cum DataFrame                          #
    # ------------------------------------------------------------------ #
 
    @classmethod
    def fit(cls, df_cum: pd.DataFrame, group_cols: list, variable: str, dist: str = 'gamma'):
        """Fit distribution per group using complete windows only.
        dist: 'norm','t','lognorm','weibull','gamma','norm_inf','gev','beta','empirical'.
        """
        if 'flag_complete_window' in df_cum.columns:
            df_in = df_cum.query('flag_complete_window==1').copy()
        else:
            df_in = df_cum.copy()
 
        records = []
        for group_key, group_df in df_in.groupby(group_cols):
            x   = group_df[variable].values
            rec = dict(zip(
                group_cols,
                group_key if isinstance(group_key, tuple) else [group_key]
            ))
 
            if dist == 'norm':
                rec.update({'loc': np.mean(x), 'scale': np.std(x, ddof=1)})
 
            elif dist == 't':
                rec.update({'loc': np.mean(x), 'scale': np.std(x, ddof=1), 'df': len(x) - 1})
 
            elif dist == 'lognorm':
                s, loc, scale = stats.lognorm.fit(x, floc=-1)
                rec.update({
                    's':     s,      # std of log(x)
                    'loc':   loc,    # shift away from zero
                    'scale': scale   # exp(mean of log(x))
                })
 
            elif dist == 'weibull':
                shape, loc, scale = stats.weibull_min.fit(x, floc=0)
                rec.update({'shape': shape, 'loc': loc, 'scale': scale})
 
            elif dist == 'gamma':
                mu, sigma = np.mean(x), np.std(x, ddof=1)
                rec.update({'shape': mu**2 / sigma**2, 'loc': 0, 'scale': sigma**2 / mu})
 
            elif dist == 'norm_inf':
                rec.update({'loc': np.mean(x), 'scale': np.std(x, ddof=1) * 1.04})
 
            elif dist == 'gev':
                shape, loc, scale = stats.genextreme.fit(x)
                rec.update({'shape': shape, 'loc': loc, 'scale': scale})
 
            elif dist == 'beta':
                mx, sx = np.mean(x), np.std(x, ddof=0)
                common = mx * (1 - mx) / sx**2 - 1
                rec.update({'a': mx * common, 'b': (1 - mx) * common, 'loc': 0, 'scale': 1})
 
            elif dist == 'empirical':
                rec.update({'ref_values': np.sort(x).tolist()})
 
            else:
                raise ValueError(f"Unsupported distribution: {dist}")
 
            records.append(rec)
 
        return cls(dist=dist, df_fit=pd.DataFrame(records), group_cols=group_cols)
 
    # ------------------------------------------------------------------ #
    #  CDF and PPF: operate on a single df_fit row                        #
    # ------------------------------------------------------------------ #
 
    def _cdf_row(self, row, x):
        """CDF at scalar x for a single df_fit row. Empirical uses np.interp
        consistent with np.nanpercentile. Returns float in [0, 1]."""
        d = self.dist
        if d == 'norm':
            return stats.norm.cdf(x, loc=row['loc'], scale=row['scale'])
        elif d == 't':
            return stats.t.cdf(x, loc=row['loc'], scale=row['scale'], df=row['df'])
        elif d == 'lognorm':
            return stats.lognorm.cdf(x, s=row['s'], loc=row['loc'], scale=row['scale'])
        elif d == 'weibull':
            return stats.weibull_min.cdf(x, c=row['shape'], loc=row['loc'], scale=row['scale'])
        elif d == 'gamma':
            return stats.gamma.cdf(x, a=row['shape'], loc=row['loc'], scale=row['scale'])
        elif d == 'norm_inf':
            return stats.norm.cdf(x, loc=row['loc'], scale=row['scale'])
        elif d == 'gev':
            return stats.genextreme.cdf(x, row['shape'], loc=row['loc'], scale=row['scale'])
        elif d == 'empirical':
            ref    = np.asarray(row['ref_values'], dtype=float)
            p_grid = np.linspace(0, 100, len(ref))
            return float(np.interp(x, ref, p_grid) / 100)
        return np.nan
 
    def _ppf_row(self, row, p_frac):
        """Inverse CDF at p_frac in [0,1] for a single df_fit row.
        Empirical inverts np.nanpercentile via np.interp."""
        d = self.dist
        if d == 'norm':
            return stats.norm.ppf(p_frac, loc=row['loc'], scale=row['scale'])
        elif d == 't':
            return stats.t.ppf(p_frac, loc=row['loc'], scale=row['scale'], df=row['df'])
        elif d == 'lognorm':
            return stats.lognorm.ppf(p_frac, s=row['s'], loc=row['loc'], scale=row['scale'])
        elif d == 'weibull':
            return stats.weibull_min.ppf(p_frac, row['shape'], loc=row['loc'], scale=row['scale'])
        elif d == 'gamma':
            return stats.gamma.ppf(p_frac, row['shape'], loc=row['loc'], scale=row['scale'])
        elif d == 'norm_inf':
            return stats.norm.ppf(p_frac, loc=row['loc'], scale=row['scale'])
        elif d == 'gev':
            return stats.genextreme.ppf(p_frac, row['shape'], loc=row['loc'], scale=row['scale'])
        elif d == 'empirical':
            ref    = np.asarray(row['ref_values'], dtype=float)
            p_grid = np.linspace(0, 100, len(ref))
            return float(np.interp(p_frac * 100, p_grid, ref))
        return np.nan
 
    # ------------------------------------------------------------------ #
    #  Public method: assign percentiles to a DataFrame                   #
    # ------------------------------------------------------------------ #
 
    def compute_percentiles(self, x: pd.DataFrame, variable: str = 'index_value') -> np.ndarray:
        """Compute percentiles (0-1) for each row in x by merging with df_fit
        and evaluating the CDF per group. Out-of-sample clips to [0, 1]."""
        missing = set(self.group_cols) - set(x.columns)
        if missing:
            raise ValueError(
                f"x is missing group columns required for merging: {missing}"
            )
 
        df = x.copy().merge(self.df_fit, on=self.group_cols, how='left')
 
        return np.round(
            df.apply(lambda row: self._cdf_row(row, row[variable]), axis=1).values, 4
        )
 
 
########____STAGE 3: CONTRACT VALIDATION AND PAYOUT SCHEDULE____########
 
 
# ---- low-level payout functions ----
 
def _continuous_payout(p, start_pct, exit_pct, max_payout=1.0, min_payout=0.0, tail='lower'):
    """Linear-ramp payout in [0,100] percentile space.
    lower: start_pct > exit_pct. upper: start_pct < exit_pct."""
    p = np.asarray(p, dtype=float)
    if tail == 'lower':
        slope = (max_payout - min_payout) / (start_pct - exit_pct)
        ramp  = min_payout + slope * (start_pct - p)
        return np.where(p > start_pct, 0.0, np.where(p <= exit_pct, max_payout, ramp))
    elif tail == 'upper':
        slope = (max_payout - min_payout) / (exit_pct - start_pct)
        ramp  = min_payout + slope * (p - start_pct)
        return np.where(p < start_pct, 0.0, np.where(p >= exit_pct, max_payout, ramp))
    else:
        raise ValueError("tail must be 'lower' or 'upper'")
 
 
def _layered_payout(p, levels, tail='lower'):
    """Step-function payout. levels: {percentile: coverage_fraction}."""
    p          = np.asarray(p, dtype=float)
    payout     = np.zeros_like(p)
    level_list = [(int(k), v) for k, v in levels.items()]
 
    if tail == 'lower':
        for threshold, fraction in sorted(level_list, key=lambda x: x[0], reverse=True):
            payout = np.where(p <= threshold, fraction, payout)
    elif tail == 'upper':
        for threshold, fraction in sorted(level_list, key=lambda x: x[0]):
            payout = np.where(p >= threshold, fraction, payout)
    else:
        raise ValueError("tail must be 'lower' or 'upper'")
 
    return payout
 
 
# ---- contract validation ----
 
def _validate_tail(tail_spec, crop_name, window_key, tail_label,
                   parent_coverage, parent_iv, parent_tlr, parent_ded):
    """Validate and normalise a tail spec. Fills defaults from parent values.
    continuous: levels=[p1,p2], assigns start/exit_pct by tail direction.
    layered: levels must be a non-empty dict."""
    if tail_spec is None:
        raise ValueError(
            f"Crop '{crop_name}', window {window_key}, tail '{tail_label}' "
            "has no specification."
        )
 
    design = tail_spec.get('design')
    if design is None:
        raise ValueError(
            f"Crop '{crop_name}', window {window_key}, tail '{tail_label}' "
            "must specify 'design'."
        )
    if design not in ('continuous', 'layered'):
        raise ValueError(
            f"Crop '{crop_name}', window {window_key}, tail '{tail_label}': "
            f"design must be 'continuous' or 'layered', got '{design}'."
        )
 
    levels = tail_spec.get('levels')
 
    if design == 'continuous':
        if levels is None:
            raise ValueError(
                f"Crop '{crop_name}', window {window_key}, tail '{tail_label}' "
                "(continuous) requires 'levels'."
            )
        if not isinstance(levels, (list, tuple)):
            raise ValueError(
                f"Crop '{crop_name}', window {window_key}, tail '{tail_label}' "
                "(continuous) 'levels' must be a list or tuple."
            )
        if len(levels) != 2:
            raise ValueError(
                f"Crop '{crop_name}', window {window_key}, tail '{tail_label}' "
                f"(continuous) 'levels' must have exactly 2 values, got {len(levels)}."
            )
        lo, hi = min(levels), max(levels)
        if tail_label == 'lower':
            tail_spec['start_pct'] = hi
            tail_spec['exit_pct']  = lo
        else:  # upper
            tail_spec['start_pct'] = lo
            tail_spec['exit_pct']  = hi
        tail_spec.setdefault('max_payout', 1.0)
        tail_spec.setdefault('min_payout', 0.0)
 
    elif design == 'layered':
        if not isinstance(levels, dict) or not levels:
            raise ValueError(
                f"Crop '{crop_name}', window {window_key}, tail '{tail_label}' "
                "(layered) 'levels' must be a non-empty dict."
            )
 
    tail_spec.setdefault('coverage',          parent_coverage)
    tail_spec.setdefault('insured_value',     parent_iv)
    tail_spec.setdefault('target_loss_ratio', parent_tlr)
    tail_spec.setdefault('deductions',        parent_ded)
 
 
def _validate_contract(params_contract, valid_window_keys):
    """Validate params_contract and fill defaults. Cascade priority (high->low):
    tail > window > crop > top-level > default (coverage=1.0, insured_value=1).
    Raises ValueError on unknown window keys or missing required fields.
    """
    import copy
    specs = copy.deepcopy(params_contract)
 
    specs.setdefault('target_loss_ratio', 0.74)
    specs.setdefault('deductions',        0.14)
    specs.setdefault('coverage',          1.0)
    specs.setdefault('insured_value',     1)
 
    crops = specs.get('crops', {})
    if not crops:
        raise ValueError("params_contract must contain a non-empty 'crops' dict.")
 
    TAIL_KEYS = {'upper', 'lower'}
 
    for crop_name, crop_spec in crops.items():
        if crop_spec is None:
            raise ValueError(f"Crop '{crop_name}' has no specification.")
 
        crop_spec.setdefault('coverage',      specs['coverage'])
        crop_spec.setdefault('insured_value', specs['insured_value'])
 
        windows = crop_spec.get('windows')
        if not windows:
            raise ValueError(f"Crop '{crop_name}' must specify a non-empty 'windows' dict.")
        if not isinstance(windows, dict):
            raise ValueError(
                f"Crop '{crop_name}': 'windows' must be a dict "
                f"{{window_key: {{tail_specs}}}}."
            )
 
        # Cross-validate window keys
        unknown = set(windows.keys()) - set(valid_window_keys)
        if unknown:
            raise ValueError(
                f"Crop '{crop_name}' references unknown window key(s): {unknown}. "
                f"Valid keys: {set(valid_window_keys)}."
            )
 
        for window_key, window_spec in windows.items():
            if window_spec is None:
                raise ValueError(
                    f"Crop '{crop_name}', window {window_key} has no specification."
                )
 
            window_spec.setdefault('coverage',      crop_spec['coverage'])
            window_spec.setdefault('insured_value', crop_spec['insured_value'])
 
            present_tails = TAIL_KEYS.intersection(window_spec.keys())
            if not present_tails:
                raise ValueError(
                    f"Crop '{crop_name}', window {window_key} must define at least "
                    "one of: 'upper', 'lower'."
                )
 
            for tail_label in present_tails:
                _validate_tail(
                    tail_spec     = window_spec[tail_label],
                    crop_name     = crop_name,
                    window_key    = window_key,
                    tail_label    = tail_label,
                    parent_coverage = window_spec['coverage'],
                    parent_iv       = window_spec['insured_value'],
                    parent_tlr      = specs['target_loss_ratio'],
                    parent_ded      = specs['deductions'],
                )
 
    return specs
 
 
 
# ---- policy table (trigger thresholds) ----
 
def _build_policy_table(fit, group_cols, params_contract, index_desc):
    """Build df_policy. One row per (group, crop, window, tail, bound/level).
    Uses fit._ppf_row for trigger values -- works for all dist types including
    empirical, since ref_values are stored in fit.df_fit at fit time."""
    crops   = params_contract['crops']
    records = []
 
    for _, row in fit.df_fit.iterrows():
        group_rec = row[group_cols].to_dict()
        window    = group_rec.get('window')
 
        for crop_name, crop_spec in crops.items():
            if window not in crop_spec['windows']:
                continue
            window_spec = crop_spec['windows'][window]
 
            for tail_label in ('upper', 'lower'):
                if tail_label not in window_spec:
                    continue
                tail_spec = window_spec[tail_label]
                design    = tail_spec['design']
                coverage  = tail_spec['coverage']
                iv        = tail_spec['insured_value']
                base      = {**group_rec, 'crop': crop_name, 'tail': tail_label}
 
                if design == 'continuous':
                    for bound, pct, frac in [
                        ('start', tail_spec['start_pct'], tail_spec['min_payout']),
                        ('exit',  tail_spec['exit_pct'],  tail_spec['max_payout']),
                    ]:
                        records.append({
                            **base,
                            'pct':             pct,
                            'bound':           bound,
                            'trigger_value':   np.round(fit._ppf_row(row, pct / 100), 4),
                            'payout_fraction': frac,
                            'coverage':        coverage,
                            'insured_value':   iv,
                            'index_desc':      index_desc,
                        })
 
                elif design == 'layered':
                    for pct_int, fraction in tail_spec['levels'].items():
                        records.append({
                            **base,
                            'pct':             int(pct_int),
                            'bound':           'level',
                            'trigger_value':   np.round(
                                fit._ppf_row(row, int(pct_int) / 100), 4
                            ),
                            'payout_fraction': fraction,
                            'coverage':        coverage,
                            'insured_value':   iv,
                            'index_desc':      index_desc,
                        })
 
    return pd.DataFrame(records)
 
 
# ---- payout schedule ----
 
def compute_payout_schedule(df_cum, group_cols, params_contract):
    """Compute per-year payouts and pricing for each (crop, window, tail).
    Rate columns are pure fractions (no IV scaling). total_payout and
    total_premium multiply by coverage * insured_value at the final step.
    Only complete windows (flag_complete_window==1) are used.
    """
    crops = params_contract['crops']
 
    if 'flag_complete_window' in df_cum.columns:
        df = df_cum.query('flag_complete_window==1').copy()
    else:
        df = df_cum.copy()
 
    df['_pct100'] = df['percentile'] * 100
    base_cols     = group_cols + [
        'window_year', 'start_date', 'end_date',
        'index_value', 'index_desc', 'percentile', '_pct100'
    ]
    df_base       = df[base_cols].copy()
 
    chunks = []
 
    for crop_name, crop_spec in crops.items():
        for window_key, window_spec in crop_spec['windows'].items():
            df_win = df_base[df_base['window'] == window_key].copy()
            if df_win.empty:
                continue
 
            df_win['crop'] = crop_name
 
            for tail_label in ('upper', 'lower'):
                if tail_label not in window_spec:
                    continue
 
                tail_spec    = window_spec[tail_label]
                design       = tail_spec['design']
                tlr          = tail_spec['target_loss_ratio']
                ded          = tail_spec['deductions']
                coverage = tail_spec['coverage']
                iv       = tail_spec['insured_value']
 
                df_tail           = df_win.copy()
                df_tail['tail']   = tail_label
                df_tail['design'] = design
 
                # ---- perc_payout: raw fraction, no IV ----
                if design == 'continuous':
                    start_pct = tail_spec['start_pct']
                    exit_pct  = tail_spec['exit_pct']
                    max_p     = tail_spec['max_payout']
                    min_p     = tail_spec['min_payout']
 
                    df_tail['perc_payout'] = np.round(
                        _continuous_payout(
                            df_tail['_pct100'], start_pct, exit_pct,
                            max_p, min_p, tail_label
                        ), 10
                    )
                    # Area under the ramp (trapezoid):
                    #   rectangle height min_p over the activation probability,
                    #   plus triangle (max_p - min_p) over half that probability.
                    # lower: activation prob = start_pct / 100 (left tail)
                    # upper: activation prob = (100 - start_pct) / 100 (right tail)
                    if tail_label == 'lower':
                        act_prob         = start_pct / 100
                        exit_prob        = exit_pct  / 100
                        ramp_width       = act_prob - exit_prob
                        pure_premium_pct = (
                            min_p * act_prob
                            + (max_p - min_p) * (act_prob + exit_prob) / 2
                        )
                    else:
                        act_prob         = (100 - start_pct) / 100
                        exit_prob        = (100 - exit_pct)  / 100
                        ramp_width       = act_prob - exit_prob
                        pure_premium_pct = (
                            min_p * act_prob
                            + (max_p - min_p) * (act_prob + exit_prob) / 2
                        )
 
                elif design == 'layered':
                    levels = tail_spec['levels']
                    df_tail['perc_payout'] = np.round(
                        _layered_payout(df_tail['_pct100'], levels, tail_label), 10
                    )
                    level_list = [(int(k), v) for k, v in levels.items()]
                    if tail_label == 'lower':
                        sorted_levels = sorted(level_list, key=lambda x: x[0], reverse=True)
                        fractions     = [0.0] + [f for _, f in sorted_levels]
                        thresholds    = [t / 100 for t, _ in sorted_levels] + [0.0]
                    else:
                        sorted_levels = sorted(level_list, key=lambda x: x[0])
                        fractions     = [0.0] + [f for _, f in sorted_levels]
                        thresholds    = [(1 - t / 100) for t, _ in sorted_levels] + [0.0]
                    pure_premium_pct = sum(
                        (fractions[i + 1] - fractions[i]) * thresholds[i]
                        for i in range(len(sorted_levels))
                    )
 
                # ---- pricing: coverage/IV-agnostic rates ----
                df_tail['pure_premium_pct']  = np.round(pure_premium_pct, 10)
                df_tail['target_loss_ratio'] = tlr
                df_tail['re_premium_pct']    = np.round(pure_premium_pct / tlr, 10)
                df_tail['deductions']        = ded
                df_tail['gross_premium_pct'] = np.round(
                    df_tail['re_premium_pct'] * (1 + ded), 10
                )
 
                # ---- reference scalars + absolute columns ----
                df_tail['coverage']      = coverage
                df_tail['insured_value'] = iv
                df_tail['total_premium'] = np.round(
                    df_tail['gross_premium_pct'] * coverage * iv, 2
                )
                df_tail['total_payout'] = np.round(
                    df_tail['perc_payout'] * coverage * iv, 2
                )
 
                chunks.append(df_tail)
 
    if not chunks:
        raise ValueError(
            "No rows matched any (crop, window, tail) combination in params_contract. "
            "Check that window keys in params_contract crops exist in "
            "params_trigger['windows']."
        )
 
    df_payouts = pd.concat(chunks, axis=0, ignore_index=True)
    df_payouts = df_payouts.drop(columns=['_pct100'])
 
    non_group = ['crop', 'window_year', 'tail', 'design',
                 'start_date', 'end_date',
                 'index_value', 'index_desc', 'percentile', 'perc_payout',
                 'pure_premium_pct', 'target_loss_ratio', 're_premium_pct',
                 'deductions', 'gross_premium_pct',
                 'coverage', 'insured_value',
                 'total_premium', 'total_payout']
    col_order = group_cols + non_group
    return df_payouts[[c for c in col_order if c in df_payouts.columns]]
 
 
########____STAGE 4: ORCHESTRATORS____########
 
 
def create_index_values(
    df_orig: pd.DataFrame,
    params_trigger: dict,
):
    """Accumulate variable into windows, fit distribution, assign percentiles.
    Returns (df_cum, fit). df_cum has one row per (group, window_year, window)
    with 'percentile' added. fit is a FittedDistribution instance.
 
    params_trigger keys: variable, windows, group_cols, dist, start_time.
    """
    variable   = params_trigger['variable']
    windows    = params_trigger['windows']
    group_cols = list(params_trigger.get('group_cols', ['location_id']))
    dist       = params_trigger.get('dist', 'empirical')
    start_time = params_trigger.get('start_time', None)
 
    df_hist         = df_orig.copy()
    df_hist['time'] = pd.to_datetime(df_hist['time'])
 
    if start_time:
        df_hist = df_hist[df_hist['time'] >= pd.to_datetime(start_time)].copy()
 
    df_cum = generate_accumulated_anomalies(
        df_hist,
        window=windows,
        group_cols=group_cols,
        variable_name=variable,
        set_variable='index_value',
    )
 
    groups = group_cols + ['window']
    fit    = FittedDistribution.fit(df_cum, groups, 'index_value', dist)
    df_cum['percentile'] = fit.compute_percentiles(df_cum, variable='index_value')
 
    return df_cum, fit
 
 
def generate_payout_policy(
    df_cum: pd.DataFrame,
    fit,
    params_request: dict,
    params_trigger: dict,
    params_contract: dict,
):
    """Compute payout and pricing schedule from pre-fitted percentiles.
 
    Returns (df_payouts, df_policy).
      df_payouts : one row per (group, crop, window_year, window, tail).
      df_policy  : trigger thresholds with coverage, insured_value, index_desc.
    lead_id from params_request is prepended to both if provided.
    Window keys in params_contract are cross-validated before any computation.
    """
    group_cols = list(params_trigger.get('group_cols', ['location_id']))
    lead_id    = params_request.get('lead_id', None)
    index_desc = f"cum_{params_trigger['variable']}"
 
    # Cross-validate window keys before any computation
    valid_window_keys = set(params_trigger['windows'].keys())
    params_contract   = _validate_contract(params_contract, valid_window_keys)
 
    groups = group_cols + ['window']
 
    df_policy = _build_policy_table(fit, groups, params_contract, index_desc)
 
    df_payouts = compute_payout_schedule(df_cum, groups, params_contract)
 
    if lead_id is not None:
        df_policy.insert(0, 'lead_id', lead_id)
        df_payouts.insert(0, 'lead_id', lead_id)
 
    return df_payouts, df_policy