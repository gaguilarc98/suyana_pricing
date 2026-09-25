from .utils import *

from .a02_create_clusters import *

GROUP_COLS = ['location_id']

#———————————————————————————————————————————
# REGISTER LEAD LOCATION DATASET
#———————————————————————————————————————————


def register_lead_locations(
    data,
    params: dict
) -> gpd.GeoDataFrame:
    """Reconstructs gdf_locations from a pd.DataFrame or gpd.GeoDataFrame,
    filling in any columns not already present (fallback values from
    params). Reuses an existing 'location_id' column if it already maps
    to geometries 1-to-1, otherwise reassigns it."""
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
        is_points = (gdf.geom_type == 'Point').all()
        if 'exposure_weight' not in gdf.columns and not is_points:
            gdf['exposure_weight'] = get_area_column(gdf, 'ha')
        gdf = gdf.to_crs(epsg=4326)
        centroids = gdf.geometry.centroid
        gdf['lon'] = centroids.x
        gdf['lat'] = centroids.y
        #gdf = gpd.GeoDataFrame(
        #    gdf.drop(columns='geometry'),
        #    geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']),
        #    crs='EPSG:4326'
        #)
    else:
        gdf = gpd.GeoDataFrame(
            data.copy(),
            geometry=gpd.points_from_xy(data['lon'], data['lat']),
            crs='EPSG:4326'
        )

    if 'cod_lead' not in gdf.columns:
        gdf['cod_lead'] = params['lead_id']

    if 'cod_sub_lead' not in gdf.columns:
        width = len(str(len(gdf)))
        gdf['cod_sub_lead'] = [
            f"{params['lead_id'].upper()}-{str(i + 1).zfill(width)}" for i in range(len(gdf))
        ]

    for col in ['value_density', 'exposure_weight', 'target_premium', 'coverage_pct', 'crop', 'window', 'peril']:
        if col not in gdf.columns:
            gdf[col] = params.get(col)

    if 'value_unit' not in gdf.columns:
        gdf['value_unit'] = params.get('value_unit', 'USD/ha')

    if 'total_value' not in gdf.columns:
        gdf['total_value'] = gdf['value_density'] * gdf['exposure_weight']

    if 'location_id' in gdf.columns:
        pairs = gdf[['geometry', 'location_id']].drop_duplicates()
        keep_location_id = pairs['geometry'].is_unique and pairs['location_id'].is_unique
    else:
        keep_location_id = False

    if keep_location_id:
        gdf_aux = pairs
    else:
        pad_width = len(str(len(gdf)))
        gdf_aux = gdf[['geometry']].drop_duplicates()
        gdf_aux['location_id'] = [
            f"ID-{str(i + 1).zfill(max(pad_width, 3))}" for i in range(len(gdf_aux))
        ]
    gdf = gdf.drop(columns=['location_id'], errors='ignore').merge(gdf_aux, how='left', on=['geometry'])

    col_order = [
        'location_id', 'cod_lead', 'cod_sub_lead', 'lon', 'lat',
        'value_density', 'exposure_weight', 'total_value', 'value_unit',
        'target_premium', 'coverage_pct', 'crop', 'window', 'peril','geometry'
    ]
    return gdf[col_order]


#———————————————————————————————————————————
# SUMMARIZE GROUPED DATASET
#———————————————————————————————————————————


def summarize_processed_data(
    ds_orig : xr.Dataset,
    gdf : gpd.GeoDataFrame,
    params_process: dict,
):
    """Groups the original pixel values into climate area values."""
    variable = params_process['variable']
    add_abs_anom = params_process.get('add_abs_anom', True)
    mode = params_process.get('summarize_mode', 'within')
    k_neighbors = params_process.get('k_neighbors', 1)

    # Resolve dict-style variable (Planet config) to the value present in dataset
    if isinstance(variable, dict):
        variable = next(
            (v for v in variable.values() if v in ds_orig.data_vars),
            list(variable.values())[0]
        )

    ds = ds_orig.copy()
    LOCATION_NAME = 'location_id'
    gdf_aoi = gdf.drop_duplicates(subset='geometry')
    gdf_aoi = gdf_aoi.to_crs(epsg='4326')

    def nearest_method(ds, gdf_aoi, LOCATION_NAME, check_var, k):
        lon, lat, time =  get_coordinates(ds)
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME, method='nearest', check_var=check_var, k=k)
        ds_sum = summarize_data(ds_clean, group_coords=['points'])
        print(f"Nearest neighbors selection done")

        df_sum = ds_sum.reset_index('points').to_dataframe()
        df_sum = df_sum.reset_index().drop(columns='points', errors='ignore')

        # Round to avoid float-precision mismatches between df_clusters
        # (numpy source) and df_sum (xarray reset_index source)
        _PREC = 6
        df_key = df_clusters[[lon, lat, LOCATION_NAME]].copy()
        df_key[lon] = df_key[lon].round(_PREC)
        df_key[lat] = df_key[lat].round(_PREC)
        if lon in df_sum.columns:
            df_sum[lon] = df_sum[lon].round(_PREC)
        if lat in df_sum.columns:
            df_sum[lat] = df_sum[lat].round(_PREC)

        merge_on = [c for c in [lon, lat] if c in df_sum.columns]
        df_sum = df_key.merge(
            df_sum,
            how='right',
            on=merge_on
        ).drop(columns=[lon, lat], errors='ignore')

        return df_sum, df_clusters

    if mode == 'nearest':
        df_sum, df_clusters = nearest_method(ds, gdf_aoi, LOCATION_NAME, variable, k=k_neighbors)
        print(f"Clustering with nearest neighbor strategy done")

    elif mode == 'within':
        # v2 is faster since it applies a spatial join
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME)

        # No pixel falls within any polygon: grouping on an all-NaN coord would fail
        if bool(ds_clean[LOCATION_NAME].isnull().all()):
            print("No pixels within polygons, applying nearest strategy for all locations")
            df_sum, df_clusters = nearest_method(ds, gdf_aoi, LOCATION_NAME, variable, k=k_neighbors)
        else:
            ds_sum = summarize_data(ds_clean, group_coords=[LOCATION_NAME])
            print(f"Clustering and summarizing done using within strategy")

            df_sum = ds_sum.to_dataframe().reset_index()
            df_sum = df_sum.groupby(LOCATION_NAME).filter(lambda x: x[variable].count() != 0)

            # Locations left out of 'within' due to their size get a nearest-neighbor fallback
            list_out = list(set(gdf_aoi[LOCATION_NAME].values) - set(df_sum[LOCATION_NAME].unique()))
            if len(list_out) > 0:
                print(f"Applying nearest strategy for {len(list_out)} polygons")
                gdf_out = gdf_aoi[gdf_aoi[LOCATION_NAME].isin(list_out)]

                df_sum_out, df_clusters_out = nearest_method(ds, gdf_out, LOCATION_NAME, variable, k=k_neighbors)

                df_sum = pd.concat([df_sum, df_sum_out], axis=0, ignore_index=True)
                df_clusters = pd.concat([df_clusters, df_clusters_out], axis=0, ignore_index=True)
    
    elif mode == 'pixel':
        ds_clean, df_clusters = create_cluster_coord(ds, gdf_aoi, LOCATION_NAME, method='within')
        lon, lat, time =  get_coordinates(ds_clean)
        df_sum = ds_clean.to_dataframe().reset_index()
        df_sum = df_sum.dropna(subset=[LOCATION_NAME])
        df_sum['pixel_id'] = (
            np.round(df_sum[lon].values, 3).astype(str)
            + '_'
            + np.round(df_sum[lat].values, 3).astype(str)
        )
    
    if add_abs_anom:
        df_sum[f'neg_anom_{variable}'] = np.where(
            df_sum[f'anom_{variable}'] >=0, 0, df_sum[f'anom_{variable}'] * (-1)
        )
        df_sum[f'pos_anom_{variable}'] = np.where(
            df_sum[f'anom_{variable}'] <=0, 0, df_sum[f'anom_{variable}']
        )

    return df_sum, df_clusters


########____GENERATE ACCUMULATED ANOMALIES____########


MAP_VARS = {
    'intensity_cold_spell': ('cum_intensity_cold_spell', 'n_cold_spell'),
    'intensity_heat_wave': ('cum_intensity_heat_wave', 'n_heat_wave'),
    'intensity_strong_wind': ('cum_intensity_strong_wind', 'n_strong_wind'),
}

def flag_episode_severity(df, cum_col='cum_intensity_heat_wave', n_col='n_heat_wave',
                           min_duration=2, group_cols=GROUP_COLS, var_name='episode_intensity'):
    df = df.sort_values(group_cols + ['time']).copy()

    next_n = df.groupby(group_cols)[n_col].shift(-1)
    is_episode_end = next_n.fillna(0) == 0          # NaN at series end treated as "ended"
    qualifies = df[n_col] >= min_duration

    mask = is_episode_end & qualifies

    df[var_name] = np.where(mask, df[cum_col], 0.0)
    return df
 
 
def generate_accumulated_anomalies(
    df_orig,
    window,
    time_dim='time',
    group_cols=GROUP_COLS,
    variable_name='swc_adjusted',
    set_variable='index_value',
    duration=3
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
    if variable_name in MAP_VARS.keys():
        intensity, count = MAP_VARS[variable_name]
        df_total = flag_episode_severity(
            df_total, 
            intensity, 
            count, 
            group_cols = group_cols, 
            var_name = variable_name,
            min_duration = duration,
        )
 
    df_cumulated = df_total.groupby(
        group_cols + ['window_year', 'window', 'start_date', 'end_date'],
        as_index=False
    ).agg(**{
        'first_date':      (time_dim, 'min'),
        'last_date':       (time_dim, 'max'),
        f'{set_variable}': (variable_name, lambda x: np.round(np.sum(x), 4))
    })
 
    period_by_window = {
        code: (
            pd.Timestamp(2000, *map(int, lapse[0].split('-'))).strftime('%b %d')
            + ' - ' +
            pd.Timestamp(2000, *map(int, lapse[1].split('-'))).strftime('%b %d')
        )
        for code, lapse in window.items()
    }
    df_cumulated['period']      = df_cumulated['window'].map(period_by_window)
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
    """Per-group fitted distribution (parametric or empirical). Also
    carries peril/index_desc metadata from create_index_values so
    downstream stages don't need to re-derive them. Serialise with pickle."""

    def __init__(self, dist: str, df_fit: pd.DataFrame, group_cols: list,
                 peril: str = None, index_desc: str = None):
        self.dist       = dist
        self.df_fit     = df_fit
        self.group_cols = group_cols
        self.peril      = peril
        self.index_desc = index_desc

    @classmethod
    def fit(cls, df_cum: pd.DataFrame, group_cols: list, variable: str, dist: str = 'gamma',
            peril: str = None, index_desc: str = None):
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
 
        return cls(dist=dist, df_fit=pd.DataFrame(records), group_cols=group_cols,
                   peril=peril, index_desc=index_desc)
 
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
    level_list = [(float(k), v) for k, v in levels.items()]
 
    if tail == 'lower':
        for threshold, fraction in sorted(level_list, key=lambda x: x[0], reverse=True):
            payout = np.where(p <= threshold, fraction, payout)
    elif tail == 'upper':
        for threshold, fraction in sorted(level_list, key=lambda x: x[0]):
            payout = np.where(p >= threshold, fraction, payout)
    else:
        raise ValueError("tail must be 'lower' or 'upper'")
 
    return payout


########____GENERATE OUTPUT____########


# ---- contract validation ----

def _validate_monotonic_levels(levels, tail_label, window_key):
    """Check that payout fractions move consistently with percentile keys.
    upper: fractions must strictly increase as percentile keys increase.
    lower: fractions must strictly decrease as percentile keys increase.
    """
    sorted_items = sorted(levels.items(), key=lambda kv: int(kv[0]))
    values       = [v for _, v in sorted_items]

    if tail_label == 'upper':
        ok, direction = all(
            values[i] < values[i + 1] for i in range(len(values) - 1)
        ), 'increase'
    else:  # lower
        ok, direction = all(
            values[i] > values[i + 1] for i in range(len(values) - 1)
        ), 'decrease'

    if not ok:
        raise ValueError(
            f"Window {window_key}, tail '{tail_label}': "
            f"'levels' payout fractions must strictly {direction} as percentile "
            f"keys increase, got {dict(sorted_items)}."
        )


def _validate_tail(tail_spec, window_key, tail_label,
                   parent_tlr, parent_ded):
    """Validate and normalise a tail spec, filling target_loss_ratio/markup
    defaults from parent values. levels={percentile: payout_fraction},
    checked for monotonic direction by tail. continuous requires exactly
    2 entries, unpacked into start_pct/exit_pct/min_payout/max_payout;
    layered allows any non-empty dict."""
    if tail_spec is None:
        raise ValueError(
            f"Window {window_key}, tail '{tail_label}' has no specification."
        )

    design = tail_spec.get('design')
    if design is None:
        raise ValueError(
            f"Window {window_key}, tail '{tail_label}' must specify 'design'."
        )
    if design not in ('continuous', 'layered'):
        raise ValueError(
            f"Window {window_key}, tail '{tail_label}': "
            f"design must be 'continuous' or 'layered', got '{design}'."
        )

    levels = tail_spec.get('levels')

    if design == 'continuous':
        if levels is None:
            raise ValueError(
                f"Window {window_key}, tail '{tail_label}' (continuous) requires 'levels'."
            )
        if not isinstance(levels, dict):
            raise ValueError(
                f"Window {window_key}, tail '{tail_label}' "
                "(continuous) 'levels' must be a dict of {percentile: payout_fraction}."
            )
        if len(levels) != 2:
            raise ValueError(
                f"Window {window_key}, tail '{tail_label}' "
                f"(continuous) 'levels' must have exactly 2 entries, got {len(levels)}."
            )

        _validate_monotonic_levels(levels, tail_label, window_key)

        (k_lo, v_lo), (k_hi, v_hi) = sorted(levels.items(), key=lambda kv: int(kv[0]))
        if tail_label == 'lower':
            tail_spec['start_pct']  = round(k_hi, 4)
            tail_spec['exit_pct']   = round(k_lo, 4)
            tail_spec['min_payout'] = v_hi
            tail_spec['max_payout'] = v_lo
        else:  # upper
            tail_spec['start_pct']  = round(k_lo, 4)
            tail_spec['exit_pct']   = round(k_hi, 4)
            tail_spec['min_payout'] = v_lo
            tail_spec['max_payout'] = v_hi

    elif design == 'layered':
        if not isinstance(levels, dict) or not levels:
            raise ValueError(
                f"Window {window_key}, tail '{tail_label}' (layered) 'levels' must be a non-empty dict."
            )

        _validate_monotonic_levels(levels, tail_label, window_key)

    tail_spec.setdefault('target_loss_ratio', parent_tlr)
    tail_spec.setdefault('markup',        parent_ded)
 
 
def _validate_contract(params_contract, valid_window_keys):
    """Validate params_contract and fill defaults. Cascade priority
    (high->low): tail > window > top-level > default. Raises ValueError
    on unknown window keys or missing required fields."""
    import copy
    specs = copy.deepcopy(params_contract)
 
    specs.setdefault('target_loss_ratio', 0.74)
    specs.setdefault('markup',        0.14)
 
    windows = specs.get('windows', {})
    if not windows:
        raise ValueError("params_contract must contain a non-empty 'windows' dict.")
    if not isinstance(windows, dict):
        raise ValueError("params_contract['windows'] must be a dict {window_key: {tail_specs}}.")
 
    # Cross-validate window keys
    unknown = set(windows.keys()) - set(valid_window_keys)
    if unknown:
        raise ValueError(
            f"params_contract references unknown window key(s): {unknown}. "
            f"Valid keys: {set(valid_window_keys)}."
        )
 
    TAIL_KEYS = {'upper', 'lower'}
 
    for window_key, window_spec in windows.items():
        if window_spec is None:
            raise ValueError(f"Window {window_key} has no specification.")
 
        present_tails = TAIL_KEYS.intersection(window_spec.keys())
        if not present_tails:
            raise ValueError(
                f"Window {window_key} must define at least one of: 'upper', 'lower'."
            )
 
        for tail_label in present_tails:
            _validate_tail(
                tail_spec  = window_spec[tail_label],
                window_key = window_key,
                tail_label = tail_label,
                parent_tlr = specs['target_loss_ratio'],
                parent_ded = specs['markup'],
            )
 
    return specs
 
 
 
# ---- policy table (trigger thresholds) ----
 
def _build_policy_table(fit, group_cols, params_contract):
    """Build df_policy: one row per (group, window, tail, bound/level),
    with trigger values from fit._ppf_row."""
    windows = params_contract['windows']
    records = []
 
    for _, row in fit.df_fit.iterrows():
        group_rec = row[group_cols].to_dict()
        window    = group_rec.get('window')
 
        if window not in windows:
            continue
        window_spec = windows[window]
 
        for tail_label in ('upper', 'lower'):
            if tail_label not in window_spec:
                continue
            tail_spec = window_spec[tail_label]
            design    = tail_spec['design']
            base      = {**group_rec, 'tail': tail_label}
 
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
                    })
 
            elif design == 'layered':
                for pct_int, fraction in tail_spec['levels'].items():
                    records.append({
                        **base,
                        'pct':             np.round(pct_int, 4),
                        'bound':           'level',
                        'trigger_value':   np.round(
                            fit._ppf_row(row, int(pct_int) / 100), 4
                        ),
                        'payout_fraction': fraction,
                    })
 
    return pd.DataFrame(records)
 
 
# ---- payout schedule ----
 
def compute_payout_schedule(df_cum, group_cols, params_contract):
    """Compute per-year payout/premium rate fractions for each (window,
    tail), using only complete windows. Pricing in absolute terms happens
    later, in reprice_dataframe."""
    windows = params_contract['windows']
 
    if 'flag_complete_window' in df_cum.columns:
        df = df_cum.query('flag_complete_window==1').copy()
    else:
        df = df_cum.copy()
 
    df['_pct100'] = df['percentile'] * 100
    base_cols     = group_cols + [
        'peril', 'window_year', 'start_date', 'end_date', 'period',
        'index_value', 'index_desc', 'percentile', '_pct100'
    ]
    df_base       = df[base_cols].copy()
 
    chunks = []
 
    for window_key, window_spec in windows.items():
        df_win = df_base[df_base['window'] == window_key].copy()
        if df_win.empty:
            continue
 
        for tail_label in ('upper', 'lower'):
            if tail_label not in window_spec:
                continue

            tail_spec    = window_spec[tail_label]
            design       = tail_spec['design']
            tlr          = tail_spec['target_loss_ratio']
            ded          = tail_spec['markup']

            df_tail           = df_win.copy()
            df_tail['tail']   = tail_label
            df_tail['design'] = design
            df_tail['levels'] = str(tail_spec['levels'])
            # Used downstream in reprice_dataframe to cap payout_limit
            df_tail['max_payout_fraction'] = max(tail_spec['levels'].values())

            if design == 'continuous':
                start_pct = tail_spec['start_pct']
                exit_pct  = tail_spec['exit_pct']
                max_p     = tail_spec['max_payout']
                min_p     = tail_spec['min_payout']
 
                df_tail['payout_pct'] = np.round(
                    _continuous_payout(
                        df_tail['_pct100'], start_pct, exit_pct,
                        max_p, min_p, tail_label
                    ), 10
                )
                # Trapezoid area under the ramp: rectangle (min_p) over the
                # activation probability, plus triangle (max_p - min_p) over
                # the average of activation and exit probabilities.
                if tail_label == 'lower':
                    act_prob  = start_pct / 100
                    exit_prob = exit_pct  / 100
                else:
                    act_prob  = (100 - start_pct) / 100
                    exit_prob = (100 - exit_pct)  / 100
                pure_premium_pct = (
                    min_p * act_prob
                    + (max_p - min_p) * (act_prob + exit_prob) / 2
                )
 
            elif design == 'layered':
                levels = tail_spec['levels']
                df_tail['payout_pct'] = np.round(
                    _layered_payout(df_tail['_pct100'], levels, tail_label), 10
                )
                level_list = [(float(k), v) for k, v in levels.items()]
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
 
            df_tail['pure_premium_pct']  = np.round(pure_premium_pct, 10)
            df_tail['target_loss_ratio'] = tlr
            df_tail['re_premium_pct']    = np.round(pure_premium_pct / tlr, 10)
            df_tail['markup']        = ded
            df_tail['gross_premium_pct'] = np.round(
                df_tail['re_premium_pct'] * (1 + ded), 10
            )
 
            chunks.append(df_tail)
 
    if not chunks:
        raise ValueError(
            "No rows matched any (window, tail) combination in params_contract. "
            "Check that window keys in params_contract['windows'] exist in "
            "params_indices['windows']."
        )
 
    df_payouts = pd.concat(chunks, axis=0, ignore_index=True)
    df_payouts = df_payouts.drop(columns=['_pct100'])
 
    non_group = [
        'window_year', 'tail', 'design', 'levels', 'start_date', 'end_date', 'period',
        'index_value', 'index_desc', 'percentile', 'payout_pct',
        'pure_premium_pct', 'target_loss_ratio', 're_premium_pct',
        'markup', 'gross_premium_pct', 'max_payout_fraction',
    ]
    col_order = ['peril'] + group_cols + non_group
    return df_payouts[[c for c in col_order if c in df_payouts.columns]]
 
 
########____STAGE 4: ORCHESTRATORS____########
 
MAP_PERIL_VAR = {
    ('Moisture deficit',      'swc'):      'neg_anom_swc',
    ('Moisture excess',       'swc'):      'pos_anom_swc',
    ('Precipitation deficit', 'prcp'):     'prcp',
    ('Precipitation excess',  'prcp'):     'prcp',
    ('Cold spell',            'tmin'):     'intensity_cold_spell',
    ('Heat wave',             'tmax'):     'intensity_heat_wave',
    ('Strong wind',           'wind'):     'intensity_strong_wind',
    ('Strong wind',           'windgust'): 'intensity_strong_wind',
}

def create_index_values(
    df_orig: pd.DataFrame,
    params_indices: dict,
):
    """Accumulate variable into windows, fit distribution, assign percentiles.
    Returns (df_cum, fit); fit also carries peril/index_desc for
    generate_payout_policy to reuse."""
    variable   = params_indices['variable']
    peril      = params_indices['peril']
    windows    = params_indices['windows']
    group_cols = list(params_indices.get('group_cols', GROUP_COLS))
    dist       = params_indices.get('dist', 'empirical')
    start_time = params_indices.get('start_time', None)
    duration   = params_indices.get('duration', 2)

    accum_variable = MAP_PERIL_VAR.get((peril, variable))
    if accum_variable is None:
        raise ValueError(
            f"No accumulation variable mapped for (peril, variable) = "
            f"({peril!r}, {variable!r}). Valid combinations: "
            f"{list(MAP_PERIL_VAR.keys())}."
        )


    df_hist         = df_orig.copy()
    df_hist['time'] = pd.to_datetime(df_hist['time'])
 
    if start_time:
        df_hist = df_hist[df_hist['time'] >= pd.to_datetime(start_time)].copy()
 
    df_cum = generate_accumulated_anomalies(
        df_hist,
        window=windows,
        group_cols=group_cols,
        variable_name=accum_variable,
        set_variable='index_value',
        duration=duration
    )
 
    df_cum['peril'] = peril

    groups = group_cols + ['window']
    fit    = FittedDistribution.fit(
        df_cum, groups, 'index_value', dist,
        peril=peril, index_desc=df_cum['index_desc'].iloc[0]
    )
    df_cum['percentile'] = fit.compute_percentiles(df_cum, variable='index_value')

    return df_cum, fit
 

def reprice_dataframe(df_payouts_orig, gdf_locations_orig):
    """Prices payouts/premium/insured_value in absolute terms, joined in
    from gdf_locations. The only place insured_value is computed."""
    df_payouts = df_payouts_orig.copy()
    gdf_locations = gdf_locations_orig.copy()
    if 'geometry' in gdf_locations.columns:
        gdf_locations = gdf_locations.drop(columns=['geometry'])
    df_aux = df_payouts.merge(
        gdf_locations,
        how = 'inner',
        on = ['location_id', 'window', 'peril']
    )

    df_aux['insured_value'] = df_aux['total_value'] * df_aux['coverage_pct']
    df_aux['payout_limit']  = df_aux['insured_value'] * df_aux['max_payout_fraction']
    df_aux                  = df_aux.drop(columns=['max_payout_fraction'])

    df_aux['payout']  = df_aux['insured_value'] * df_aux['payout_pct']
    df_aux['premium'] = df_aux['insured_value'] * df_aux['gross_premium_pct']

    df_aux['average_payout']    = df_aux['payout'].sum() / df_aux['insured_value'].sum()
    df_aux['empirical_premium'] = (
        df_aux['average_payout'] / df_aux['target_loss_ratio']
        * (1 + df_aux['markup']) * df_aux['insured_value']
    )
    return df_aux
 

def generate_payout_policy(
    df_cum: pd.DataFrame,
    fit,
    params_indices: dict,
    params_contract: dict,
    gdf_locations: gpd.GeoDataFrame,
):
    """Compute payout and pricing schedule from pre-fitted percentiles.
    Returns (df_payouts, df_policy). peril/index_desc/period come from
    df_cum and fit rather than being recomputed here. lead_id comes from
    params_contract and is prepended to both if provided."""
    group_cols = list(params_indices.get('group_cols', GROUP_COLS))
    lead_id    = params_contract.get('lead_id', None)
    windows    = params_indices['windows']

    valid_window_keys = set(windows.keys())
    params_contract    = _validate_contract(params_contract, valid_window_keys)

    groups = group_cols + ['window']

    df_policy  = _build_policy_table(fit, groups, params_contract)
    df_payouts = compute_payout_schedule(df_cum, groups, params_contract)
    df_payouts = reprice_dataframe(df_payouts, gdf_locations)

    df_policy['index_desc'] = fit.index_desc
    df_policy['peril']      = fit.peril

    period_by_window   = df_cum.drop_duplicates('window').set_index('window')['period']
    df_policy['period'] = df_policy['window'].map(period_by_window)

    df_policy = df_policy.merge(
        gdf_locations[['location_id', 'value_unit']].drop_duplicates(subset='location_id'),
        on='location_id', how='left'
    )

    if lead_id is not None:
        df_policy.insert(0, 'lead_id', lead_id)
        df_payouts.insert(0, 'lead_id', lead_id)
 
    return df_payouts, df_policy