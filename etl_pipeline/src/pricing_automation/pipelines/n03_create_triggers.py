from .utils import *

from scipy.stats import rankdata

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

########____GENERATE ACCUMULATED ANOMALIES____########


def generate_accumulated_anomalies(
    df_orig, 
    window, 
    time_dim = 'time', 
    group_cols = ['climate_area_id'],
    variable_name = 'swc_adjusted',
    set_variable = 'index_value'
):
    """Generate accumulated anomaly for the time series given the time window"""
    DESC_VARIABLE = f'cum_{variable_name}'
    df = df_orig.copy()
    df[time_dim] = pd.to_datetime(df[time_dim])
    month_day = df[time_dim].dt.strftime('%m-%d')
    year = df[time_dim].dt.year

    df_total = []

    for code, spec in window.items():
        if isinstance(spec, list):
            lapse = spec
            crop = 'all'
        else:
            lapse = spec['dates']
            crop = spec.get('crop', 'all')

        start_month, start_day = map(int, lapse[0].split('-'))
        end_month, end_day = map(int, lapse[1].split('-'))
        crosses_year = (end_month < start_month) or (end_month == start_month and end_day < start_day)

        # Determine window year
        if crosses_year:
            in_window = (month_day >= lapse[0]) | (month_day <= lapse[1])
            # Adjust window_year: if we're in the "end" part (Jan-Feb), use previous year
            window_year = np.where(month_day <= lapse[1], year - 1, year)
        else:
            in_window = (month_day >= lapse[0]) & (month_day <= lapse[1])
            window_year = year
        
        # Filter to rows in this window
        if not in_window.any():
            continue

        df_window = df[in_window].copy()
        df_window['window'] = code
        df_window['crop'] = crop
        df_window['window_year'] = window_year[in_window]
        df_window['start_date'] = pd.to_datetime(df_window['window_year'].astype(str) + '-' + lapse[0])
        df_window['end_date'] = pd.to_datetime(
            (df_window['window_year'] + crosses_year).astype(str) + '-' + lapse[1]
        )
        df_total.append(df_window)

    df_total = pd.concat(df_total, axis=0, ignore_index=True)

    df_cumulated = df_total.groupby(
        group_cols + ['crop', 'window_year', 'window', 'start_date', 'end_date'], as_index = False
    ).agg(**{
        'first_date': (time_dim, 'min'),
        'last_date': (time_dim, 'max'),
        #f'cum_{variable_name}': (variable_name, lambda x: np.round(np.sum(x), 4))
        f'{set_variable}': (variable_name, lambda x: np.round(np.sum(x), 4))
    })
    df_cumulated['index_desc'] = DESC_VARIABLE
    df_cumulated['first_date'] = pd.to_datetime(df_cumulated['first_date'].dt.date)
    df_cumulated['last_date'] = pd.to_datetime(df_cumulated['last_date'].dt.date)
    
    # Remove windows that are incomplete from the beginning
    df_cumulated = df_cumulated[~(df_cumulated['first_date']>df_cumulated['start_date'])].copy()
    df_cumulated = df_cumulated.drop(columns=['first_date']).reset_index(drop=True)

    df_cumulated['flag_complete_window'] = np.where(
        df_cumulated['last_date'] < df_cumulated['end_date'], 0, 1
    )

    return df_cumulated


########____CREATE CLUSTER COORDINATE____########


def apply_pca(df, n_comp):
    """Performs a Principal Component Analysis on the given Dataframe"""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(df_scaled)
    df_pca = pd.DataFrame(
        pca.transform(df_scaled), 
        index=df.index, 
        columns=[f'comp{i}' for i in range(n_comp)]
    )
    return df_pca


def get_cluster(df, n_cluster):
    """Performs a K means algorithm to assign a cluster in the DataFrame"""
    # Apply K-Means
    km = KMeans(n_clusters=n_cluster, n_init=12, random_state=42)
    series = km.fit_predict(df)
    return series


def create_cluster_coordinate(df_orig, params_polygons):
    """Create climate area geometries from a Time Series for Soil Water Content"""
    
    variable = params_polygons['variable']
    n_comp = params_polygons['n_comp']
    n_cluster = params_polygons['n_cluster']
    group_cols = params_polygons['group_cols']
    
    # Convert time into datetime
    df_sum = df_orig.copy()

    #df_sum['time'] = pd.to_datetime(df_sum['time'])
    #df_sum['time'] = df_sum['time'].dt.strftime('%Y%m%d')
    # Transform daily observations to columns
    df = df_sum.pivot_table(index=group_cols, columns=['window_year', 'window'], values=[variable], aggfunc='mean')
    df.columns = [col[0] + '_' + str(col[1]) + '_' + str(col[2]) for col in df.columns]

    # Apply Principal Component Analysis
    df_pca = apply_pca(df, n_comp)
    # Apply KMeans algorithm based on the Principal Component Analysis
    df_pca['cluster'] = get_cluster(df_pca, n_cluster)
    # Reset index to add coordinates to dataframe
    df_pca = df_pca.reset_index()
    
    # Add cluster column to the original dataset
    df_cluster = df_orig.merge(
        df_pca[group_cols + ['cluster']]
    )

    return df_cluster


########____FUNCTIONS TO FIT PARAMETRIC DISTRIBUTION____########


def fit_distribution(df_cum, group_cols, variable, dist='t'):
    """"
    Fit a parametric distribution to the data within the specified groups
    Args:
        - df_cum_orig : (pd.DataFrame) DataFrame with original data to fit the parametric distrbition
        - group_cols : (list) List of column names to group on in order to fit a distribution
        - variable : (str) Name of variable to fit
        - dist : (str) Name of distribution to fit could be either of {'norm', 't', 'lognorm', 'weibull', 'gamma', 'norm_inf'}
    Returns:
        - pd.DataFrame with the keys of each group and the parameters of the fitted distribution
    """
    if 'flag_complete_window' in df_cum.columns:
        df_cum = df_cum.query('flag_complete_window==1').copy()
    grouped = df_cum.groupby(group_cols)
    all_param_records = []

    for group_key, group_df in grouped:
        x = group_df[variable].values
        param_record = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        
        if dist == 'norm':
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            param_record.update({'dist': 'norm', 'mu': mu, 'sigma': sigma})

        elif dist == 't':
            mu, sigma, dfree = np.mean(x), np.std(x, ddof=1), len(x) - 1
            param_record.update({'dist': 't', 'mu': mu, 'sigma': sigma, 'df': dfree})

        elif dist == 'lognorm':
            sigma, loc, scale = stats.lognorm.fit(x + 1, floc=0)
            param_record.update({'dist': 'lognorm', 's': sigma, 'loc': loc, 'scale': scale})

        elif dist == 'weibull':
            shape, loc, scale = stats.weibull_min.fit(x, floc=0)
            param_record.update({'dist': 'weibull', 'shape': shape, 'loc': loc, 'scale': scale})

        elif dist == 'gamma':
            shape, loc, scale = stats.gamma.fit(x)
            param_record.update({'dist': 'gamma', 'shape': shape, 'loc': loc, 'scale': scale})

        elif dist == 'norm_inf':
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            param_record.update({'dist': 'norm_inf', 'mu': mu, 'sigma': sigma})

        elif dist == 'gev':
            shape, loc, scale = stats.genextreme.fit(x)
            param_record.update({'dist': 'gev', 'shape': shape, 'loc': loc, 'scale': scale})

        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        all_param_records.append(param_record)

    return pd.DataFrame(all_param_records)


def calculate_triggers_from_params(
    df_fit, 
    percentile_dict, 
    group_cols
):
    """
    Calculate triggers from percentile list using the parameters
    Args:
        - df_fit : (pd.DataFrame) DataFrame with the parameters for each group
        - percentile_dict : (dict) Dictionary whose keys are the desired percentiles
        - group_cols : (list) List of column names from grouping set to preserve
    Returns:
        - pd.DataFrame DataFrame with computed percentiles for each group
    """
    trigger_records = []

    for _, row in df_fit.iterrows():
        rec = row[group_cols].to_dict()

        for p in percentile_dict['layers'].keys():
            if row['dist'] == 'norm':
                value = np.round(row['mu'] + row['sigma'] * stats.norm.ppf(p / 100), 4)

            elif row['dist'] == 't':
                value = np.round(row['mu'] + row['sigma'] * stats.t.ppf(p / 100, row['df']), 4)

            elif row['dist'] == 'lognorm':
                value = np.round(stats.lognorm.ppf(p / 100, s=row['s'], loc=row['loc'], scale=row['scale']) - 1, 4)

            elif row['dist'] == 'weibull':
                value = np.round(stats.weibull_min.ppf(p / 100, row['shape'], loc=row['loc'], scale=row['scale']), 4)

            elif row['dist'] == 'gamma':
                value = np.round(stats.gamma.ppf(p / 100, row['shape'], loc=row['loc'], scale=row['scale']), 4)

            elif row['dist'] == 'norm_inf':
                value = np.round(row['mu'] + 1.04 * row['sigma'] * stats.norm.ppf(p / 100), 4)
            
            elif row['dist'] == 'gev':
                value = np.round(stats.genextreme.ppf(p/100, row['shape'], loc=row['loc'], scale=row['scale']),4)
            
            else:
                value = np.nan

            rec[f'P{p}_trigger'] = value

        trigger_records.append(rec)

    return pd.DataFrame(trigger_records)


def add_percentiles_from_params(
    df, 
    df_fit, 
    group_cols, 
    variable,
    perc_name='percentile'
):
    """
    Add percentile column based on the fitted distributions for each index
    Args:
        - df : (pd.DataFrame) DataFrame with the fitted distributions
        - df_fit : (pd.DataFrame) DataFrame with the fitted parameters per group
        - group_cols : (list) List of column names of grouping columns
        - variable : (str) Name of variable to compute the percentile
    Returns:
        - pd.DataFrame with the original variables plus an additional column for percentiles from variable
    """
    df = df.copy()

    cols_to_drop = set(df_fit.columns) - set(group_cols)
    
    # Merge fitted parameters into your anomaly dataframe
    df = df.merge(df_fit, on=group_cols, how='left')

    def compute_percentile(row):
        x = row[variable]
        dist = row['dist']

        if dist == 'norm':
            return np.round(stats.norm.cdf((x - row['mu']) / row['sigma']), 4)
        elif dist == 't':
            return np.round(stats.t.cdf((x - row['mu']) / row['sigma'], df=row['df']), 4)
        elif dist == 'lognorm':
            return np.round(stats.lognorm.cdf(x + 1, s=row['s'], loc=row['loc'], scale=row['scale']), 4)
        elif dist == 'weibull':
            return np.round(stats.weibull_min.cdf(x, c=row['shape'], loc=row['loc'], scale=row['scale']), 4)
        elif dist == 'gamma':
            return np.round(stats.gamma.cdf(x, a=row['shape'], loc=row['loc'], scale=row['scale']), 4)
        elif dist == 'norm_inf':
            return np.round(stats.norm.cdf((x - row['mu']) / (1.04 * row['sigma'])), 4)
        elif dist == 'gev':
            return np.round(stats.genextreme.cdf(x, row['shape'], loc=row['loc'], scale=row['scale']), 4)
        else:
            return np.nan

    df[perc_name] = df.apply(compute_percentile, axis=1)

    df = df.drop(columns=cols_to_drop)
    
    return df


def get_empirical_percentiles(
    df_cum,
    group_cols,
    variable,
    percentiles_dict=None,
    perc_name='percentile'
):
    """
    Create columns of empirical percentiles and the corresponding percentile value of each record
    Args:
        - df_cum_orig : (pd.DataFrame) DataFrame with computed anomalies 
        - group_cols: (list) List of columns to group by
        - variable : (str) Name of variable to compare against the trigger
        - percentiles_dict : (dict) Dictionary of percentile values and sides to mark as 1 {90:'upper', 10:'lower'}
    """
    if 'flag_complete_window' in df_cum.columns:
        df_cum_slice = df_cum.query('flag_complete_window==1')
    else:
        df_cum_slice = df_cum.copy()
    grouped = df_cum_slice.groupby(group_cols, as_index=False)
    
    # Create a dataframe of empirical percentile thesholds per group
    df_percentiles = None
    if percentiles_dict['layers']:
        df_percentiles = grouped[variable].agg(**{
            f'P{p}_trigger': lambda x, p=p: np.nanpercentile(x, p)
            for p in percentiles_dict['layers'].keys()
        })

    df_cum[perc_name] = df_cum.groupby(group_cols)[variable].transform(
        lambda x: np.round((rankdata(x) - 1) / (np.size(x)-1) , 4)
    )

    return df_cum, df_percentiles



def get_activation_years(
    df_cum_anomalies, 
    df_percentiles, 
    group_cols,
    variable,
    percentiles_dict
):
    """
    Create columns with activation years given the triggers
    Args:
        - df_cum_anomalies : (pd.DataFrame) DataFrame with computed anomalies 
        - df_percentiles : (pd.DataFrame) DataFrame with computed percentiles
        - group_cols : (list) List of columns to group by
        - variable : (str) Name of variable to compare against the trigger
        - percentiles_dict : (dict) Dictionary of percentile values and sides to mark as 1 {90:'upper', 10:'lower'}
    """
    df = df_cum_anomalies.copy()
    
    df = df.merge(
        df_percentiles,
        how = 'left',
        on = group_cols
    )
    for p in percentiles_dict['layers'].keys():
        if percentiles_dict['side'] == 'upper':
            df[f'P{p}_activated'] = np.where(df[variable] >= df[f'P{p}_trigger'], 1, 0)
        elif percentiles_dict['side'] == 'lower':
            df[f'P{p}_activated'] = np.where(df[variable] < df[f'P{p}_trigger'], 1, 0)
        else:
            raise ValueError('Side must be either upper or lower')
    return df


def compute_layered_payout(percentile, percentile_dict):
    """
    Step-function payout for a layered policy.

    Args:
        percentile : array-like. Empirical percentile(s) in [0, 1].
        percentile_dict : dict
        Keys: 'side' ('lower' or 'upper'), 'layers' (dict of {percentile: payout_fraction}).
            Example:
            {'side': 'upper', 'layers': {90: 0.4, 95: 0.5, 99: 1.0}}
            {'side': 'lower', 'layers': {20: 0.3, 10: 0.6,  5: 1.0}}
    """
    tail = percentile_dict['side']
    layers = [(int(k) / 100, v) for k, v in percentile_dict['layers'].items()]

    p = np.asarray(percentile, dtype=float)
    payout = np.zeros_like(p)

    if tail == 'lower':
        for threshold, fraction in sorted(layers, key=lambda x: x[0], reverse=True):
            payout = np.where(p <= threshold, fraction, payout)
    elif tail == 'upper':
        for threshold, fraction in sorted(layers, key=lambda x: x[0]):
            payout = np.where(p >= threshold, fraction, payout)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    return payout


########____GENERATE OUTPUT____########


def generate_triggers(
    df_hist: pd.DataFrame, 
    params: dict,
    params_request: dict,
):
    """
    Generate triggers
    Args:
        - df_hist : (pd.DataFrame) DataFrame with historical dataset of indices
        - params : (dict) Dictionary with parameters to compute triggers
    Returns:
        - pd.DataFrame DataFrame with percentiles, triggers and activation flags
        - pd.DataFrame Original DataFrame
    """
    windows = params['windows']
    variable = params['variable']
    group_cols = params.get('group_cols', 'location_id')
    percentile_dict = params['percentile_dict']
    dist = params['dist']

    if params_request.get('lead_id', None) is not None:
        df_hist['lead_id'] = params_request['lead_id']        
        group_cols.append('lead_id')
  
    
    df_hist['time'] = pd.to_datetime(df_hist['time']) 

    # Accumulate the values of the variable within the specified windows
    SUM_VARIABLE = 'index_value' #f'cum_{variable}'
    df_cum_variable = generate_accumulated_anomalies(
        df_hist, 
        window = windows,
        group_cols = group_cols,
        variable_name = variable,
        set_variable = SUM_VARIABLE
    )

    groups = group_cols + ['crop', 'window']

    if dist == 'empirical':
        df_cum_variable, df_percentiles = get_empirical_percentiles(
            df_cum_variable,
            groups,
            SUM_VARIABLE,
            percentile_dict
        )
    
    else:
        # Fit distribution and get parameters
        df_fit = fit_distribution(
            df_cum_variable,
            groups, 
            SUM_VARIABLE,
            dist
        )
        # Add the associated percentile values for each window
        df_cum_variable = add_percentiles_from_params(
            df_cum_variable,
            df_fit,
            groups,
            SUM_VARIABLE
        )
        # Calculate trigger for each percentile value only with complete window
        df_percentiles = calculate_triggers_from_params(
            df_fit, 
            percentile_dict, 
            groups
        )
    
    # Get flags for activated windows
    df_cum_variable =  get_activation_years(
        df_cum_variable, 
        df_percentiles,
        groups, 
        SUM_VARIABLE, 
        percentile_dict,
    )

    df_cum_variable['empirical_loss'] = compute_layered_payout(
        df_cum_variable['percentile'],
        percentile_dict
    ) 

    return df_cum_variable, df_percentiles