from .utils import *

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
from .a01_aoi_period import *
from .a01_data_downloaders import *
from .a01_default_registry import *


#——————————————————————————————————————————————
# AOI DEFINITION
#——————————————————————————————————————————————

def create_geodataframe(
    data: Union[gpd.GeoDataFrame,pd.DataFrame], 
    params: dict
) -> gpd.GeoDataFrame:
    """
    Create a standardized GeoDataFrame from a GeoDataFrame or DataFrame.
    Args:
        data : GeoDataFrame or DataFrame
        params : dict :
            For GeoDataFrame: {'location_var': str} - column to rename to 'location_id'
            For DataFrame: {'lon_name': str, 'lat_name': str} - coordinate columns

    Returns:
        GeoDataFrame with standardized 'location_id' column and point geometries
    """
    n_initial = len(data)

    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
        gdf = gdf.drop_duplicates(subset=["geometry"])
        n_dropped = n_initial - len(gdf)
        print(f"Initial records: {n_initial} | Dropped (geometry duplicates): {n_dropped} | Remaining: {len(gdf)}")

        pad_width = len(str(len(gdf)))
        gdf["location_id"] = [f"ID-{str(i + 1).zfill(max(pad_width, 3))}" for i in range(len(gdf))]

    elif isinstance(data, pd.DataFrame):
        lon_name = params.get("lon_name")
        lat_name = params.get("lat_name")
        if lon_name is None or lat_name is None:
            raise ValueError("params must include 'lon_name' and 'lat_name' for DataFrame input.")
        if lon_name not in data.columns or lat_name not in data.columns:
            raise ValueError(f"Columns '{lon_name}' and/or '{lat_name}' not found in DataFrame.")

        df = data.drop_duplicates(subset=[lon_name, lat_name]).copy()
        n_dropped = n_initial - len(df)
        print(f"Initial records: {n_initial} | Dropped (coordinate duplicates): {n_dropped} | Remaining: {len(df)}")

        geometry = gpd.points_from_xy(df[lon_name], df[lat_name])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # Sort left-to-right, top-to-bottom (ascending lon, descending lat)
        gdf = gdf.dropna(subset=[lon_name, lat_name]).copy()
        gdf = gdf.sort_values(by=[lon_name, lat_name], ascending=[True, False]).reset_index(drop=True)
        pad_width = len(str(len(gdf)))
        gdf["location_id"] = [f"ID-{str(i + 1).zfill(max(pad_width, 3))}" for i in range(len(gdf))]

    else:
        raise TypeError("Input must be a GeoDataFrame or DataFrame.")

    return gdf


def count_vertices(geom) -> int:
    if geom.geom_type == 'Polygon':
        return sum(len(r.coords) for r in [geom.exterior] + list(geom.interiors))
    elif geom.geom_type.startswith('Multi') or geom.geom_type == 'GeometryCollection':
        return sum(count_vertices(g) for g in geom.geoms)
    return len(geom.coords)


def simplify_to_vertex_limit(geom, max_vertices: int, tol_start=0.001, tol_factor=2.0, max_iter=20):
    """
    Iteratively simplify a geometry until it has <= max_vertices.
    Falls back to convex_hull if simplification is too aggressive.
    """
    if count_vertices(geom) <= max_vertices:
        return geom

    tol = tol_start
    for _ in range(max_iter):
        simplified = geom.simplify(tol, preserve_topology=True)
        if count_vertices(simplified) <= max_vertices:
            return simplified
        tol *= tol_factor

    # Last resort
    return geom.convex_hull


def simplify_gdf(gdf: gpd.GeoDataFrame, max_vertices: int, **kwargs) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf['geometry'] = gdf['geometry'].apply(
        lambda g: simplify_to_vertex_limit(g, max_vertices, **kwargs)
    )
    return gdf

'''
def get_aoi(
    gdf: gpd.GeoDataFrame, 
    params_s: dict = {}
):
    """
    Creates spatial context parameters for the target population
    Args:
        - gdf : (gpd.GeoDataFrame) containing the AOI that sets boundaries to the area
        - params_s : (dict) Spatial parameters to slice the GeoDataFrame 
    Returns:
        - dict of spatial bounding box
    """
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    override  = params_s.get('override_gdf', False)
    simplify = params_s.get('simplify', False)

    if {'minx', 'miny', 'maxx', 'maxy'}.issubset(params_s.keys()) and override:
        dict_s = get_bounds(**params_s)
        # Create GeoDataFrame for post processing
        polygons = [
            box(dict_s['minx'], dict_s['miny'], dict_s['maxx'], dict_s['maxy'])
        ]

        gdf_aoi = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
        gdf_aoi['location_id'] = 'ID-000'
    else:
        # Create GeoDataFrame for post processing
        gdf = create_geodataframe(gdf, params_s)

        gdf_aoi = subset_geometry(gdf, params_s)
        if len(gdf_aoi) == 0:
            raise AssertionError('Warning: Geometry slice has no elements')
        # Create bounds from AOI in the form of longitude and latitude borders
        minx, miny, maxx, maxy = gdf_aoi.total_bounds
        dict_s = get_bounds(minx, maxx, maxy, miny, 10)
    
    if simplify:
        gdf_aoi = simplify_gdf(gdf_aoi, max_vertices=1400)
        
    gdf_aoi['area_km2'] = get_area_column(gdf_aoi, 'km2')
    
    return BoundingBox(**dict_s).to_dict(), gdf_aoi
'''
def register_geometry(
    gdf: gpd.GeoDataFrame,
    params_s: dict = {}
) -> gpd.GeoDataFrame:
    """
    Reprojects and registers the full administrative geometry, assigning
    location_id consistently across the whole dataset. Runs once per
    country/region in the from_request pipeline only. Its output is
    reused by every lead that subsets it in create_aoi_geometry, so
    location_id stays stable regardless of which subset a lead requests.

    Args:
        - gdf: (gpd.GeoDataFrame) raw administrative boundaries (gdf_request)
        - params_s: (dict) spatial parameters passed to create_geodataframe
    Returns:
        - gpd.GeoDataFrame: registered geometry with stable location_id
    """
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return create_geodataframe(gdf, params_s)


def create_aoi_geometry(
    params_s: dict = {},
    gdf: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Produces the final, per-lead AOI geometry with area_km2. Subsets the
    registered geometry (if provided) or builds a bbox polygon directly
    from params_s. Runs per lead, on top of register_geometry's output.

    Args:
        - params_s: (dict) spatial parameters. Must contain
          minx/miny/maxx/maxy when gdf is None or override_gdf is True.
        - gdf: (gpd.GeoDataFrame | None) output of register_geometry, or None
    Returns:
        - gpd.GeoDataFrame: final sliced AOI geometry with area_km2
    """
    override = params_s.get('override_gdf', False)

    if gdf is None or override:
        dict_s = get_bounds(**params_s)
        polygon = box(dict_s['minx'], dict_s['miny'], dict_s['maxx'], dict_s['maxy'])
        gdf_aoi = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        gdf_aoi['location_id'] = 'ID-000'
    else:
        gdf_aoi = subset_geometry(gdf, params_s)
        if len(gdf_aoi) == 0:
            raise AssertionError('Warning: Geometry slice has no elements')

    if params_s.get('simplify', False):
        gdf_aoi = simplify_gdf(gdf_aoi, max_vertices=1400)

    gdf_aoi['area_km2'] = get_area_column(gdf_aoi, 'km2')

    return gdf_aoi


def get_aoi_bounds(
    gdf_aoi: gpd.GeoDataFrame
) -> dict:
    """
    Returns the bounding box as BoundingBox(...).to_dict(), read directly
    off the final AOI geometry produced by create_aoi_geometry.

    Args:
        - gdf_aoi: (gpd.GeoDataFrame) output of create_aoi_geometry
    Returns:
        - dict: bounding box
    """
    minx, miny, maxx, maxy = gdf_aoi.total_bounds
    dict_s = get_bounds(minx, maxx, maxy, miny, 10)

    return BoundingBox(**dict_s).to_dict()

#——————————————————————————————————————————————
# DATA REQUEST CLASSES
#——————————————————————————————————————————————

@dataclass
class DataRequest:
    """
    Lightweight container for a single download job.
 
    Args:
        lead_id       : Identifier for the insurance lead.
        provider      : Data source -- "ERA5" or "UCSB".
        field         : Short name of requested variable.
        aoi           : BoundingBox or dict with keys minx/maxx/miny/maxy.
        params_request: Dict of kwargs forwarded to ERA5Config / UCSBConfig.
        period        : (start_year, end_year) tuple. If None, compute_period() is used.
    """
    lead_id: str
    provider: str
    field: str
    aoi: Union[BoundingBox, dict]
    params_request: dict = field(default_factory=dict)
    period: Optional[tuple] = None

    def parse_request(self):
        self.aoi = BoundingBox.from_dict(self.aoi) if isinstance(self.aoi, dict) else self.aoi
        if self.period is None:
            self.period = compute_period()

@dataclass
class DataRequestResult(DataRequest):
    """Holds the output of a completed DataRequest."""
    data: Dict[str, object] = field(default_factory=dict)   # {year_str: xr.Dataset}
    errors: List[str] = field(default_factory=list)
    
    @classmethod
    def from_request(cls, request: DataRequest, data: dict, errors: list) -> "DataRequestResult":
        return cls(
            lead_id=request.lead_id,
            provider=request.provider,
            field=request.field,
            aoi=request.aoi,
            params_request=request.params_request,
            period=request.period,
            data=data,
            errors=errors,
        )

    @property
    def success(self) -> bool:
        return bool(self.data) and not self.errors

    @property
    def partial(self) -> bool:
        return bool(self.data) and bool(self.errors)


class DataRequestService:

    def run(self, request: DataRequest) -> DataRequestResult:
        request.parse_request()
        start_year, end_year = request.period
        
        print(
            f"lead: {request.lead_id} | provider: {request.provider} | "
            f"period: {describe_period(start_year, end_year)}"
        )

        # Get configuration and downloaders from provider
        provider_cfg  = get_provider_config(request.provider)
        config_cls = provider_cfg.config_cls
        downloader_cls = provider_cfg.downloader_cls

        # Instantiate Configuration and Downloader
        config = config_cls.from_dict(request.params_request)
        downloader = downloader_cls(aoi=request.aoi, params_request=config)

        dict_ds = {}
        errors: List[str] = []

        try:
            dict_ds = downloader.download_period(start_year=start_year, end_year=end_year)
        except NotImplementedError as exc:
                errors.append(str(exc))
                print(f"not implemented: {exc}")
        except Exception as exc:
                msg = f"{request.provider} download error: {exc}"
                errors.append(msg)
                print(msg)

        return DataRequestResult.from_request(
            request=request,
            data=dict_ds,
            errors=errors
        )


#——————————————————————————————————————————————
# EXTRACT DATA FUNCTION
#——————————————————————————————————————————————


def extract_data(
    params_s: dict,
    params_request: dict,
    params_t: Optional[dict] = None,
) -> DataRequestResult:
    """
    Download climate data for a given AOI, source configuration, and time span.
 
    Args:
        params_s       : Spatial bounds. Required keys: minx, maxx, miny, maxy.
                         Any extra keys (e.g. admin filters) are ignored.
        params_request : Source configuration forwarded to ERA5Config / UCSBConfig.
                         Must include key "origin" ("ERA5" or "UCSB") and any
                         source-specific parameters (variable, product_type, ...).
                         May also include "lead_id", "start_year", "end_year".
        params_t       : Optional time span dict with keys start_year and end_year.
                         If None, those keys are read from params_request; if still
                         absent, compute_period() supplies a 30-year default window.
 
    Returns:
        DataRequestResult with .data dict keyed by year string and .errors list.
 
    Examples:
        # ERA5 download
        result = extract_data(
            params_s={"minx": -75.0, "maxx": -69.9, "miny": 1.6, "maxy": 6.3},
            params_request={
                "origin": "ERA5",
                "lead_id": "lead_001",
                "variable": ["total_precipitation"],
                "product_type": "reanalysis-era5-land",
            },
            params_t={"start_year": 1993, "end_year": 2023},
        )
 
        # UCSB / CHIRPS download (params_t inside params_request)
        result = extract_data(
            params_s={"minx": -75.0, "maxx": -69.9, "miny": 1.6, "maxy": 6.3},
            params_request={
                "origin": "UCSB",
                "lead_id": "lead_002",
                "variable": "PRCP",
                "origin_ucsb": "CHIRPS",
                "start_year": 1993,
                "end_year": 2023,
            },
        )
    """
    # Get period from params_t
    start_year = params_t.get('start_year', None)
    end_year = params_t.get('end_year', None)
    period = start_year, end_year
    if start_year is None or end_year is None:
        period = None

    # Get lead id and provider from params_request
    lead_id = str(params_request.get("lead_id", "unknown"))
    provider = params_request.get("provider")
    field = params_request.get("field")
    country = params_request.get("country", "")
    data_path = params_request.get("data_path", "")

    params_default = get_peril_config(provider, field)
    params_request = params_default | params_request

    request = DataRequest(
        lead_id=lead_id,
        provider=provider,
        field=field,
        aoi=params_s,
        params_request=params_request,
        period=period,
    )
    request.parse_request()
    start_year, end_year = request.period
    all_years = get_year_list(start_year, end_year)

    # ── Cache check ────────────────────────────────────────────────────────────
    # The catalog saves each year as:
    #   {data_path}/{country}/{lead_id}/sources/{provider}_{field}_{year}.zarr
    # If the zarr exists AND has actual data chunks, skip the CDS request.
    # A valid zarr must have at least one chunk file (named without a leading '.')
    # inside a DATA-variable sub-directory.  Coordinate-only zarrs (lat/lon/time
    # written but the main variable never flushed) look non-empty but are all-NaN.

    def _zarr_has_data_chunks(zarr_path: Path) -> bool:
        """Return True only if at least one non-metadata file exists
        inside any sub-directory other than pure coordinate arrays."""
        coord_only = {'lat', 'lon', 'latitude', 'longitude', 'time', 'number', 'expver'}
        for sub in zarr_path.iterdir():
            if not sub.is_dir() or sub.name in coord_only:
                continue
            for f in sub.iterdir():
                if f.is_file() and not f.name.startswith('.'):
                    return True
        return False

    cached: Dict[str, xr.Dataset] = {}
    missing_years: List[int] = []

    if data_path:
        cache_dir = Path(data_path) / country / lead_id / "sources"
        for year in all_years:
            zarr_path = cache_dir / f"{provider}_{field}_{year}.zarr"
            if zarr_path.exists() and _zarr_has_data_chunks(zarr_path):
                print(f"cache hit  → {year} (skipping download)")
                cached[str(year)] = xr.open_zarr(str(zarr_path), consolidated=False, zarr_format=2)
            else:
                if zarr_path.exists():
                    print(f"cache miss → {year} (zarr exists but has no data chunks — re-downloading)")
                missing_years.append(year)
    else:
        missing_years = all_years

    if missing_years:
        print(f"downloading {len(missing_years)} year(s): {missing_years[0]}–{missing_years[-1]}")
        request.period = (missing_years[0], missing_years[-1])
        result = DataRequestService().run(request)
        # Only keep years that were actually requested (avoids re-downloading extras)
        downloaded = {k: v for k, v in result.data.items() if int(k) in missing_years}
    else:
        downloaded = {}

    print(f"summary → {len(cached)} cached, {len(downloaded)} downloaded, "
          f"{len(all_years) - len(cached) - len(downloaded)} failed")

    return {**cached, **downloaded}


#—————————————————————————————————————————————
# REQUEST DATA FROM PLANET
#—————————————————————————————————————————————


def create_planet_subscriptions(
    params_subscription: dict,
    params_t: dict,
    params_s: dict,
    gdf: gpd.GeoDataFrame,
    credentials: dict,
    coordinates=None,
) -> pd.DataFrame:
    """
    Create Planet subscriptions for each area in gdf.

    Args:
        params_subscription : subscription config (id_column, variable, version, band, prefix, suffix).
                              Can include 'start_date' and 'end_date' to override params_t derivation.
        params_t            : temporal config with start_year and end_year. Ignored if
                              start_date and end_date are set directly in params_subscription.
        params_s            : spatial bounding parameters passed to subset_geometry.
        gdf                 : GeoDataFrame of AOIs.
        credentials         : Planet + AWS credentials.
        coordinates         : optional fixed coordinate list (overrides gdf geometry).

    Returns:
        pd.DataFrame with subscription details and keys.
    """
    PLANET_FIRST_DATE = pd.Timestamp('2002-06-15')
    GAP_START         = pd.Timestamp('2011-10-04')
    GAP_END           = pd.Timestamp('2012-07-24')
    today             = pd.Timestamp.today().normalize()

    # --- Resolve start_date and end_date ---
    # params_subscription can override directly, skipping params_t derivation
    if 'start_date' in params_subscription and 'end_date' in params_subscription:
        start_date = pd.Timestamp(params_subscription['start_date'])
        end_date   = pd.Timestamp(params_subscription['end_date'])
    else:
        start_year = int(params_t['start_year'])
        end_year   = int(params_t['end_year'])

        raw_start = pd.Timestamp(f'{start_year}-01-01')
        if GAP_START <= raw_start <= GAP_END:
            start_date = GAP_END + pd.Timedelta(days=1)
        else:
            start_date = max(raw_start, PLANET_FIRST_DATE)

        raw_end = pd.Timestamp(f'{end_year}-12-31')
        if GAP_START <= raw_end <= GAP_END:
            end_date = GAP_START - pd.Timedelta(days=1)
        elif end_year >= today.year:
            end_date = today - pd.Timedelta(days=5)
        else:
            end_date = raw_end

    # Guard against empty range
    if start_date > end_date:
        raise ValueError(
            f"Resolved date range [{start_date.date()} - {end_date.date()}] is empty. "
            "The requested period may fall entirely within the Planet data gap (2011-10-04 to 2012-07-24)."
        )

    # Split around the gap if the period straddles it
    if start_date < GAP_START and end_date > GAP_END:
        date_ranges = [
            (start_date,                     GAP_START - pd.Timedelta(days=1)),  # up to 2011-10-03
            (GAP_END + pd.Timedelta(days=1), end_date),                          # from 2012-07-25
        ]
    else:
        date_ranges = [(start_date, end_date)]

    id_column  = params_subscription['id_column']
    gdf        = subset_geometry(gdf, params_s)
    list_areas = np.sort(gdf[id_column].unique())

    dict_list = []
    for area_id in list_areas:
        gdf_request = gdf[gdf[id_column] == area_id].copy()
        for sd, ed in date_ranges:
            params_sub_iter = params_subscription.copy()
            params_sub_iter['start_date'] = sd
            params_sub_iter['end_date']   = ed
            dict_params       = get_params_subscription(params_sub_iter, gdf_request, coordinates)
            dict_subscription = create_subscription(dict_params, credentials)
            if dict_subscription is not None:
                dict_list.append(dict_subscription)

    df_sub = pd.DataFrame(dict_list)
    df_sub['status'] = df_sub['key'].apply(lambda k: get_status(k, credentials))

    pending_status = ['preparing', 'pending']
    running_status = ['running', 'failed', 'invalid', 'suspended', 'cancelled', 'unsent', 'completed']

    while (
        df_sub['status'].isin(pending_status).any() and
        (~df_sub['status'].isin(running_status)).any()
    ):
        time.sleep(15)
        df_sub['status'] = df_sub['key'].apply(lambda k: get_status(k, credentials))

    df_sub = df_sub[df_sub['status'].isin(running_status)].drop(columns=['status'])
    return df_sub