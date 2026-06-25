from .utils import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from rasterio.features import shapes
from rasterio.transform import from_origin

from shapely.geometry import shape, Point, MultiPoint
from shapely.ops import voronoi_diagram


#———————————————————————————————————————————
# ASSIGN CLUSTER COORDINATE TO DATASET
#———————————————————————————————————————————


def create_cluster_coord(ds_orig, gdf_orig, cluster_var, method='within', check_var=None, k=1):
    ds = ds_orig.copy()
    gdf = gdf_orig.copy()

    lon_var, lat_var, time_var = get_coordinates(ds)

    if method == 'within':
        points = ds[[lat_var, lon_var]].to_dataframe().reset_index()
        points = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points[lon_var], points[lat_var]),
            crs=gdf.crs
        )
        points = gpd.sjoin(points, gdf[[cluster_var, "geometry"]], how="left", predicate="within")
        ds_points = points.set_index([lat_var, lon_var])[[cluster_var]]
        ds_points = ds_points.to_xarray().set_coords(cluster_var)
        ds = xr.merge([ds, ds_points])
        df_cluster = ds[[lon_var, lat_var, cluster_var]].to_dataframe()
        df_cluster = df_cluster.reset_index().dropna(subset=[cluster_var])

    elif method == 'nearest':
        gdf['lon'] = gdf.centroid.x
        gdf['lat'] = gdf.centroid.y

        # Build candidate grid points from 1D axes (one row per pixel, no time dim)
        lons = ds[lon_var].values
        lats = ds[lat_var].values
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        points = pd.DataFrame({
            lon_var: lon_mesh.ravel(),
            lat_var: lat_mesh.ravel(),
        })

        # Discard all-NaN pixels before the tree query so nearest is always valid.
        # valid_count reduces over time in xarray; ravel must match the (lat, lon)
        # meshgrid C-order, so transpose the mask to (lat, lon) first.
        if check_var:
            valid_count = (~ds[check_var].isnull()).sum(dim=time_var)
            valid_mask = valid_count.transpose(lat_var, lon_var).values.ravel() != 0
            points = points[valid_mask].reset_index(drop=True)

        df_cluster = match_grid_points(points, gdf, k=k)
        df_cluster = df_cluster[[lon_var, lat_var, cluster_var]].copy()

        ds = select_coordinates(ds, df_cluster, grid_coords=(lon_var, lat_var))

    df_cluster['method'] = method

    return ds, df_cluster


#———————————————————————————————————————————
# SUMMARIZE DATASET
#———————————————————————————————————————————


def summarize_data(ds, red_dims=[None], group_coords=[None], func='mean'):
    """Summarize data along the specified coordinates"""    
    if set(red_dims).issubset(set(list(ds.sizes.keys()))):
        # Average over all grouping columns (dimensions)
        if func == 'mean':
            ds = ds.mean(dim = red_dims)
        elif func == 'min':
            ds = ds.min(dim = red_dims)
        elif func == 'max':
            ds = ds.max(dim = red_dims)
        elif func == 'median':
            ds = ds.median(dim = red_dims)
        elif func == 'sum':
            ds = ds.sum(dim = red_dims)
        else: 
            raise ValueError(f'The summarizing function {func} is not implemented.')
    elif set(group_coords).issubset(set(list(ds.coords))):
        for coord in group_coords:
            if func == 'mean':
                ds = ds.groupby(coord).mean()
            elif func == 'min':
                ds = ds.groupby(coord).max()
            elif func == 'max':
                ds = ds.groupby(coord).max()
            elif func == 'median':
                ds = ds.groupby(coord).median()
            elif func == 'sum':
                ds = ds.groupby(coord).sum()
            else: 
                raise ValueError(f'The summarizing function {func} is not implemented.')
    else:
        raise ValueError(f"Input for red_cols must be a subset of dimensions")
    return ds


#—————————————————————————————————————————————————————————
# AUXILIARY FUNCTIONS
#—————————————————————————————————————————————————————————

def apply_pca(df, n_comp):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(df_scaled)
    df_pca = pd.DataFrame(
        pca.transform(df_scaled),
        index=df.index,
        columns=[f"comp{i}" for i in range(n_comp)],
    )
    return df_pca
 
 
def get_cluster(df, n_cluster):
    #gm = GaussianMixture(n_components=n_cluster, random_state=42)
    #return gm.fit_predict(df)
    km = KMeans(n_clusters=n_cluster, n_init=12, random_state=42)
    return km.fit_predict(df)
 
 

def create_cluster_polygons(
    gdf, df, cluster_col="cluster", lon_coord="lon", lat_coord="lat",
    bandwidth=0.05, grid_res=300,
):
    clusters = np.unique(df[cluster_col])
    minx, miny, maxx, maxy = gdf.total_bounds
    lon_lin = np.linspace(minx, maxx, grid_res)
    lat_lin = np.linspace(maxy, miny, grid_res)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
    grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
 
    kde_scores = np.zeros((len(grid_points), len(clusters)))
    for i, c in enumerate(clusters):
        pts = df[df[cluster_col] == c][[lon_coord, lat_coord]].values
        kde = KernelDensity(bandwidth=bandwidth, kernel="exponential")
        kde.fit(pts)
        kde_scores[:, i] = np.exp(kde.score_samples(grid_points))
 
    labels = clusters[np.argmax(kde_scores, axis=1)]
    label_grid = labels.reshape(lon_grid.shape)
 
    res_x = (lon_lin.max() - lon_lin.min()) / (grid_res - 1)
    res_y = (lat_lin.max() - lat_lin.min()) / (grid_res - 1)
    transform = from_origin(
        lon_lin.min() - res_x / 2, lat_lin.max() + res_y / 2, res_x, res_y
    )
 
    mask = np.ones_like(label_grid, dtype=bool)
    polygons = []
    for geom, val in shapes(label_grid.astype(np.int16), mask=mask, transform=transform):
        polygons.append({"cluster": int(val), "geometry": shape(geom)})
 
    return gpd.GeoDataFrame(polygons, crs="EPSG:4326")


def create_voronoi_polygons(gdf, df_pca, cluster_var, lon_coord, lat_coord):
    """
    Build cluster polygons via Voronoi tessellation on cluster centroids.
    Each centroid is the mean lon/lat of pixels assigned to that cluster.
    Regions are clipped to the bounding envelope of the original geometry.
    """
    centroids = (
        df_pca.groupby(cluster_var)[[lon_coord, lat_coord]]
        .mean()
        .reset_index()
    )
 
    # Buffer envelope slightly so edge centroids get full Voronoi cells
    envelope = gdf.union_all().convex_hull.buffer(0.5)
    multipoint = MultiPoint(centroids[[lon_coord, lat_coord]].values)
    regions = voronoi_diagram(multipoint, envelope=envelope)
 
    # Match each Voronoi region to its nearest centroid
    centroid_points = [Point(r[lon_coord], r[lat_coord]) for _, r in centroids.iterrows()]
    polygons = []
    for region in regions.geoms:
        distances = [region.centroid.distance(p) for p in centroid_points]
        nearest = centroids.iloc[np.argmin(distances)]
        polygons.append({cluster_var: int(nearest[cluster_var]), "geometry": region})
 
    return gpd.GeoDataFrame(polygons, crs="EPSG:4326")
