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


def create_cluster_coord(ds_orig, gdf_orig, cluster_var, method='within', check_var=None, k=6):
    ds = ds_orig.copy()
    gdf = gdf_orig.copy()

    # Extract lon and lat from the dataset
    lon_var, lat_var, time_var = get_coordinates(ds)
    points = ds[[lat_var, lon_var]].to_dataframe().reset_index()  

    # Create a DataFrame of points
    points = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points[lon_var], points[lat_var]),
        crs=gdf.crs
    )

    # Perform spatial join
    if method == 'within':
        points = gpd.sjoin(points, gdf[[cluster_var, "geometry"]], how="left", predicate="within")
        # Assign as coordinate to the dataset
        ds_points = points.set_index([lat_var, lon_var])[[cluster_var]]
        ds_points = ds_points.to_xarray().set_coords(cluster_var)
        ds = xr.merge([ds, ds_points])
        # Create a pixel assignation dataframe
        df_cluster = ds[[lon_var, lat_var, cluster_var]].to_dataframe()
        df_cluster = df_cluster.reset_index().dropna(subset=[cluster_var])

    elif method == 'nearest':
        # Match nearest coordinates from grid to centroid of polygons
        gdf['lon'] = gdf.centroid.x
        gdf['lat'] = gdf.centroid.y 
        # Filter only valid points by checking if check_var is all null
        if check_var:
            valid_count = (~ds[check_var].isnull()).sum(dim=time_var)
            points = (
                ds.assign_coords(valid_count=valid_count)
                [[lon_var, lat_var, 'valid_count']]
                .to_dataframe()
                .reset_index()
            )
            points = points[points['valid_count']!=0].copy()
            
        df_cluster = match_grid_points(points, gdf, k=k)
        df_cluster = df_cluster[[lon_var, lat_var, cluster_var]].copy()

        ds = select_coordinates(ds, df_cluster)
        #ds = ds.reset_index('points')
    
    df_cluster['method'] = method

    return ds, df_cluster


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
