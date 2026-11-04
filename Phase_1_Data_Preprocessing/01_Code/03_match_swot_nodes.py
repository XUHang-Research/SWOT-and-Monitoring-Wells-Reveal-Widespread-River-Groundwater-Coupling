
from importlib import import_module as _load_module
C = _load_module("15_pipeline_config").from_environment()

import os
import sys
import time
import argparse
import collections
import csv
import json
import re
from pathlib import Path
import pandas as pd
import pyogrio
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
try:
    from tqdm import tqdm
except ImportError:
    # tqdm displays progress only; its absence from a notebook kernel does not affect calculations.
    def tqdm(iterable, *args, **kwargs):
        return iterable
from datetime import datetime
from pyproj import Geod
import numpy as np

print('Packages loaded')

# ================= Configuration =================
# Input and output paths
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from importlib import import_module as _import_module
_geodesic_candidate_search = _import_module("16_geodesic_candidate_search")
conservative_chord_radius = _geodesic_candidate_search.conservative_chord_radius
HERE = C.tables(5000)["directory"]
EXCEL_FILE = str(C.inputs["well_metadata"])
SHP_FOLDER = str(C.inputs["swot_merged_root"] / "Node")
OUTPUT_FOLDER = str(C.tables(5000)["directory"])
SWORD_NODE_FILE = str(C.inputs["sword_nodes"])
MERGE_ROOT = C.inputs["swot_merged_root"]
MERGE_MANIFEST = MERGE_ROOT / 'merge_granule_selection.csv'
MERGE_SUMMARY = MERGE_ROOT / 'merge_granule_selection_summary.json'
MAX_PRIOR_GEOMETRY_OFFSET_M = 100.0

# Parallel processing settings
NUM_WORKERS = C.workers["matching"]

# Matching thresholds and observation quality filters
DEFAULT_RADIUS_M = 5000.0        # Maximum well-to-prior-bank distance
MAX_ACCEPTABLE_UNCERTAINTY = 0.3 # Maximum uncertainty for node_q=1, metres
MIN_PRIOR_WIDTH = 50.0           # Minimum prior river width
MIN_PIXEL_COUNT = 8              # Minimum good-pixel count
MAX_DARK_FRAC = 0.5              # Maximum dark-water fraction
MIN_XTRK_DIST = 10000.0          # Minimum cross-track distance, 10 km
MAX_XTRK_DIST = 60000.0          # Maximum cross-track distance, 60 km

GEOD = Geod(ellps='WGS84')
SWORD_PRIOR = None
STATIC_CANDIDATES = None

OUTPUT_COLUMNS = [
    'site_no', 'wells_lon', 'wells_lat', 'shp_file',
    'reach_id', 'node_id',
    'river_lon', 'river_lat', 'river_p_lon', 'river_p_lat',
    'river_width', 'river_p_width', 'river_wse', 'river_wse_u',
    'river_node_q', 'n_good_pix', 'dark_frac', 'xtrk_dist',
    'prior_geometry_offset_m',
    'distance_adjusted_m', 'distance_m',
    'distance_prior_bank_m', 'distance_prior_center_m',
    'distance_observed_bank_m', 'distance_observed_center_m',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Static groundwater-river matching using SWORD v17b Node prior coordinates and widths."
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M,
                        help="Maximum well-to-prior-bank distance, metres.")
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER,
                        help="Output directory for matching results.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="Number of processes reading RiverSP Node files.")
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error("--radius must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def load_sword_prior():
    """Read one SWORD v17b Node prior so that coordinates and widths are fixed across dates."""
    print(f"Reading SWORD Node prior: {SWORD_NODE_FILE}")
    prior = pyogrio.read_dataframe(
        SWORD_NODE_FILE,
        columns=['node_id', 'x', 'y', 'width'],
        read_geometry=False,
        use_arrow=True,
    ).rename(columns={'x': 'p_lon', 'y': 'p_lat', 'width': 'p_width'})
    for column in ['node_id', 'p_lon', 'p_lat', 'p_width']:
        prior[column] = pd.to_numeric(prior[column], errors='coerce')
    prior = prior.dropna(subset=['node_id', 'p_lon', 'p_lat', 'p_width']).copy()
    prior['node_id'] = prior['node_id'].astype(np.int64)
    if prior['node_id'].duplicated().any():
        raise ValueError("Duplicate node_id values in SWORD Node prior; matching aborted.")
    prior = prior.set_index('node_id').sort_index()
    print(f"Unique SWORD Nodes: {len(prior):,}")
    return prior


def find_version_d_dates():
    """Verify the frozen merge contract and return its Version-D Node dates."""
    if not MERGE_MANIFEST.is_file() or not MERGE_SUMMARY.is_file():
        raise FileNotFoundError("Missing FlatGeobuf merge manifest or success summary; aborted.")
    with MERGE_SUMMARY.open(encoding='utf-8') as handle:
        summary = json.load(handle)
    if summary.get('status') != 'passed':
        raise RuntimeError(f"FlatGeobuf merge has not passed: status={summary.get('status')!r}")
    if set(summary.get('allowed_crids', [])) != {'PGD0', 'PID0'}:
        raise RuntimeError("FlatGeobuf merge is not restricted to the PGD0/PID0 inventory.")
    with MERGE_MANIFEST.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row['status'] == 'selected' and row['file_type'] == 'Node'
    ]
    if not selected or any(row['selected_crid'] not in {'PGD0', 'PID0'} for row in selected):
        raise RuntimeError("Node merge manifest is empty or contains a non-Version-D CRID.")
    dates = {row['date'] for row in selected}
    print(
        f"Verified Version-D-only FlatGeobuf: {len(selected):,} Node granules, "
        f"{len(dates):,} UTC start dates."
    )
    return dates


def lonlat_to_unit_xyz(lon, lat):
    """Convert lon/lat degrees to unit-sphere Cartesian coordinates."""
    lon_rad = np.radians(np.asarray(lon, dtype=float))
    lat_rad = np.radians(np.asarray(lat, dtype=float))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad))
    )


def build_static_candidates(final_radius_m):
    """Compute each well-to-SWORD candidate pair exactly once.

    Width bins give each cKDTree query a tight conservative search radius;
    ellipsoidal Geod distances then apply the exact prior-bank criterion.
    """
    prior = SWORD_PRIOR.reset_index()
    prior = prior[
        (prior['p_width'] > MIN_PRIOR_WIDTH) &
        prior['node_id'].astype(str).str.endswith('1')
    ].copy()
    wells = wells_df.dropna(subset=['site_no', 'wells_lon', 'wells_lat']).copy()
    wells = wells.drop_duplicates(subset=['site_no'], keep='first').reset_index(drop=True)
    wells_xyz = lonlat_to_unit_xyz(wells['wells_lon'], wells['wells_lat'])

    width_edges = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 32768]
    candidate_parts = []
    print(
        f"Precomputing static well-Node candidates: {len(wells):,} wells, "
        f"{len(prior):,} river Nodes with width > {MIN_PRIOR_WIDTH:g} m"
    )
    for lower, upper in zip(width_edges[:-1], width_edges[1:]):
        nodes = prior[(prior['p_width'] > lower) & (prior['p_width'] <= upper)].copy()
        if nodes.empty:
            continue
        tree = cKDTree(
            lonlat_to_unit_xyz(nodes['p_lon'], nodes['p_lat']),
            balanced_tree=False,
            compact_nodes=False,
        )
        conservative_radius_m = final_radius_m + upper / 2.0
        chord_radius = conservative_chord_radius(conservative_radius_m)
        neighbour_lists = tree.query_ball_point(
            wells_xyz, chord_radius, workers=-1, return_sorted=True
        )
        counts = np.fromiter(
            (len(indices) for indices in neighbour_lists),
            dtype=np.int64,
            count=len(neighbour_lists),
        )
        rough_count = int(counts.sum())
        if rough_count == 0:
            continue
        well_indices = np.repeat(np.arange(len(wells), dtype=np.int64), counts)
        node_indices = np.concatenate(neighbour_lists).astype(np.int64, copy=False)
        selected_nodes = nodes.iloc[node_indices]
        _, _, center_distance = GEOD.inv(
            wells['wells_lon'].to_numpy()[well_indices],
            wells['wells_lat'].to_numpy()[well_indices],
            selected_nodes['p_lon'].to_numpy(),
            selected_nodes['p_lat'].to_numpy(),
        )
        center_distance = np.asarray(center_distance)
        bank_distance = np.maximum(
            center_distance - selected_nodes['p_width'].to_numpy() / 2.0, 0.0
        )
        keep = bank_distance <= final_radius_m
        if not np.any(keep):
            continue
        kept_well_indices = well_indices[keep]
        kept_nodes = selected_nodes.iloc[np.flatnonzero(keep)]
        candidate_parts.append(
            pd.DataFrame(
                {
                    'site_no': wells['site_no'].to_numpy()[kept_well_indices],
                    'wells_lon': wells['wells_lon'].to_numpy()[kept_well_indices],
                    'wells_lat': wells['wells_lat'].to_numpy()[kept_well_indices],
                    'node_id': kept_nodes['node_id'].to_numpy(),
                    'river_p_lon': kept_nodes['p_lon'].to_numpy(),
                    'river_p_lat': kept_nodes['p_lat'].to_numpy(),
                    'river_p_width': kept_nodes['p_width'].to_numpy(),
                    'distance_prior_center_m': center_distance[keep],
                    'distance_prior_bank_m': bank_distance[keep],
                }
            )
        )
        print(
            f"  Prior width ({lower}, {upper}] m: "
            f"{rough_count:,} coarse candidates, {int(keep.sum()):,} retained after exact screening"
        )

    if not candidate_parts:
        return pd.DataFrame(columns=[
            'site_no', 'wells_lon', 'wells_lat', 'node_id',
            'river_p_lon', 'river_p_lat', 'river_p_width',
            'distance_prior_center_m', 'distance_prior_bank_m',
        ])
    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates['node_id'] = candidates['node_id'].astype(np.int64)
    duplicate_count = int(candidates.duplicated(['site_no', 'node_id']).sum())
    if duplicate_count:
        raise AssertionError(f"Duplicate keys in static well-Node candidates: {duplicate_count:,}")
    candidates = candidates.sort_values(
        ['node_id', 'site_no', 'distance_prior_bank_m'], kind='stable'
    ).reset_index(drop=True)
    print(
        f"Static candidates complete: {len(candidates):,} well-Node pairs, "
        f"{candidates['site_no'].nunique():,} wells, "
        f"{candidates['node_id'].nunique():,} Nodes"
    )
    return candidates

# ================= Step 0: initialization =================
start_time = time.time()
print("Starting observation quality screening...")

# Read well metadata
wells_df = pd.read_excel(EXCEL_FILE)
print(f"Wells within the study extent: {len(wells_df)}")

# ================= Read and screen river observations =================
def read_shp_file(shp_file):
    """
    Observation screening for groundwater-river comparisons:
    1. p_width > 50 m (prior river width).
    2. p_dam_id == 0 and topology type == 1 (exclude dams and connected lakes).
    3. ice_dyn_f and ice_clim_f must each belong to {0, -999}; unavailable flags are retained.
    4. n_good_pix >= 8.
    5. dark_frac < 0.5.
    6. 10 km <= abs(xtrk_dist) <= 60 km.
    7. Quality control: good, or suspect with low uncertainty.
    """
    shp_path = os.path.join(SHP_FOLDER, shp_file)
    try:
        # 1. Read only required fields to avoid decoding unused attributes.
        required_cols = [
            'reach_id', 'node_id', 'node_q', 'width', 'wse', 'wse_u', 
            'lon', 'lat', 'p_width', 'p_dam_id', 'ice_dyn_f', 'n_good_pix',
            'dark_frac', 'ice_clim_f', 'xtrk_dist'
        ]
        gdf = pyogrio.read_dataframe(
            shp_path,
            columns=required_cols,
            use_arrow=True,
        )
        
        missing_columns = [col for col in required_cols if col not in gdf.columns]
        if missing_columns:
            raise KeyError(f"{shp_file} missing required columns: {missing_columns}")

        # RiverSP Node geometry represents the product's prior Node position.
        # Retain it for version checks; final coordinates use the common SWORD v17b prior.
        gdf['_product_p_lon'] = gdf.geometry.x
        gdf['_product_p_lat'] = gdf.geometry.y

        # 2. Convert types and clean values
        for col in required_cols:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
        
        # Extract the final node_id digit as the waterbody topology type.
        # 1 = river, 3 = connected lake, 4 = dam
        gdf['water_type'] = gdf['node_id'].astype(str).str[-1]

        # Exclude missing required values
        core_cols = [
            'width', 'wse', 'wse_u', 'lon', 'lat', 'node_q', 'p_width',
            '_product_p_lon', '_product_p_lat', 'n_good_pix'
        ]
        gdf = gdf.dropna(subset=core_cols)
        
        gdf = gdf[
            np.isfinite(gdf['width']) & 
            np.isfinite(gdf['wse']) & 
            np.isfinite(gdf['wse_u']) & 
            np.isfinite(gdf['n_good_pix'])
        ]
        
        # 3. Apply observation quality screens
        gdf = gdf[
            (gdf['p_width'] > MIN_PRIOR_WIDTH) & 
            (gdf['width'] > 0) &                 
            (gdf['width'] < 100000) &            
            (gdf['wse'] > -1000) &   
            
            # Waterbody topology: retain rivers and exclude connected lakes and dams.
            (gdf['water_type'] == '1') & 
            
            # Physical environment: exclude dams and explicit ice; retain unavailable (-999) ice flags.
            (gdf['p_dam_id'] == 0) & 
            ((gdf['ice_dyn_f'] == 0) | (gdf['ice_dyn_f'] == -999)) & 
            ((gdf['ice_clim_f'] == 0) | (gdf['ice_clim_f'] == -999)) & 
            
            # Radar observation quality: low dark-water fraction and accepted cross-track range.
            (gdf['dark_frac'] >= 0) & (gdf['dark_frac'] < MAX_DARK_FRAC) &
            (gdf['xtrk_dist'].abs() >= MIN_XTRK_DIST) &
            (gdf['xtrk_dist'].abs() <= MAX_XTRK_DIST) &

            # Pixel count: at least MIN_PIXEL_COUNT (currently 8) good pixels.
            (gdf['n_good_pix'] >= MIN_PIXEL_COUNT) & 

            # Product quality classification
            (
                (gdf['node_q'] == 0) |           # Good
                (
                    (gdf['node_q'] == 1) &       # Suspect
                    (gdf['wse_u'] < MAX_ACCEPTABLE_UNCERTAINTY) # Accepted uncertainty limit
                )
            )
        ]

        if gdf.empty:
            return None

        # 4. Join accepted observations to the unique SWORD v17b Node prior.
        # A large offset between product geometry and v17b can indicate older Node IDs
        # or mixed database versions; exclude these rows to prevent incorrect matching.
        gdf['node_id'] = gdf['node_id'].astype(np.int64)
        prior = SWORD_PRIOR.reindex(gdf['node_id'].to_numpy())
        gdf['p_lon'] = prior['p_lon'].to_numpy()
        gdf['p_lat'] = prior['p_lat'].to_numpy()
        gdf['p_width'] = prior['p_width'].to_numpy()
        gdf = gdf.dropna(subset=['p_lon', 'p_lat', 'p_width']).copy()
        _, _, prior_offset = GEOD.inv(
            gdf['_product_p_lon'].to_numpy(),
            gdf['_product_p_lat'].to_numpy(),
            gdf['p_lon'].to_numpy(),
            gdf['p_lat'].to_numpy(),
        )
        gdf['prior_geometry_offset_m'] = np.asarray(prior_offset)
        gdf = gdf[
            (gdf['prior_geometry_offset_m'] <= MAX_PRIOR_GEOMETRY_OFFSET_M) &
            (gdf['p_width'] > MIN_PRIOR_WIDTH)
        ].copy()

        if gdf.empty:
            return None
        
        # Retain required columns
        out_cols = [
            'reach_id', 'node_id', 'lon', 'lat', 'p_lon', 'p_lat',
            'width', 'wse', 'wse_u', 'p_width', 'node_q', 'p_dam_id', 'n_good_pix',
            'dark_frac', 'xtrk_dist', 'prior_geometry_offset_m'
        ]
        gdf = gdf[out_cols].copy()
        
        return shp_file, gdf

    except Exception as e:
        # Propagate corrupt/schema-incompatible daily files to the parent
        # process.  Silently returning None here would produce an incomplete
        # multi-date table while the batch job still exits successfully.
        raise RuntimeError(f"Error reading or processing {shp_file}: {e}") from e

# ================= Calculation helpers =================
def meters_to_deg_window(lat_deg, r_m):
    lat_rad = np.radians(lat_deg)
    coslat = np.abs(np.cos(lat_rad))
    coslat = np.maximum(coslat, 0.01) 
    dlat = r_m / 111000.0
    dlon = r_m / (111000.0 * coslat)
    return float(dlat), float(dlon)

# ================= Per-task processing =================
def process_loaded_shp(shp_file, wells, max_half_width, gdf, final_radius_m):
    results = []

    # Convert to NumPy arrays
    r_reach_id = gdf['reach_id'].to_numpy()
    r_node_id = gdf['node_id'].to_numpy()
    r_lons = gdf['lon'].to_numpy()
    r_lats = gdf['lat'].to_numpy()
    r_p_lons = gdf['p_lon'].to_numpy()
    r_p_lats = gdf['p_lat'].to_numpy()
    r_width = gdf['width'].to_numpy()
    r_wse = gdf['wse'].to_numpy()
    r_wse_u = gdf['wse_u'].to_numpy()     
    r_p_width = gdf['p_width'].to_numpy() 
    r_node_q = gdf['node_q'].to_numpy()   
    r_n_pix = gdf['n_good_pix'].to_numpy()
    r_dark_frac = gdf['dark_frac'].to_numpy()
    r_xtrk_dist = gdf['xtrk_dist'].to_numpy()
    r_prior_geometry_offset = gdf['prior_geometry_offset_m'].to_numpy()

    # Candidate selection and final distances use common prior Node coordinates.
    # Observed longitude, latitude and width are retained only for quality audits.
    lat_order = np.argsort(r_p_lats, kind='mergesort')
    sorted_lats = r_p_lats[lat_order]

    # The latitude-sorted index reduces rectangular candidates; restore original
    # Node order to retain the full-scan result and the order of distance ties.
    del gdf 

    STATIC_FAST_RADIUS_M = final_radius_m + max_half_width + 500.0

    for _, well in wells.iterrows():
        wlon = float(well['wells_lon'])
        wlat = float(well['wells_lat'])
        
        dlat, dlon = meters_to_deg_window(wlat, STATIC_FAST_RADIUS_M)
        lat_start = np.searchsorted(sorted_lats, wlat - dlat, side='left')
        lat_stop = np.searchsorted(sorted_lats, wlat + dlat, side='right')
        candidate_indices = lat_order[lat_start:lat_stop]
        if candidate_indices.size == 0:
            continue

        candidate_indices = candidate_indices[
            (r_p_lons[candidate_indices] >= wlon - dlon) &
            (r_p_lons[candidate_indices] <= wlon + dlon)
        ]
        if candidate_indices.size == 0:
            continue
        candidate_indices = np.sort(candidate_indices)

        c_p_lons = r_p_lons[candidate_indices]
        c_p_lats = r_p_lats[candidate_indices]
        c_p_width = r_p_width[candidate_indices]

        _, _, dist_prior_center = GEOD.inv(
            np.full_like(c_p_lons, wlon),
            np.full_like(c_p_lats, wlat),
            c_p_lons, c_p_lats,
        )

        dist_prior_bank = np.maximum(dist_prior_center - (c_p_width / 2.0), 0.0)

        good_mask = dist_prior_bank <= final_radius_m
        if not np.any(good_mask):
            continue

        good_distances_prior_bank = dist_prior_bank[good_mask]
        good_distances_prior_center = dist_prior_center[good_mask]
        
        good_original_indices = candidate_indices[good_mask]

        order = np.lexsort((good_distances_prior_center, good_distances_prior_bank))

        final_indices = good_original_indices[order]
        final_dist_prior_bank = good_distances_prior_bank[order]
        final_dist_prior_center = good_distances_prior_center[order]

        selected_obs_lons = r_lons[final_indices]
        selected_obs_lats = r_lats[final_indices]
        _, _, final_dist_observed_center = GEOD.inv(
            np.full_like(selected_obs_lons, wlon),
            np.full_like(selected_obs_lats, wlat),
            selected_obs_lons,
            selected_obs_lats,
        )
        final_dist_observed_bank = np.maximum(
            final_dist_observed_center - r_width[final_indices] / 2.0, 0.0
        )

        for i, idx in enumerate(final_indices):
            results.append({
                'site_no': well['site_no'],
                'wells_lon': wlon, 
                'wells_lat': wlat,
                'shp_file': shp_file,
                'reach_id': r_reach_id[idx],     # Reach identifier for time-series grouping
                'node_id': r_node_id[idx],
                'river_lon': r_lons[idx],        # Observed longitude
                'river_lat': r_lats[idx],        # Observed latitude
                'river_p_lon': r_p_lons[idx],    # Fixed prior longitude
                'river_p_lat': r_p_lats[idx],    # Fixed prior latitude
                'river_width': r_width[idx],       
                'river_p_width': r_p_width[idx],   
                'river_wse': r_wse[idx],
                'river_wse_u': r_wse_u[idx],       
                'river_node_q': r_node_q[idx],     
                'n_good_pix': r_n_pix[idx],      
                'dark_frac': r_dark_frac[idx],   # Retained for quality audits
                'xtrk_dist': r_xtrk_dist[idx],   # Retained for quality audits
                'prior_geometry_offset_m': r_prior_geometry_offset[idx],
                # Compatibility aliases now refer explicitly to static prior distances.
                'distance_adjusted_m': float(final_dist_prior_bank[i]),
                'distance_m': float(final_dist_prior_center[i]),
                'distance_prior_bank_m': float(final_dist_prior_bank[i]),
                'distance_prior_center_m': float(final_dist_prior_center[i]),
                'distance_observed_bank_m': float(final_dist_observed_bank[i]),
                'distance_observed_center_m': float(final_dist_observed_center[i]),
            })

    return results


def process_shp_file(task):
    """Quality-filter one daily product and join precomputed static candidates."""
    shp_file = task
    read_result = read_shp_file(shp_file)
    if read_result is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _, gdf = read_result
    observation_columns = [
        'reach_id', 'node_id', 'lon', 'lat', 'width', 'wse', 'wse_u',
        'node_q', 'n_good_pix', 'dark_frac', 'xtrk_dist',
        'prior_geometry_offset_m',
    ]
    observations = gdf[observation_columns].copy()
    matched = observations.merge(
        STATIC_CANDIDATES,
        on='node_id',
        how='inner',
        sort=False,
        validate='many_to_many',
    )
    if matched.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _, _, observed_center = GEOD.inv(
        matched['wells_lon'].to_numpy(),
        matched['wells_lat'].to_numpy(),
        matched['lon'].to_numpy(),
        matched['lat'].to_numpy(),
    )
    matched['distance_observed_center_m'] = np.asarray(observed_center)
    matched['distance_observed_bank_m'] = np.maximum(
        matched['distance_observed_center_m'] - matched['width'] / 2.0, 0.0
    )
    # Backward-compatible aliases now deliberately mean static prior distances.
    matched['distance_m'] = matched['distance_prior_center_m']
    matched['distance_adjusted_m'] = matched['distance_prior_bank_m']
    matched['shp_file'] = shp_file
    matched = matched.rename(
        columns={
            'lon': 'river_lon',
            'lat': 'river_lat',
            'width': 'river_width',
            'wse': 'river_wse',
            'wse_u': 'river_wse_u',
            'node_q': 'river_node_q',
        }
    )
    matched = matched.sort_values(
        ['site_no', 'reach_id', 'distance_prior_bank_m', 'node_id'], kind='stable'
    )
    return matched[OUTPUT_COLUMNS]

# ================= Parallel execution =================
def run_parallel_processing(shp_files, num_workers=NUM_WORKERS):
    result_frames = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_iterable = executor.map(process_shp_file, shp_files)
        
        for result in tqdm(results_iterable, total=len(shp_files), desc="Joining daily observations to static candidates"):
            if result is not None and not result.empty:
                result_frames.append(result)
    if not result_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(result_frames, ignore_index=True)

# ================= Main entry point =================
if __name__ == '__main__':
    args = parse_args()
    FINAL_RADIUS_M = float(args.radius)
    OUTPUT_FOLDER = args.output_folder
    NUM_WORKERS = int(args.workers)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    SWORD_PRIOR = load_sword_prior()
    STATIC_CANDIDATES = build_static_candidates(FINAL_RADIUS_M)
    VERSION_D_DATES = find_version_d_dates()

    # Single pass: executor.map preserves the os.listdir input order.
    all_shp_files = sorted(f for f in os.listdir(SHP_FOLDER) if f.endswith(".fgb"))
    shp_files = [
        name for name in all_shp_files
        if (match := re.search(r"(20\d{6})", name)) and match.group(1) in VERSION_D_DATES
    ]
    selected_dates = [re.search(r"(20\d{6})", name).group(1) for name in shp_files]
    missing_dates = sorted(VERSION_D_DATES.difference(selected_dates))
    duplicate_dates = sorted(
        date for date, count in collections.Counter(selected_dates).items() if count != 1
    )
    if missing_dates or duplicate_dates:
        raise RuntimeError(
            "Version-D Node input set is incomplete or duplicated: "
            f"missing_dates={missing_dates[:20]}, duplicate_dates={duplicate_dates[:20]}"
        )
    print(f"Version-D-only Node FlatGeobuf files to process: {len(shp_files):,}")
    if shp_files:
        df_res = run_parallel_processing(shp_files, NUM_WORKERS)
    else:
        df_res = pd.DataFrame(columns=OUTPUT_COLUMNS)
        print("No FlatGeobuf files found.")

    # Save results
    radius_label = int(FINAL_RADIUS_M) if FINAL_RADIUS_M.is_integer() else FINAL_RADIUS_M
    output_filename = (
        f"1matched_results_all_country_nodeid_distance{radius_label}"
        "_minwidth50m——20260321_v5.txt"
    )
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    temporary_output_path = output_path + ".tmp"

    for column in OUTPUT_COLUMNS:
        if column not in df_res.columns:
            df_res[column] = np.nan
    df_res[OUTPUT_COLUMNS].to_csv(
        temporary_output_path, sep='\t', index=False, encoding='utf-8-sig'
    )
    os.replace(temporary_output_path, output_path)

    elapsed = time.time() - start_time
    print(f"\nMatching complete. Records: {len(df_res):,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Results saved to: {output_path}")
