import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from itertools import combinations
from matplotlib.font_manager import FontProperties
import os
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. Global Configuration
# ==============================================================================

# Set Times New Roman font
# Note: The font path is a system path, usually remains absolute or relies on system installation.
try:
    font_title = FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf", size=24) 
except:
    font_title = FontProperties(family='serif', weight='bold', size=24)

# Define Base Directories (Relative Paths)
# Assuming the script is running from the project root
DATA_DIR = "./data"
OUTPUT_DIR = "./results"

# Construct relative file paths
# Please adjust the sub-folders according to your actual 'data' folder structure
PATH_ZONES = os.path.join(DATA_DIR, "Fujian_Zones/Fuzhou.shp")
PATH_ROADS = os.path.join(DATA_DIR, "Fuzhou_Roads/Urban_Main_Roads.shp")
PATH_BUS = os.path.join(DATA_DIR, "Public_Transport/Fuzhou_Bus_Lines.shp")
PATH_SUBWAY = os.path.join(DATA_DIR, "Public_Transport/Fuzhou_Subway_Lines.shp")

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 1. Network Construction Logic
# ==============================================================================

def build_network_generic(zones_gdf, data_path, group_col, mode='default'):
    """Generic network construction function"""
    print(f"Reading data: {data_path} ...")
    gdf_data = gpd.read_file(data_path)
    
    # Data cleaning and preprocessing
    if mode == 'subway':
        # Extract line name, handle format like "Line 1 (Phase 1)"
        gdf_data['clean_group'] = gdf_data['name'].apply(lambda x: x.split('(')[0].strip() if pd.notna(x) else "Unknown")
        group_col = 'clean_group'
    elif mode == 'road':
        # Filter valid road types
        valid_types = ['motorway', 'motorway_link', 'secondary', 'secondary_link', 
                       'trunk', 'trunk_link', 'tertiary', 'tertiary_link', 'primary', 'primary_link']
        if 'fclass' in gdf_data.columns:
            gdf_data = gdf_data[gdf_data['fclass'].isin(valid_types)]
            
    # Ensure the grouping column exists and is not empty
    if group_col in gdf_data.columns:
        gdf_data = gdf_data[gdf_data[group_col].notna() & (gdf_data[group_col] != '')]
            
    # Spatial join
    if zones_gdf.crs != gdf_data.crs:
        gdf_data = gdf_data.to_crs(zones_gdf.crs)

    overlay = gpd.sjoin(gdf_data, zones_gdf, how="inner", predicate="intersects")
    groups = overlay.groupby(group_col)['QH_NAME'].apply(list)
    
    # Initialize Graph
    G = nx.Graph()
    G.add_nodes_from(zones_gdf['QH_NAME'].unique())
    
    # Add edges
    for _, zones in groups.items():
        unique_zones = list(set(zones))
        if len(unique_zones) >= 2:
            for u, v in combinations(unique_zones, 2):
                G.add_edge(u, v)
    return G

# ==============================================================================
# 2. Core Metrics Calculation Function
# ==============================================================================

def calculate_metrics(G, gdf_zones_proj, network_name):
    """
    Calculate all statistical metrics for the graph and return a dictionary
    """
    # 1. Basic counts
    num_total_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # 2. Active nodes
    active_node_list = [n for n, d in G.degree() if d > 0]
    num_active_nodes = len(active_node_list)
    active_pct = (num_active_nodes / num_total_nodes * 100) if num_total_nodes > 0 else 0
    
    # 3. Density and Degree
    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / num_total_nodes if num_total_nodes > 0 else 0
    
    # 4. Connected Components
    num_components = nx.number_connected_components(G)
    
    # 5. Coverage Area
    if active_node_list:
        active_zones = gdf_zones_proj[gdf_zones_proj['QH_NAME'].isin(active_node_list)]
        coverage_area_sq_m = active_zones.area.sum()
        coverage_area_km2 = coverage_area_sq_m / 1e6
        
        total_area_sq_m = gdf_zones_proj.area.sum()
        coverage_pct = (coverage_area_sq_m / total_area_sq_m) * 100
    else:
        coverage_area_km2 = 0
        coverage_pct = 0
        
    return {
        'Network': network_name,
        'Total Nodes': num_total_nodes,
        'Active Nodes': num_active_nodes,
        'Active Node Ratio (%)': round(active_pct, 2),
        'Edges': num_edges,
        'Avg Degree': round(avg_degree, 2),
        'Density': round(density, 4),
        'Components': num_components,
        'Coverage Area (km2)': round(coverage_area_km2, 2),
        'Coverage Ratio (%)': round(coverage_pct, 2)
    }

# ==============================================================================
# 3. Plotting Function (Modified: No Statistical Text)
# ==============================================================================

def plot_network_visual(ax, gdf_zones_proj, G, title, edge_color, stats_dict):
    """
    Plot network visualization (Clean Version)
    """
    # --- A. Prepare plotting data ---
    gdf_nodes = gdf_zones_proj.dissolve(by='QH_NAME')
    gdf_nodes['centroid'] = gdf_nodes.geometry.centroid
    pos = {name: (geo.x, geo.y) for name, geo in zip(gdf_nodes.index, gdf_nodes['centroid'])}
    
    lines = []
    for u, v in G.edges():
        if u in pos and v in pos:
            lines.append(LineString([pos[u], pos[v]]))
            
    # --- B. Plotting ---
    bounds = gdf_zones_proj.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    
    # 1. Base map
    gdf_zones_proj.plot(ax=ax, color='#FAFAFA', edgecolor='white', linewidth=1.5, zorder=1)
    gdf_zones_proj.plot(ax=ax, color='none', edgecolor='#B0B0B0', linewidth=0.8, zorder=2)
    
    # 2. Edges
    if lines:
        gdf_edges = gpd.GeoDataFrame(geometry=lines, crs=gdf_zones_proj.crs)
        # Use statistical data to determine line width and transparency
        n_e = stats_dict['Edges']
        alpha_val = 0.8 if n_e < 300 else (0.2 if n_e < 5000 else 0.1)
        lw_val = 1.0 if n_e < 300 else (0.5 if n_e < 5000 else 0.2)
        gdf_edges.plot(ax=ax, color=edge_color, linewidth=lw_val, alpha=alpha_val, zorder=3)
        
    # 3. Nodes
    valid_nodes = [n for n in G.nodes() if n in pos]
    active_coords = [(pos[n][0], pos[n][1]) for n in valid_nodes if G.degree(n) > 0]
    isolated_coords = [(pos[n][0], pos[n][1]) for n in valid_nodes if G.degree(n) == 0]
    
    if active_coords:
        ax.scatter(*zip(*active_coords), c='#2C3E50', s=25, edgecolors='white', lw=0.8, zorder=4)
    if isolated_coords:
        ax.scatter(*zip(*isolated_coords), c='#E0E0E0', s=15, edgecolors='none', zorder=3)

    # --- C. Decoration (Title only, removed stats text) ---
    ax.set_title(title, fontproperties=font_title, pad=12, color='black', loc='center')
    ax.axis('off')

# ==============================================================================
# 4. Main Program
# ==============================================================================

if __name__ == "__main__":
    # 1. Read Administrative Zones
    print(">>> Reading administrative zones...")
    gdf_zones = gpd.read_file(PATH_ZONES)
    if gdf_zones.crs is None: gdf_zones.set_crs(epsg=4326, inplace=True)
    
    # Convert to Projected Coordinate System
    gdf_zones_proj = gdf_zones.to_crs(epsg=3857)

    # 2. Build Networks
    print("\n>>> Building networks (Road uses fclass for grouping, please wait)...")
    G_road = build_network_generic(gdf_zones, PATH_ROADS, 'fclass', mode='road')
    
    # Check for column name compatibility (LineName vs name)
    # Note: Pre-check allows reading the file once to determine column name, 
    # but here we assume the logic inside build_network_generic handles path reading.
    # To properly handle the conditional column name without reading twice, we can handle it inside or just try.
    # For simplicity, passing the path and handling logic:
    bus_gdf_temp = gpd.read_file(PATH_BUS)
    bus_group_col = 'LineName' if 'LineName' in bus_gdf_temp.columns else 'name'
    G_bus = build_network_generic(gdf_zones, PATH_BUS, bus_group_col)
    
    G_subway = build_network_generic(gdf_zones, PATH_SUBWAY, 'name', mode='subway')

    # 3. Calculate metrics and collect data
    print("\n>>> Calculating statistical metrics...")
    all_stats_data = []

    stats_road = calculate_metrics(G_road, gdf_zones_proj, "Urban Road")
    stats_bus = calculate_metrics(G_bus, gdf_zones_proj, "Bus")
    stats_subway = calculate_metrics(G_subway, gdf_zones_proj, "Subway")
    
    all_stats_data.append(stats_road)
    all_stats_data.append(stats_bus)
    all_stats_data.append(stats_subway)

    # 4. Export CSV Table
    df_stats = pd.DataFrame(all_stats_data)
    csv_path = os.path.join(OUTPUT_DIR, "Network_Statistics.csv")
    df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Statistics table saved: {csv_path}")
    print(df_stats) 

    # 5. Plotting (Clean Version)
    print("\n>>> Generating images (Clean Version)...")
    fig, axes = plt.subplots(1, 3, figsize=(30, 10)) 
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0.02, hspace=0)

    # Plot Road
    plot_network_visual(axes[0], gdf_zones_proj, G_road, "(a) Urban Road", "#D62728", stats_road)
    # Plot Bus
    plot_network_visual(axes[1], gdf_zones_proj, G_bus, "(b) Bus", "#1F77B4", stats_bus)
    # Plot Subway
    plot_network_visual(axes[2], gdf_zones_proj, G_subway, "(c) Subway", "#FF7F0E", stats_subway)

    # Save Image
    img_path = os.path.join(OUTPUT_DIR, 'Semantic_Networks_Clean.png')
    plt.savefig(img_path, dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    print(f"✅ Image successfully saved: {img_path}")