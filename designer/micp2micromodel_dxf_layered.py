# fixed_micp2micromodel_dxf_layered.py (user-defined layer properties: depth, points, line width + per-layer color)

import numpy as np
from scipy.spatial import Voronoi
import ezdxf
from shapely.geometry import LineString, MultiLineString, box, Polygon
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import networkx as nx

# === CONFIGURATION ===
pixel_size_nm = 10
pixel_size_um = pixel_size_nm / 1000
width_mm = 3
height_mm = 1
chip_width_mm = 5
chip_height_mm = 2
borehole_diameter_um = 50
borehole_offset_um = 500
buffer_zone_width_um = 10
conn_line_width_um = 10

region_map_file = "throat_region_mapping.json"
output_dxf_file = "micromodel_design.dxf"
metrics_file = "micromodel_metrics.json"
depth_color_file = "../depth_to_color_map.json"

# === USER-DEFINED LAYER PROPERTIES ===
layer_properties = [
    {"depth_nm": 300, "n_points": 250, "width_um": 0.6},
    {"depth_nm": 300, "n_points": 250, "width_um": 0.45},
    {"depth_nm": 300, "n_points": 250, "width_um": 0.3}
]
n_layers = len(layer_properties)

# === DERIVED DIMENSIONS ===
width_px = int((width_mm * 1000) / pixel_size_um)
height_px = int((height_mm * 1000) / pixel_size_um)
chip_width_px = int((chip_width_mm * 1000) / pixel_size_um)
chip_height_px = int((chip_height_mm * 1000) / pixel_size_um)
offset_x = (chip_width_px - width_px) // 2 + int(buffer_zone_width_um / pixel_size_um)
offset_y = (chip_height_px - height_px) // 2

# === LAYERED POINT GENERATION ===
np.random.seed(42)
layer_bounds = np.linspace(0, height_px, n_layers + 1)
points = []
layer_tags = []
for i, props in enumerate(layer_properties):
    y_min, y_max = layer_bounds[i], layer_bounds[i + 1]
    layer_pts = np.column_stack((
        np.random.uniform(0, width_px, props["n_points"]),
        np.random.uniform(y_min, y_max, props["n_points"])
    ))
    points.append(layer_pts)
    layer_tags.extend([i] * props["n_points"])
points = np.vstack(points)
layer_tags = np.array(layer_tags)

# === VORONOI ===
vor = Voronoi(points)

# === CLIP VORONOI RIDGES ===
bbox = box(offset_x, offset_y, offset_x + width_px, offset_y + height_px)
throat_lines = []
ridge_vertices = []
layer_indices = []

for (i, j) in vor.ridge_vertices:
    if -1 in (i, j): continue
    v1 = vor.vertices[i] + [offset_x, offset_y]
    v2 = vor.vertices[j] + [offset_x, offset_y]
    seg = LineString([v1, v2]).intersection(bbox)
    if not seg.is_empty:
        mid_y = np.mean([v1[1], v2[1]])
        for s in seg.geoms if isinstance(seg, MultiLineString) else [seg]:
            throat_lines.append(np.array(s.coords))
            ridge_vertices.append((i, j))
            layer_idx = min(max(np.searchsorted(layer_bounds, (mid_y - offset_y), side='right') - 1, 0), n_layers - 1)
            layer_indices.append(layer_idx)

# === WRITE DXF AND JSON ===
doc = ezdxf.new(setup=True)
msp = doc.modelspace()
region_map = []
vertex_diam_map = defaultdict(list)
lengths_um = []
depths_um = []
widths_um = []

for idx, coords in enumerate(throat_lines):
    layer_idx = layer_indices[idx]
    props = layer_properties[layer_idx]
    d_nm = props["depth_nm"]
    d_um = d_nm / 1000
    w_um = props["width_um"]
    l = LineString(coords * pixel_size_um).length
    v = l * w_um * d_um
    p1_um = coords[0] * pixel_size_um
    p2_um = coords[1] * pixel_size_um
    line = LineString([p1_um, p2_um])
    poly = line.buffer(w_um / 2, cap_style=2)

    poly_coords = []
    if isinstance(poly, Polygon) and poly.is_valid:
        poly_coords = list(map(lambda p: [round(p[0], 2), round(p[1], 2)], poly.exterior.coords))
        hatch = msp.add_hatch()
        hatch.dxf.layer = "throats"
        hatch.dxf.color = layer_idx + 1
        hatch.paths.add_polyline_path(poly.exterior.coords, is_closed=True)

    msp.add_line(tuple(p1_um), tuple(p2_um), dxfattribs={
        "layer": "throat_lines",
        "color": layer_idx + 1,
        "lineweight": int(w_um * 100)
    })

    region_map.append({
        "depth_nm": float(d_nm),
        "width_um": float(w_um),
        "length_um": float(l),
        "volume_um3": float(v),
        "polygon": poly_coords
    })

    i, j = ridge_vertices[idx]
    vertex_diam_map[i].append(w_um)
    vertex_diam_map[j].append(w_um)

    lengths_um.append(l)
    depths_um.append(d_um)
    widths_um.append(w_um)

# === ADD JUNCTIONS ===
for idx, widths in vertex_diam_map.items():
    if len(widths) < 3: continue
    max_width = max(widths)
    if idx >= len(vor.vertices): continue
    x, y = vor.vertices[idx] + [offset_x, offset_y]
    center_um = (x * pixel_size_um, y * pixel_size_um)
    msp.add_circle(center_um, max_width / 2, dxfattribs={"layer": "junctions"})

# === BOREHOLES ===
borehole_centers_um = [
    (borehole_offset_um, borehole_offset_um),
    (chip_width_mm * 1000 - borehole_offset_um, borehole_offset_um),
    (borehole_offset_um, chip_height_mm * 1000 - borehole_offset_um),
    (chip_width_mm * 1000 - borehole_offset_um, chip_height_mm * 1000 - borehole_offset_um)
]
for center_um in borehole_centers_um:
    msp.add_circle(center_um, borehole_diameter_um / 2, dxfattribs={"layer": "boreholes"})

# === FRAME ===
frame_box = [
    (0, 0),
    (chip_width_mm * 1000, 0),
    (chip_width_mm * 1000, chip_height_mm * 1000),
    (0, chip_height_mm * 1000)
]
msp.add_lwpolyline(frame_box + [frame_box[0]], dxfattribs={"layer": "frame"}, close=True)

# === BUFFERS ===
left_rect = [
    ((offset_x * pixel_size_um) - buffer_zone_width_um, offset_y * pixel_size_um),
    (offset_x * pixel_size_um, offset_y * pixel_size_um),
    (offset_x * pixel_size_um, (offset_y + height_px) * pixel_size_um),
    ((offset_x * pixel_size_um) - buffer_zone_width_um, (offset_y + height_px) * pixel_size_um)
]
right_rect = [
    ((offset_x + width_px) * pixel_size_um, offset_y * pixel_size_um),
    ((offset_x + width_px) * pixel_size_um + buffer_zone_width_um, offset_y * pixel_size_um),
    ((offset_x + width_px) * pixel_size_um + buffer_zone_width_um, (offset_y + height_px) * pixel_size_um),
    ((offset_x + width_px) * pixel_size_um, (offset_y + height_px) * pixel_size_um)
]
for rect in [left_rect, right_rect]:
    poly = Polygon(rect)
    if poly.is_valid:
        hatch = msp.add_hatch(color=7)
        hatch.dxf.layer = "buffer"
        hatch.paths.add_polyline_path(list(poly.exterior.coords), is_closed=True)

# === CONNECTIONS ===
left_center = ((offset_x * pixel_size_um - buffer_zone_width_um / 2), ((offset_y + height_px / 2) * pixel_size_um))
right_center = (((offset_x + width_px) * pixel_size_um + buffer_zone_width_um / 2), ((offset_y + height_px / 2) * pixel_size_um))
for i in [0, 2]:
    c = np.array(borehole_centers_um[i])
    unit = (np.array(left_center) - c) / np.linalg.norm(np.array(left_center) - c)
    start = c + unit * borehole_diameter_um / 2
    poly = LineString([start, left_center]).buffer(conn_line_width_um / 2, cap_style=2)
    if poly.is_valid:
        hatch = msp.add_hatch(color=7)
        hatch.dxf.layer = "connections"
        hatch.paths.add_polyline_path(list(poly.exterior.coords), is_closed=True)
for i in [1, 3]:
    c = np.array(borehole_centers_um[i])
    unit = (np.array(right_center) - c) / np.linalg.norm(np.array(right_center) - c)
    start = c + unit * borehole_diameter_um / 2
    poly = LineString([start, right_center]).buffer(conn_line_width_um / 2, cap_style=2)
    if poly.is_valid:
        hatch = msp.add_hatch(color=7)
        hatch.dxf.layer = "connections"
        hatch.paths.add_polyline_path(list(poly.exterior.coords), is_closed=True)

# === EXPORT FILES ===
doc.saveas(output_dxf_file)
with open(region_map_file, "w") as f:
    json.dump(region_map, f, indent=2)

# === DEPTH-TO-COLOR MAP ===
cmap = plt.get_cmap("plasma")
depth_vals = sorted(set(p["depth_nm"] for p in layer_properties))
depth_to_color = {}
for idx, d in enumerate(depth_vals):
    rgba = cmap(idx / max(1, len(depth_vals) - 1))
    hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    depth_to_color[str(d)] = hex_color
with open(depth_color_file, "w") as f:
    json.dump(depth_to_color, f, indent=2)

# === METRICS ===
lengths_um = np.array(lengths_um)
depths_um = np.array(depths_um)
widths_um = np.array(widths_um)
throat_volumes_um3 = widths_um * depths_um * lengths_um
total_pore_volume_um3 = np.sum(throat_volumes_um3)
design_width_um = width_mm * 1000
design_height_um = height_mm * 1000
max_depth_um = np.max(depths_um)
design_volume_um3 = max_depth_um * design_width_um * design_height_um
porosity = total_pore_volume_um3 / design_volume_um3

# === GRAPH AND PERMEABILITY ===
G = nx.Graph()
for idx, v in enumerate(vor.vertices):
    x, y = v + [offset_x, offset_y]
    G.add_node(idx, pos=(x * pixel_size_um, y * pixel_size_um))
for i, j in ridge_vertices:
    if i == -1 or j == -1: continue
    p1 = vor.vertices[i] + [offset_x, offset_y]
    p2 = vor.vertices[j] + [offset_x, offset_y]
    G.add_edge(i, j, weight=np.linalg.norm((p1 - p2) * pixel_size_um))
x_positions = vor.vertices[:, 0] + offset_x
inlet_nodes = [idx for idx, x in enumerate(x_positions) if x <= offset_x + 0.05 * width_px]
outlet_nodes = [idx for idx, x in enumerate(x_positions) if x >= offset_x + 0.95 * width_px]
path_lengths = []
for inlet in inlet_nodes:
    lengths = nx.single_source_dijkstra_path_length(G, inlet, weight='weight')
    path_lengths.extend([lengths[o] for o in outlet_nodes if o in lengths])
tortuosity = (np.mean(path_lengths) / design_width_um) if path_lengths else 1.0
C = 5
d_um = np.min(depths_um)
perm_um2 = (porosity**3 * d_um**2) / (tortuosity**2 * (1 - porosity)**2 * C)
metrics = {
    "porosity": porosity,
    "total_pore_volume_um3": float(total_pore_volume_um3),
    "design_volume_um3": float(design_volume_um3),
    "max_depth_um": float(max_depth_um),
    "effective_depth_um": float(d_um),
    "tortuosity_estimate": float(tortuosity),
    "permeability_um2": float(perm_um2),
    "permeability_m2": float(perm_um2 * 1e-12),
    "permeability_mD": float(perm_um2 * 1e-12 / 9.869e-16),
    "permeability_nD": float(perm_um2 * 1e-12 / 9.869e-22),
    "pixel_size_nm": pixel_size_nm,
    "design_width_um": design_width_um,
    "design_height_um": design_height_um
}
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ DXF, metrics, and region mapping exported.")
