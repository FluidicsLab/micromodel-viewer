import numpy as np
from scipy.spatial import Voronoi
import pandas as pd
import json
import ezdxf
from shapely.geometry import LineString, MultiLineString, box, Polygon
from collections import defaultdict

# === CONFIGURATION ===
pixel_size_nm = 100
pixel_size_um = pixel_size_nm / 1000
width_mm = 10
height_mm = 3
chip_width_mm = 15
chip_height_mm = 10
n_points = 4000
borehole_diameter_um = 200
borehole_offset_um = 500
buffer_zone_width_um = 50
conn_line_width_um = 50
throat_line_width_um = 0.5  # ← NEW user-configurable line width in microns


# === MICP BIN LOGIC ===
micp_bins = [
  {"depth_nm": 6.55, "vol_frac": 0.1343},
  {"depth_nm": 11.05, "vol_frac": 0.1355},
  {"depth_nm": 17.21, "vol_frac": 0.1647},
  {"depth_nm": 31.68, "vol_frac": 0.1993},
  {"depth_nm": 56.74, "vol_frac": 0.0890},
  {"depth_nm": 86.59, "vol_frac": 0.0548},
  {"depth_nm": 147.07, "vol_frac": 0.0819},
  {"depth_nm": 289.72, "vol_frac": 0.0545},
  {"depth_nm": 398.97, "vol_frac": 0.0320},
  {"depth_nm": 747.85, "vol_frac": 0.0541}
]


# === FILENAMES ===
region_map_file = "../../viewer/designs/FXL250703-UTA003/throat_region_mapping.json"
output_dxf_file = "../../viewer/designs/FXL250703-UTA001/micromodel_design.dxf"

# === DERIVED DIMENSIONS ===
width_px = int((width_mm * 1000) / pixel_size_um)
height_px = int((height_mm * 1000) / pixel_size_um)
chip_width_px = int((chip_width_mm * 1000) / pixel_size_um)
chip_height_px = int((chip_height_mm * 1000) / pixel_size_um)
offset_x = (chip_width_px - width_px) // 2
offset_y = (chip_height_px - height_px) // 2
offset_x += int(buffer_zone_width_um / pixel_size_um)

# === VORONOI ===
np.random.seed(42)
points = np.column_stack((np.random.uniform(0, width_px, n_points), np.random.uniform(0, height_px, n_points)))
vor = Voronoi(points)

# === CLIP VORONOI RIDGES ===
bbox = box(offset_x, offset_y, offset_x + width_px, offset_y + height_px)
throat_lines = []
ridge_vertices = []

for (i, j) in vor.ridge_vertices:
    if -1 in (i, j): continue
    v1 = vor.vertices[i] + [offset_x, offset_y]
    v2 = vor.vertices[j] + [offset_x, offset_y]
    segment = LineString([v1, v2])
    clipped = segment.intersection(bbox)
    if not clipped.is_empty:
        if clipped.geom_type == "LineString":
            throat_lines.append(np.array(clipped.coords))
            ridge_vertices.append((i, j))
        elif clipped.geom_type == "MultiLineString":
            for part in clipped.geoms:
                throat_lines.append(np.array(part.coords))
                ridge_vertices.append((i, j))

print(f"🔍 {len(throat_lines)} throat lines generated.")

# === LENGTH & AREA ===
width_um = throat_line_width_um
lengths_um = np.array([LineString(coords * pixel_size_um).length for coords in throat_lines])
areas_um2 = lengths_um * width_um

# === SORT + ASSIGN DEPTHS ===
sorted_idx = np.argsort(-areas_um2)
assigned_depths_nm = np.full(len(throat_lines), -1.0)
unassigned = set(range(len(throat_lines)))
total_area = np.sum(areas_um2)

for bin_info in micp_bins:
    d_nm = bin_info["depth_nm"]
    d_um = d_nm / 1000
    V_target = bin_info["vol_frac"] * total_area * d_um
    V_acc = 0.0
    for i in sorted_idx:
        if i in unassigned:
            V_i = areas_um2[i] * d_um
            if V_acc + V_i > V_target:
                break
            assigned_depths_nm[i] = d_nm
            V_acc += V_i
            unassigned.remove(i)

fallback_d_nm = micp_bins[-1]["depth_nm"]
for i in unassigned:
    assigned_depths_nm[i] = fallback_d_nm

depths_um = assigned_depths_nm / 1000
region_map = []
total_volume = 0.0
doc = ezdxf.new(setup=True)
msp = doc.modelspace()

# === TRACK JUNCTION WIDTHS ===
vertex_diam_map = defaultdict(list)

# === DRAW THROATS ===
for idx, coords in enumerate(throat_lines):
    l = lengths_um[idx]
    d = depths_um[idx]
    v = l * width_um * d

    p1_um = coords[0] * pixel_size_um
    p2_um = coords[1] * pixel_size_um
    line = LineString([p1_um, p2_um])
    poly = line.buffer(width_um / 2, cap_style=2)

    poly_coords = []
    if isinstance(poly, Polygon) and poly.is_valid:
        poly_coords = list(map(lambda p: [round(p[0], 2), round(p[1], 2)], poly.exterior.coords))
        hatch = msp.add_hatch(color=7)
        hatch.dxf.layer = "throats"
        hatch.paths.add_polyline_path(poly.exterior.coords, is_closed=True)

    msp.add_line(tuple(p1_um), tuple(p2_um), dxfattribs={"layer": "throat_lines", "color": 1})
    region_map.append({
        "depth_nm": float(assigned_depths_nm[idx]),
        "width_um": float(width_um),
        "length_um": float(l),
        "volume_um3": float(v),
        "polygon": poly_coords
    })
    total_volume += v

    # Junction mapping
    i, j = ridge_vertices[idx]
    vertex_diam_map[i].append(width_um)
    vertex_diam_map[j].append(width_um)

# === DRAW JUNCTIONS ===
for idx, widths in vertex_diam_map.items():
    if len(widths) < 3:
        continue
    max_width = max(widths)
    if idx >= len(vor.vertices):
        continue
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
    direction = np.array(left_center) - c
    unit = direction / np.linalg.norm(direction)
    start = c + unit * borehole_diameter_um / 2
    line = LineString([start, left_center])
    poly = line.buffer(conn_line_width_um / 2, cap_style=2)
    if poly.is_valid:
        hatch = msp.add_hatch(color=7)
        hatch.dxf.layer = "connections"
        hatch.paths.add_polyline_path(list(poly.exterior.coords), is_closed=True)

for i in [1, 3]:
    c = np.array(borehole_centers_um[i])
    direction = np.array(right_center) - c
    unit = direction / np.linalg.norm(direction)
    start = c + unit * borehole_diameter_um / 2
    line = LineString([start, right_center])
    poly = line.buffer(conn_line_width_um / 2, cap_style=2)
    if poly.is_valid:
        hatch = msp.add_hatch(color=7)
        hatch.dxf.layer = "connections"
        hatch.paths.add_polyline_path(list(poly.exterior.coords), is_closed=True)

# === FRAME ===
chip_box = [
    (0, 0),
    (chip_width_mm * 1000, 0),
    (chip_width_mm * 1000, chip_height_mm * 1000),
    (0, chip_height_mm * 1000)
]
msp.add_lwpolyline(chip_box + [chip_box[0]], dxfattribs={"layer": "frame"}, close=True)

# === EXPORT ===
doc.saveas(output_dxf_file)
with open(region_map_file, "w") as f:
    json.dump(region_map, f, indent=2)

print(f"🎉 DXF and region mapping exported to {output_dxf_file} and {region_map_file}")

# === AUTO-GENERATE DEPTH TO COLOR MAP ===
import matplotlib.pyplot as plt

# Create evenly spaced colors from a perceptually uniform colormap
cmap = plt.get_cmap("plasma")
num_bins = len(micp_bins)
depth_to_color_map = {}

for idx, bin_info in enumerate(micp_bins):
    depth = str(bin_info["depth_nm"])
    rgba = cmap(idx / max(1, num_bins - 1))  # Avoid division by zero
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255)
    )
    depth_to_color_map[depth] = hex_color

with open("../../viewer/designs/FXL250706-UTA002/depth_to_color_map.json", "w") as f:
    json.dump(depth_to_color_map, f, indent=2)

print("✅ Auto-generated depth_to_color_map.json with", num_bins, "entries.")

import networkx as nx

# === POROSITY CALCULATION ===
throat_volumes_um3 = width_um * depths_um * lengths_um
total_pore_volume_um3 = np.sum(throat_volumes_um3)

design_width_um = width_mm * 1000
design_height_um = height_mm * 1000
max_depth_um = np.max(depths_um)
design_volume_um3 = max_depth_um * design_width_um * design_height_um

porosity = total_pore_volume_um3 / design_volume_um3
print(f"🧮 Porosity = {porosity:.4f}")

# === EFFECTIVE DIAMETER ===
d_um = np.min(depths_um)  # minimum depth as diameter proxy
C = 5  # Kozeny constant

# === BUILD GRAPH FROM VORONOI RIDGES ===
G = nx.Graph()
for idx, v in enumerate(vor.vertices):
    x, y = v + [offset_x, offset_y]
    G.add_node(idx, pos=(x * pixel_size_um, y * pixel_size_um))

for i, j in ridge_vertices:
    if i == -1 or j == -1:
        continue
    p1 = vor.vertices[i] + [offset_x, offset_y]
    p2 = vor.vertices[j] + [offset_x, offset_y]
    length_um = np.linalg.norm((p1 - p2) * pixel_size_um)
    G.add_edge(i, j, weight=length_um)

# === IDENTIFY INLET AND OUTLET NODES ===
x_positions = vor.vertices[:, 0] + offset_x
inlet_threshold = offset_x + 0.05 * width_px  # 5% inside left boundary (pixels)
outlet_threshold = offset_x + 0.95 * width_px  # 5% inside right boundary (pixels)

inlet_nodes = [idx for idx, x in enumerate(x_positions) if x <= inlet_threshold]
outlet_nodes = [idx for idx, x in enumerate(x_positions) if x >= outlet_threshold]

# === COMPUTE SHORTEST PATHS FROM INLETS TO OUTLETS ===
path_lengths = []
for inlet in inlet_nodes:
    lengths = nx.single_source_dijkstra_path_length(G, inlet, weight='weight')
    outlet_lengths = [lengths[outlet] for outlet in outlet_nodes if outlet in lengths]
    path_lengths.extend(outlet_lengths)

if len(path_lengths) == 0:
    print("⚠️ No paths found between inlet and outlet nodes! Setting tortuosity = 1.0")
    tortuosity = 1.0
else:
    avg_path_length = np.mean(path_lengths)
    tortuosity = avg_path_length / design_width_um
    tortuosity = max(tortuosity, 1.0)
print(f"🧠 Graph-based tortuosity = {tortuosity:.3f}")

# === PERMEABILITY CALCULATION USING GRAPH TORTUOSITY ===
numerator = porosity**3 * d_um**2
denominator = tortuosity**2 * (1 - porosity)**2 * C
permeability_um2 = numerator / denominator
permeability_m2 = permeability_um2 * 1e-12
permeability_mD = permeability_m2 / 9.869e-16
permeability_nD = permeability_m2 / 9.869e-22

print(f"🧮 Permeability (min-depth based):")
print(f"   ≈ {permeability_m2:.3e} m²")
print(f"   ≈ {permeability_mD:.2f} mD")
print(f"   ≈ {permeability_nD:.2f} nD")

# === EXPORT METRICS TO JSON FILE ===
metrics = {
    "porosity": porosity,
    "total_pore_volume_um3": float(total_pore_volume_um3),
    "design_volume_um3": float(design_volume_um3),
    "max_depth_um": float(max_depth_um),
    "effective_depth_um": float(d_um),
    "tortuosity_estimate": float(tortuosity),
    "permeability_um2": float(permeability_um2),
    "permeability_m2": float(permeability_m2),
    "permeability_mD": float(permeability_mD),
    "permeability_nD": float(permeability_nD),
    "pixel_size_nm": pixel_size_nm,
    "design_width_um": design_width_um,
    "design_height_um": design_height_um
}
with open("../../micromodel_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("📁 Exported micromodel_metrics.json with updated tortuosity and permeability")















