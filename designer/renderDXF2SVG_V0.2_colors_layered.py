# renderDXF2SVG_dualmode.py
# Exports two SVGs: one colored by depth, one colored by width (both from region map)

import ezdxf
import svgwrite
import math
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import OrderedDict

# === SETTINGS ===
input_dxf = "micromodel_design.dxf"
region_json = "throat_region_mapping.json"
depth_color_map_json = "depth_to_color_map.json"
width_color_map_json = "width_to_color_map.json"
output_svg_depth = "micromodel_scaled_depth.svg"
output_svg_width = "micromodel_scaled_width.svg"
stroke_thin = 1.0
conn_stroke = 1.5
circle_stroke = 1.5
background_color = "white"

# === LOAD DATA ===
with open(region_json, "r") as f:
    region_map = json.load(f)

# === BUILD DEPTH COLORMAP FROM REGION MAP ===
depths = sorted(set(r["depth_nm"] for r in region_map if r["depth_nm"] > 0))
cmap_depth = plt.get_cmap("plasma")
depth_to_color = OrderedDict()
for idx, d in enumerate(depths):
    rgba = cmap_depth(idx / max(1, len(depths) - 1))
    hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    depth_to_color[str(d)] = hex_color
with open(depth_color_map_json, "w") as f:
    json.dump(depth_to_color, f, indent=2)

# === BUILD WIDTH COLORMAP FROM REGION MAP ===
widths = sorted(set(r["width_um"] for r in region_map if r["width_um"] > 0))
cmap_width = plt.get_cmap("viridis")
width_to_color = OrderedDict()
norm = plt.Normalize(min(widths), max(widths))
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_width)
for w in widths:
    rgba = scalar_map.to_rgba(w)
    hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    width_to_color[str(w)] = hex_color
with open(width_color_map_json, "w") as f:
    json.dump(width_to_color, f, indent=2)

# === COMMON DXF LOAD AND BOUNDING BOX ===
doc = ezdxf.readfile(input_dxf)
msp = doc.modelspace()

min_x, min_y = float("inf"), float("inf")
max_x, max_y = float("-inf"), float("-inf")

def update_bounds(x, y):
    global min_x, min_y, max_x, max_y
    min_x = min(min_x, x)
    min_y = min(min_y, y)
    max_x = max(max_x, x)
    max_y = max(max_y, y)

for e in msp:
    try:
        if e.dxftype() in {"LWPOLYLINE", "POLYLINE"}:
            for pt in e.get_points():
                update_bounds(pt[0], pt[1])
        elif e.dxftype() == "LINE":
            update_bounds(e.dxf.start.x, e.dxf.start.y)
            update_bounds(e.dxf.end.x, e.dxf.end.y)
        elif e.dxftype() == "CIRCLE":
            cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
            update_bounds(cx - r, cy - r)
            update_bounds(cx + r, cy + r)
        elif e.dxftype() == "HATCH":
            for path in e.paths:
                verts = getattr(path, "vertices", [])
                if not verts and hasattr(path, "edges"):
                    verts = [edge.start for edge in path.edges if hasattr(edge, "start") and edge.start]
                for v in verts:
                    update_bounds(v[0], v[1])
    except:
        pass

if math.isinf(min_x) or math.isinf(min_y):
    raise RuntimeError("❌ DXF appears empty or invalid.")

width = max_x - min_x
height = max_y - min_y

# === STYLE MAP ===
layer_styles = {
    "buffer": {"fill": "black", "stroke": "none"},
    "throats": {"fill": "white", "stroke": "none"},
    "connections": {"fill": "black", "stroke": "none"},
    "boreholes": {"fill": "black", "stroke": "none"},
    "junctions": {"fill": "#ffffff", "stroke": "none"},
    "frame": {"fill": "none", "stroke": "gray"}
}

# === FUNCTION TO GENERATE SVG ===
def generate_svg(filename, mode):
    dwg = svgwrite.Drawing(
        filename=filename,
        size=(f"{width}px", f"{height}px"),
        viewBox=f"{min_x} {min_y} {width} {height}",
        profile="full"
    )
    dwg.add(dwg.rect(insert=(min_x, min_y), size=(width, height), fill=background_color))

    throat_counter = 0
    for hatch in msp.query("HATCH"):
        try:
            layer = hatch.dxf.layer.lower()
            style = layer_styles.get(layer, {"fill": "white", "stroke": "black"})

            if layer == "throats":
                if mode == "depth":
                    depth = region_map[throat_counter].get("depth_nm")
                    fill_color = depth_to_color.get(str(depth), "#888888")
                elif mode == "width":
                    width_val = region_map[throat_counter].get("width_um", 0)
                    fill_color = width_to_color.get(str(width_val), "#888888")
                throat_counter += 1
            else:
                fill_color = style.get("fill", "white")

            for path in hatch.paths:
                verts = getattr(path, "vertices", [])
                if not verts and hasattr(path, "edges"):
                    verts = [edge.start for edge in path.edges if hasattr(edge, "start") and edge.start]
                if verts and verts[0] != verts[-1]:
                    verts.append(verts[0])
                pts = [(float(v[0]), float(v[1])) for v in verts]
                if len(pts) >= 3:
                    poly = dwg.polygon(points=pts, fill=fill_color, stroke=style.get("stroke", "black"), stroke_width=conn_stroke)
                    poly.update({"style": "vector-effect: non-scaling-stroke;"})
                    poly.set_desc(title=layer)
                    dwg.add(poly)
        except:
            continue

    for c in msp.query("CIRCLE"):
        try:
            layer = c.dxf.layer.lower()
            style = layer_styles.get(layer, {"fill": "white", "stroke": "black"})
            circ = dwg.circle(center=(c.dxf.center.x, c.dxf.center.y), r=c.dxf.radius,
                              fill=style["fill"], stroke=style["stroke"], stroke_width=circle_stroke)
            circ.update({"style": "vector-effect: non-scaling-stroke;"})
            circ.set_desc(title=layer)
            dwg.add(circ)
        except:
            continue

    dwg.save()
    print(f"✅ {mode.upper()} SVG render complete: {filename}")

# === GENERATE BOTH SVGs ===
generate_svg(output_svg_depth, "depth")
generate_svg(output_svg_width, "width")
