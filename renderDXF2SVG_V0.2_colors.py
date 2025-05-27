import ezdxf
import svgwrite
import math
import json

# === SETTINGS ===
input_dxf = "micromodel_design.dxf"
region_json = "throat_region_mapping.json"
output_svg = "micromodel_scaled.svg"
stroke_thin = 1.0
conn_stroke = 1.5
circle_stroke = 1.5
background_color = "black"

# === LOAD DEPTH TO COLOR MAP ===
with open("depth_to_color_map.json", "r") as f:
    depth_to_color = json.load(f)
depth_to_color = {round(float(k)): v for k, v in depth_to_color.items()}


# === LAYER COLORS (fallback) ===
layer_styles = {
    "buffer": {"fill": "blue", "stroke": "none"},
    "throats": {"fill": "white", "stroke": "none"},
    "connections": {"fill": "blue", "stroke": "none"},
    "boreholes": {"fill": "blue", "stroke": "white"},
    "junctions": {"fill": "white", "stroke": "none"},
    "frame": {"fill": "none", "stroke": "gray"}
}

# === LOAD MAPPING ===
with open(region_json, "r") as f:
    region_map = json.load(f)

# === LOAD DXF ===
doc = ezdxf.readfile(input_dxf)
msp = doc.modelspace()

# === BOUNDING BOX ===
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
                for v in getattr(path, "vertices", []):
                    update_bounds(v[0], v[1])
    except Exception as err:
        print(f"⚠️ Skipped during bounds check: {err}")

if math.isinf(min_x) or math.isinf(min_y):
    raise RuntimeError("❌ DXF appears empty or invalid.")

width = max_x - min_x
height = max_y - min_y

# === INIT SVG ===
dwg = svgwrite.Drawing(
    filename=output_svg,
    size=(f"{width}px", f"{height}px"),
    viewBox=f"{min_x} {min_y} {width} {height}",
    profile="full"
)
dwg.add(dwg.rect(insert=(min_x, min_y), size=(width, height), fill=background_color))

# === POLYLINES (frame + buffer) ===
for e in msp.query("LWPOLYLINE POLYLINE"):
    try:
        pts = [(pt[0], pt[1]) for pt in e.get_points()]
        if not e.closed and pts[0] != pts[-1]:
            pts.append(pts[0])
        layer = e.dxf.layer.lower()
        style = layer_styles.get(layer, {"stroke": "gray"})
        pl = dwg.polyline(points=pts, stroke=style.get("stroke", "gray"), fill="none", stroke_width=stroke_thin)
        pl.update({"style": "vector-effect: non-scaling-stroke;"})
        pl.set_desc(title=layer)
        dwg.add(pl)
    except Exception as err:
        print(f"⚠️ Skipped polyline: {err}")

# === LINES ===
for line in msp.query("LINE"):
    try:
        layer = line.dxf.layer.lower()
        style = layer_styles.get(layer, {"stroke": "gray"})
        start = (line.dxf.start.x, line.dxf.start.y)
        end = (line.dxf.end.x, line.dxf.end.y)
        dwg.add(dwg.line(start=start, end=end, stroke=style.get("stroke", "gray"), stroke_width=stroke_thin))
    except Exception as err:
        print(f"⚠️ Skipped line: {err}")

# === HATCHES (buffer, throats, connections) ===
throat_counter = 0
for hatch in msp.query("HATCH"):
    try:
        layer = hatch.dxf.layer.lower()
        style = layer_styles.get(layer, {"fill": "white", "stroke": "black"})

        if layer == "throats":
            depth_val = round(region_map[throat_counter]["depth_nm"])
            fill_color = depth_to_color.get(depth_val, "#999999")
            throat_counter += 1
        else:
            fill_color = style["fill"]

        for path in hatch.paths:
            verts = getattr(path, "vertices", [])
            pts = [(float(v[0]), float(v[1])) for v in verts]
            if len(pts) >= 3:
                poly = dwg.polygon(points=pts, fill=fill_color, stroke=style.get("stroke", "black"), stroke_width=conn_stroke)
                poly.update({"style": "vector-effect: non-scaling-stroke;"})
                poly.set_desc(title=layer)
                dwg.add(poly)
    except Exception as err:
        print(f"⚠️ Skipped hatch: {err}")

# === CIRCLES (boreholes etc.) ===
for c in msp.query("CIRCLE"):
    try:
        layer = c.dxf.layer.lower()
        style = layer_styles.get(layer, {"fill": "white", "stroke": "black"})
        circ = dwg.circle(center=(c.dxf.center.x, c.dxf.center.y), r=c.dxf.radius,
                          fill=style["fill"], stroke=style["stroke"], stroke_width=circle_stroke)
        circ.update({"style": "vector-effect: non-scaling-stroke;"})
        circ.set_desc(title=layer)
        dwg.add(circ)
    except Exception as err:
        print(f"⚠️ Skipped circle: {err}")

# === EXPORT ===
dwg.save()
print(f"✅ SVG render complete: {output_svg}")
