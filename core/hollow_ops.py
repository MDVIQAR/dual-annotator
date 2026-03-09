import math

MIN_GAP = 4
MIN_INNER_DIM = 8
MIN_INNER_R = 4

def offset_rect(x, y, w, h, offset, direction='inner'):
    if direction == 'inner':
        new_w = max(MIN_INNER_DIM, w - 2 * offset)
        new_h = max(MIN_INNER_DIM, h - 2 * offset)
        new_x = x + (w - new_w) / 2
        new_y = y + (h - new_h) / 2
    else:
        new_w = w + 2 * offset
        new_h = h + 2 * offset
        new_x = x - offset
        new_y = y - offset
    return {'type': 'box', 'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h}

def offset_circle(cx, cy, r, offset, direction='inner'):
    if direction == 'inner':
        new_r = max(MIN_INNER_R, r - offset)
    else:
        new_r = r + offset
    return {'type': 'circle', 'cx': cx, 'cy': cy, 'r': new_r}

def offset_ellipse(cx, cy, rx, ry, offset, direction='inner'):
    if direction == 'inner':
        new_rx = max(MIN_INNER_R, rx - offset)
        new_ry = max(MIN_INNER_R, ry - offset)
    else:
        new_rx = rx + offset
        new_ry = ry + offset
    return {'type': 'ellipse', 'cx': cx, 'cy': cy, 'rx': new_rx, 'ry': new_ry}

def offset_polygon(points, offset, direction='inner'):
    if not points or len(points) < 3:
        return []
        
    try:
        from shapely.geometry import Polygon
        p = Polygon(points)
        if not p.is_valid:
            p = p.buffer(0)
            
        sign = -1.0 if direction == 'inner' else 1.0
        new_p = p.buffer(sign * offset, join_style=2)
        
        if not new_p.is_empty:
            if new_p.geom_type == 'MultiPolygon':
                new_p = max(new_p.geoms, key=lambda poly: poly.area)
            if new_p.geom_type == 'Polygon' and len(new_p.exterior.coords) > 3:
                return list(new_p.exterior.coords)[:-1]
    except Exception as e:
        print(f"Shapely offset failed: {e}")

    # Fallback to centroid approach if shapely fails or geometry becomes invalid/empty
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    new_points = []
    for px, py in points:
        nx, ny = px - cx, py - cy
        dist = math.hypot(nx, ny)
        if dist == 0:
            new_points.append((px, py))
            continue
        if direction == 'inner':
            new_dist = max(MIN_INNER_R, dist - offset)
        else:
            new_dist = dist + offset
        ratio = new_dist / dist
        new_points.append((cx + nx * ratio, cy + ny * ratio))
    return new_points

def offset_bezier(anchor_points, ctrl_points, offset, direction='inner', is_closed=True):
    if not is_closed:
        raise ValueError("Open bezier cannot be hollow. Close the curve first.")
    if not anchor_points:
        return [], []
        
    cx = sum(p[0] for p in anchor_points) / len(anchor_points)
    cy = sum(p[1] for p in anchor_points) / len(anchor_points)
    
    new_anchors = []
    ratios = []
    for px, py in anchor_points:
        nx, ny = px - cx, py - cy
        dist = math.hypot(nx, ny)
        if dist == 0:
            new_anchors.append((px, py))
            ratios.append(1.0)
            continue
        if direction == 'inner':
            new_dist = max(MIN_INNER_R, dist - offset)
        else:
            new_dist = dist + offset
        ratio = new_dist / dist
        ratios.append(ratio)
        new_anchors.append((cx + nx * ratio, cy + ny * ratio))
        
    new_ctrls = []
    if ctrl_points:
        for i, cp in enumerate(ctrl_points):
            if cp is None:
                new_ctrls.append(None)
                continue
            ratio = ratios[i // 2 % len(ratios)] if ratios else 1.0
            nx, ny = cp[0] - cx, cp[1] - cy
            new_ctrls.append((cx + nx * ratio, cy + ny * ratio))
            
    return new_anchors, new_ctrls
