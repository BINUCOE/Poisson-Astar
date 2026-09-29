#!/usr/bin/env python3
"""
map_extractor.py — 地图提取器 v2.1（边框标定 + 纯占据范围 + 几何精修）

v2.1 改进（锦上添花，不粘连）：
  1. 闭运算核从 5×5 椭圆降级为 3×3 矩形 —— 保留补缝能力，消除跨障碍物粘连与磨角。
  2. 圆拟合改用最小二乘（Kåsa），替代面积等效半径 —— 圆更圆，半径不受膨胀影响。
  3. 多边形 RDP epsilon 从 2.0 收紧到 1.0，并增加"直边回归精修"：
     对每条边回到原始轮廓做 PCA 最小二乘直线拟合，再用相邻拟合直线的交点作为精确顶点。
     输出边绝对直、角绝对锐。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. 边框检测
# ═══════════════════════════════════════════════════════════════

def detect_frame(img_bgr: np.ndarray, dark_thresh: int = 40, min_run: float = 0.3) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < dark_thresh
    h, w = dark.shape
    
    def _line_positions(proj: np.ndarray, length: int) -> List[float]:
        idx = np.where(proj > min_run * length)[0]
        if len(idx) == 0:
            return []
        groups, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 3:
                groups.append((start, a))
                start = b
        groups.append((start, idx[-1]))
        return [(a + b) / 2.0 for a, b in groups]
    
    v_lines = _line_positions(dark.sum(axis=0), h)
    h_lines = _line_positions(dark.sum(axis=1), w)
    if len(v_lines) < 2 or len(h_lines) < 2:
        raise ValueError("未能检出完整矩形边框")
    x_left, x_right = min(v_lines), max(v_lines)
    y_top, y_bottom = min(h_lines), max(h_lines)
    if x_right - x_left < 100 or y_bottom - y_top < 100:
        raise ValueError(f"边框尺寸异常: {(x_left, y_top, x_right, y_bottom)}")
    return x_left, y_top, x_right, y_bottom


# ═══════════════════════════════════════════════════════════════
# 2. 障碍物掩码（3×3 矩形核保守补缝）
# ═══════════════════════════════════════════════════════════════

def obstacle_mask(img_bgr: np.ndarray, frame: Tuple[int, int, int, int]) -> np.ndarray:
    xl, yt, xr, yb = [int(round(v)) for v in frame]
    roi = img_bgr[yt:yb + 1, xl:xr + 1].astype(np.int16)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    mask = (((b - r) > 25) & (b >= g)).astype(np.uint8) * 255
    # 3×3 矩形核：只补 1 像素级抗锯齿缺口，不跨障碍物桥接
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) >= 150:
            cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


# ═══════════════════════════════════════════════════════════════
# 3. 起终点检测
# ═══════════════════════════════════════════════════════════════

def detect_start_end(img_bgr: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([90, 255, 255]))
    red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
                         cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255])))
    
    def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < 200:
            return None
        return float(cents[i][0]), float(cents[i][1])
    
    return _centroid(green), _centroid(red)


# ═══════════════════════════════════════════════════════════════
# 4. 几何精修工具
# ═══════════════════════════════════════════════════════════════

def fit_circle_least_squares(pts: np.ndarray) -> Tuple[float, float, float]:
    """Kåsa 最小二乘圆拟合。pts: Nx2。返回 (cx, cy, r)。"""
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    B = -(x ** 2 + y ** 2)
    sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    D, E, F = sol
    cx = -D / 2.0
    cy = -E / 2.0
    r = math.sqrt(max(0.0, cx ** 2 + cy ** 2 - F))
    return cx, cy, r


def refine_polygon_edges(contour: np.ndarray, approx_vertices: np.ndarray) -> np.ndarray:
    """
    对 RDP 输出的多边形顶点做直边回归精修。
    1. 找到每个 RDP 顶点在原始轮廓中的最近点索引；
    2. 按索引区间提取每条边对应的原始轮廓点；
    3. 对每条边的点做 PCA 最小二乘直线拟合；
    4. 用相邻拟合直线的交点作为新的精确顶点。
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    verts = approx_vertices.reshape(-1, 2).astype(np.float64)
    n = len(verts)
    if n < 3:
        return verts
    
    # 每个 RDP 顶点在原始轮廓中的最近索引
    idx_map = []
    for v in verts:
        dists = np.sum((pts - v) ** 2, axis=1)
        idx_map.append(int(np.argmin(dists)))
    
    fitted_lines = []  # (a, b, c) 表示 a*x + b*y + c = 0, 且 a²+b²=1
    
    for i in range(n):
        i1 = idx_map[i]
        i2 = idx_map[(i + 1) % n]
        # 按轮廓顺序提取边对应的原始点
        if i2 >= i1:
            edge_pts = pts[i1:i2 + 1]
        else:
            edge_pts = np.vstack([pts[i1:], pts[:i2 + 1]])
        
        v1, v2 = verts[i], verts[(i + 1) % n]
        if len(edge_pts) < 3:
            # 点太少，退化为原边所在直线
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            norm = np.hypot(dx, dy)
            if norm < 1e-6:
                fitted_lines.append((1.0, 0.0, -v1[0]))
            else:
                nx, ny = -dy / norm, dx / norm
                c = -(nx * v1[0] + ny * v1[1])
                fitted_lines.append((nx, ny, c))
            continue
        
        # PCA 最小二乘直线拟合：最小特征值对应的特征向量为法向量
        mx, my = np.mean(edge_pts[:, 0]), np.mean(edge_pts[:, 1])
        X = edge_pts[:, 0] - mx
        Y = edge_pts[:, 1] - my
        C = np.cov(X, Y, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(C)
        nv = eigvecs[:, 0]  # 最小特征值
        a, b = nv[0], nv[1]
        c = -(a * mx + b * my)
        norm = np.hypot(a, b)
        if norm > 0:
            a, b, c = a / norm, b / norm, c / norm
        fitted_lines.append((a, b, c))
    
    # 相邻拟合直线的交点作为新顶点
    new_verts = []
    for i in range(n):
        a1, b1, c1 = fitted_lines[i]
        a2, b2, c2 = fitted_lines[(i - 1) % n]  # 前一条边
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            new_verts.append(verts[i].tolist())
        else:
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            new_verts.append([x, y])
    
    return np.array(new_verts)


# ═══════════════════════════════════════════════════════════════
# 5. 形状提取（圆度分类 + 几何精修）
# ═══════════════════════════════════════════════════════════════

def extract_shapes(mask: np.ndarray, circularity_thresh: float = 0.80, poly_epsilon_px: float = 1.0) -> List[Dict]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1e-6:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        _, (bw, bh), _ = cv2.minAreaRect(c)
        extent = area / max(bw * bh, 1e-6)
        
        if circularity >= circularity_thresh and extent < 0.83:
            # 最小二乘圆拟合，替代面积等效半径
            pts = c.reshape(-1, 2).astype(np.float64)
            cx, cy, r = fit_circle_least_squares(pts)
            shapes.append({"shape_type": "circle", "center_px": [cx, cy], "radius_px": r, "area_px": area})
        else:
            # RDP 收紧 + 直边回归精修
            approx = cv2.approxPolyDP(c, poly_epsilon_px, True)
            verts = approx.reshape(-1, 2).astype(float)
            if len(verts) < 3:
                continue
            verts = refine_polygon_edges(c, verts)
            shapes.append({"shape_type": "polygon", "vertices_px": verts.tolist(), "area_px": area})
    return shapes


# ═══════════════════════════════════════════════════════════════
# 6. 坐标映射
# ═══════════════════════════════════════════════════════════════

class FrameMapper:
    def __init__(self, frame: Tuple[int, int, int, int]):
        self.xl, self.yt, self.xr, self.yb = [float(v) for v in frame]
        self.fw = self.xr - self.xl
        self.fh = self.yb - self.yt
        aspect = self.fw / self.fh
        if abs(aspect - 1.0) <= 0.03:
            self.W, self.H = 100.0, 100.0
        elif aspect > 1.0:
            self.W, self.H = 100.0, 100.0 / aspect
        else:
            self.W, self.H = 100.0 * aspect, 100.0
        self.sx = self.W / self.fw
        self.sy = self.H / self.fh
        self.scale = (self.sx + self.sy) / 2.0
    
    def point(self, px: float, py: float) -> Tuple[float, float]:
        x = (px - self.xl) * self.sx
        y = (self.yb - py) * self.sy
        return x, y
    
    def length(self, d_px: float) -> float:
        return d_px * self.scale


# ═══════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════

def extract_map(image_path: str) -> Dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    
    frame = detect_frame(img)
    mapper = FrameMapper(frame)
    mask = obstacle_mask(img, frame)
    shapes = extract_shapes(mask)
    start_px, end_px = detect_start_end(img)
    
    xl, yt = frame[0], frame[1]
    
    obstacles = []
    for i, s in enumerate(shapes):
        if s["shape_type"] == "circle":
            cx_w, cy_w = mapper.point(s["center_px"][0] + xl, s["center_px"][1] + yt)
            r_w = mapper.length(s["radius_px"])
            obstacles.append(
                {"uid": f"circ_{i:03d}", "shape_type": "circle", "center": [round(cx_w, 3), round(cy_w, 3)],
                 "radius": round(r_w, 3), "area": round(math.pi * r_w ** 2, 3)})
        else:
            verts_w = [mapper.point(v[0] + xl, v[1] + yt) for v in s["vertices_px"]]
            obstacles.append({"uid": f"poly_{i:03d}", "shape_type": "polygon",
                              "vertices": [[round(x, 3), round(y, 3)] for x, y in verts_w],
                              "area": round(s["area_px"] * mapper.scale ** 2, 3)})
    
    def _pt(p):
        if p is None:
            return {"coordinates": None, "is_set": False}
        xw, yw = mapper.point(p[0], p[1])
        return {"coordinates": [round(xw, 3), round(yw, 3)], "is_set": True}
    
    coverage = float(np.sum(mask > 0)) / mask.size * 100.0
    return {"meta": {"source": "image", "image_path": str(image_path), "extractor": "map_extractor v2.1",
                     "frame_px": [int(v) for v in frame]},
            "map": {"width": round(mapper.W, 3), "height": round(mapper.H, 3),
                    "total_area": round(mapper.W * mapper.H, 3)},
            "obstacles": {"count": len(obstacles), "coverage_percent": round(coverage, 2), "items": obstacles},
            "points": {"start": _pt(start_px), "end": _pt(end_px)}, "_mask": mask, "_frame": frame, "_img": img, }


# ═══════════════════════════════════════════════════════════════
# 8. 校验可视化
# ═══════════════════════════════════════════════════════════════

def render_preview(result: Dict, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle, Polygon as MplPolygon
    
    img, frame = result["_img"], result["_frame"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    xl, yt, xr, yb = frame
    axes[0].add_patch(plt.Rectangle((xl, yt), xr - xl, yb - yt, fill=False, edgecolor="magenta", lw=2))
    axes[0].set_title("Original + detected frame")
    axes[0].axis("off")
    
    ax = axes[1]
    W, H = result["map"]["width"], result["map"]["height"]
    for o in result["obstacles"]["items"]:
        if o["shape_type"] == "circle":
            ax.add_patch(MplCircle(o["center"], o["radius"], facecolor="#aec7e8", edgecolor="#1f4e79", lw=1.5))
        else:
            ax.add_patch(MplPolygon(o["vertices"], closed=True, facecolor="#aec7e8", edgecolor="#1f4e79", lw=1.5))
    s, e = result["points"]["start"], result["points"]["end"]
    if s["is_set"]:
        ax.plot(*s["coordinates"], "o", color="green", ms=12, zorder=5)
    if e["is_set"]:
        ax.plot(*e["coordinates"], "o", color="red", ms=12, zorder=5)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(f"Extracted map  {W:.1f}x{H:.1f}, "
                 f"{result['obstacles']['count']} obstacles, "
                 f"cover {result['obstacles']['coverage_percent']}%")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def strip_internal(result: Dict) -> Dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════
# 9. 一键运行入口
# ═══════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).parent.resolve()
    figs_dir = script_dir / "Figs"
    
    if not figs_dir.exists():
        print(f"[错误] 找不到 Figs 文件夹: {figs_dir}")
        return
    
    images = sorted(figs_dir.glob("*.jpg")) + sorted(figs_dir.glob("*.JPG"))
    if not images:
        print(f"[警告] Figs 文件夹中没有找到 .jpg 图片")
        return
    
    out_dir = script_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[信息] 找到 {len(images)} 张图片，输出目录: {out_dir}")
    print("-" * 50)
    
    for p in images:
        name = p.stem
        print(f"[extract] {p.name}")
        try:
            res = extract_map(str(p))
            json_path = out_dir / f"{name}_map.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(strip_internal(res), f, indent=2, ensure_ascii=False)
            render_preview(res, str(out_dir / f"{name}_preview.png"))
            r = res
            print(f"  frame={r['meta']['frame_px']}  world={r['map']['width']}x{r['map']['height']}"
                  f"  obstacles={r['obstacles']['count']}  coverage={r['obstacles']['coverage_percent']}%"
                  f"  S={r['points']['start']['coordinates']}  E={r['points']['end']['coordinates']}")
            print(f"  → {json_path}")
        except Exception as e:
            print(f"  [失败] {e}")
        print()


if __name__ == "__main__":
    main()
