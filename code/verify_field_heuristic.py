#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════
# verify_field_heuristic.py —— "场作启发"定量验证（论文 IV 节配套脚本）
#
# 验证注记 2 的三个定量结论（在整形度量 w 下、α 取默认值）：
#   V1  纯场启发   h(n) = u(n)·r²            → 路径代价相对 C* 的劣化幅度
#   V2  取大式     h(n) = max(octile, u·r²)  → 同上（不可采纳剪枝的代价）
#   V3  加权式     h(n) = octile + κ(u_max−u)·r²
#       以反向 Dijkstra 求各节点真实剩余代价 C*(n)，按认证公式精确求解
#       可采纳性所允许的最大权重（闭式，无网格量化误差）：
#         加性形式 κ*_add = min_{v: pen(v)>0} [C*(v) − octile(v)] / pen(v)
#         取大形式 κ*_max = min_{v: pen(v)>0}  C*(v) / pen(v)
#       其中 pen(v) = (u_max − u(v))·r²（米）。同时报告绑定格：
#       全局绑定格（恒为目标格，结构必然）与排除目标格后的次优绑定格。
#
# 口径说明：
#   - 图与边权同主 pipeline（共享 _graph_struct 与 poisson 乘子，α 可指定）；
#   - h 以米为单位（u 为格²单位，乘 r² 换算），与边权同量纲；
#   - 搜索为 closed-set A*、目标首次出队即终止（与主 pipeline 语义一致），
#     纯 Python 实现：本脚本测代价与可采纳性，不测速度；
#   - 反向 Dijkstra 在转置图上以目标为源，得每个节点的真实剩余代价。
#
# 用法：
#   python verify_field_heuristic.py --maps-dir ./maps            # 正式协议（同 driver）
#   python verify_field_heuristic.py                              # 冒烟（内置测试地图）
# 输出：verify_field_heuristic_results.json + 控制台表格
# ═══════════════════════════════════════════════════════════════

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra as _cs_dijkstra

_HERE = Path(__file__).resolve().parent
_REPO = _HERE if (_HERE / "unified_planner_raw.py").exists() else Path("/mnt/agents/upload")
sys.path.insert(0, str(_REPO))
from unified_planner_raw import UnifiedPlanner, UConfig, GridSearch  # noqa: E402

SQRT2M1 = math.sqrt(2.0) - 1.0


def octile_field(ny, nx, gn, res):
    """按目标 gn 预计算 octile 启发场（米，拉平为一维）。"""
    gy, gx = gn // nx, gn % nx
    dy = np.abs(np.arange(ny)[:, None] - gy)
    dx = np.abs(np.arange(nx)[None, :] - gx)
    H = (np.maximum(dy, dx) + SQRT2M1 * np.minimum(dy, dx)) * res
    return np.ascontiguousarray(H.ravel())


def astar_closed(indptr, indices, weights, sn, gn, h):
    """closed-set A*，目标首次出队即终止。返回 (路径代价, settled, found)。"""
    N = indptr.shape[0] - 1
    g = np.full(N, np.inf)
    g[sn] = 0.0
    closed = np.zeros(N, dtype=bool)
    pq = [(float(h[sn]), 0.0, int(sn))]
    settled = 0
    while pq:
        _, gu, u = heapq.heappop(pq)
        if u == gn:
            return float(gu), settled, True
        if closed[u]:
            continue
        closed[u] = True
        settled += 1
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            if closed[v]:
                continue
            ng = gu + float(weights[k])
            if ng < g[v]:
                g[v] = ng
                heapq.heappush(pq, (ng + float(h[v]), ng, v))
    return float("inf"), settled, False


def build_graph(planner, cfg):
    """与主 pipeline 同一图结构与乘子（mode='poisson'），返回 (indptr, indices, weights)。"""
    gs = GridSearch(planner.sdf, cfg, planner.poisson, timer=planner.timer)
    rows, cols, steps, N = gs._graph_struct()
    mult = gs._mult_grid("poisson")
    w = steps * mult[cols]
    order = np.argsort(rows, kind="stable")
    rows, cols, w = rows[order], cols[order], w[order]
    counts = np.bincount(rows, minlength=N)
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, np.ascontiguousarray(cols), np.ascontiguousarray(w)


def snap_node(planner, p):
    c = planner.sdf.world_to_cell(p)
    f = planner.sdf.grid > 0  # 仅用障碍性吸附；安全裕度吸附与搜索一致由 free 掩膜保证
    fc = planner.sdf.nearest_free_cell(c, (planner.sdf.grid > planner.cfg.safety_margin))
    return fc[0] * planner.sdf.nx + fc[1]


def run_verification(map_data, start, goals, resolution, margin, alpha, tag):
    cfg = UConfig(sdf_resolution=resolution, safety_margin=margin, alpha=alpha,
                  use_cache=False, verbose=False)
    planner = UnifiedPlanner(map_data, cfg)
    ny, nx, res = planner.sdf.ny, planner.sdf.nx, planner.sdf.res
    u = np.ascontiguousarray(planner.poisson.u.ravel())
    u_max = float(u.max())

    indptr, indices, weights = build_graph(planner, cfg)
    beta = 2.0 / planner.poisson.u_mean

    records = []
    for gi, E in enumerate(goals):
        gn = snap_node(planner, E)
        sn = snap_node(planner, start)
        H = octile_field(ny, nx, gn, res)
        h_pure = u * res * res
        phi = (u_max - u) * res * res
        cstar = _cs_dijkstra(_transpose(indptr, indices, weights, ny * nx), directed=True, indices=gn)
        cstar = np.where(np.isfinite(cstar), cstar, np.inf)
        base = float(cstar[sn])
        cost0, set0, ok0 = astar_closed(indptr, indices, weights, sn, gn, H)
        cost1, set1, ok1 = astar_closed(indptr, indices, weights, sn, gn, h_pure)
        cost2, set2, ok2 = astar_closed(indptr, indices, weights, sn, gn, np.maximum(H, h_pure))

        # ── V3 认证（闭式）：pen>0 且 C* 有限的自由格上取精确下确界 ──
        free = planner.sdf.grid.ravel() > cfg.safety_margin
        m = free & (phi > 1e-12) & np.isfinite(cstar)
        ratio_add = np.where(m, (cstar - H) / phi, np.inf)
        ratio_max = np.where(m, cstar / phi, np.inf)
        k_add = max(0.0, float(ratio_add.min()))
        k_max = max(0.0, float(ratio_max.min()))
        ib_g = int(ratio_add.argmin())
        m_eg = m.copy()
        m_eg[gn] = False
        ib_2 = int(np.where(m_eg, ratio_add, np.inf).argmin())
        k_add_eg = float(np.where(m_eg, ratio_add, np.inf).min())

        def _cell(ix):
            return {"cell": [int(ix % nx), int(ix // nx)], "C_star": round(float(cstar[ix]), 4),
                    "octile": round(float(H[ix]), 4), "pen_m": round(float(phi[ix]), 4),
                    "u": round(float(u[ix]), 3)}

        rec = {
            "goal": [round(float(E[0]), 3), round(float(E[1]), 3)],
            "C_star_m": round(base, 4),
            "beta": round(beta, 6),
            "u_max": round(u_max, 3),
            "u_at_goal": round(float(u[gn]), 3),
            "octile_baseline": {"cost_m": round(cost0, 4), "settled": set0},
            "V1_pure_field": {"cost_m": round(cost1, 4) if ok1 else None,
                              "inflation_pct": round((cost1 - base) / base * 100, 2) if ok1 and np.isfinite(cost1) else None,
                              "settled": set1,
                              "settled_inflation_pct": round((set1 - set0) / set0 * 100, 2),
                              "found": ok1},
            "V2_max_form": {"cost_m": round(cost2, 4) if ok2 else None,
                            "inflation_pct": round((cost2 - base) / base * 100, 2) if ok2 and np.isfinite(cost2) else None,
                            "settled": set2,
                            "settled_inflation_pct": round((set2 - set0) / set0 * 100, 2),
                            "found": ok2},
            "V3_kappa_add_star": k_add,
            "V3_kappa_max_star": k_max,
            "V3_kappa_add_excluding_goal": k_add_eg,
            "V3_binding_goal": _cell(ib_g),
            "V3_binding_second": _cell(ib_2),
        }
        records.append(rec)
        print(f"  [{tag}] E{gi + 1} {rec['goal']}: C*={rec['C_star_m']:.2f}m | "
              f"V1 劣化 {rec['V1_pure_field']['inflation_pct']}%（扩展 {rec['V1_pure_field']['settled_inflation_pct']}%）| "
              f"V2 劣化 {rec['V2_max_form']['inflation_pct']}%（扩展 {rec['V2_max_form']['settled_inflation_pct']}%）| "
              f"κ*_add={k_add:.4g} κ*_max={k_max:.4g}", flush=True)
    summary = {
        "tag": tag, "alpha": alpha, "margin": margin, "resolution": resolution,
        "V1_inflation_mean_pct": round(float(np.mean([r["V1_pure_field"]["inflation_pct"] for r in records if r["V1_pure_field"]["inflation_pct"] is not None])), 2),
        "V2_inflation_mean_pct": round(float(np.mean([r["V2_max_form"]["inflation_pct"] for r in records if r["V2_max_form"]["inflation_pct"] is not None])), 2),
        "V1_settled_inflation_mean_pct": round(float(np.mean([r["V1_pure_field"]["settled_inflation_pct"] for r in records])), 2),
        "V2_settled_inflation_mean_pct": round(float(np.mean([r["V2_max_form"]["settled_inflation_pct"] for r in records])), 2),
        "kappa_add_star_max": round(float(max(r["V3_kappa_add_star"] for r in records)), 6),
        "kappa_max_star_max": round(float(max(r["V3_kappa_max_star"] for r in records)), 6),
        "records": records,
    }
    print(f"  [{tag}] 汇总：V1 平均劣化 {summary['V1_inflation_mean_pct']}%（扩展 {summary['V1_settled_inflation_mean_pct']}%），"
          f"V2 平均劣化 {summary['V2_inflation_mean_pct']}%（扩展 {summary['V2_settled_inflation_mean_pct']}%），"
          f"κ*_add 最大 {summary['kappa_add_star_max']}", flush=True)
    return summary


_t_cache = {}


def _transpose(indptr, indices, weights, N):
    key = (indptr.tobytes(), N)
    if key not in _t_cache:
        from scipy.sparse import csr_matrix
        G = csr_matrix((weights, indices, indptr), shape=(N, N))
        _t_cache[key] = G.T.tocsr()
    return _t_cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default=None, help="基准地图目录（同 driver 协议：首图起点 × 各图终点）")
    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="verify_field_heuristic_results.json")
    args = ap.parse_args()

    if args.maps_dir:
        from unified_planner_raw import load_maps_from_dir
        maps = load_maps_from_dir(args.maps_dir, require_points=True)
        S0 = tuple(maps[0][2])
        goals = [tuple(m[3]) for m in maps]
        tag = Path(args.maps_dir).name
        summaries = [run_verification(maps[0][1], S0, goals, args.resolution, args.margin, args.alpha, tag)]
    else:
        from unified_planner_raw import test_maps
        maps = test_maps()
        md, S = maps[0][1], maps[0][2]
        goals = [maps[0][3], (95.0, 55.0), (20.0, 10.0)]
        print("[smoke] 使用内置测试地图 A_open_field（正式数字须用 --maps-dir 跑基准地图）")
        summaries = [run_verification(md, S, goals, args.resolution, args.margin, args.alpha, "smoke")]

    out = Path(args.out)
    out.write_text(json.dumps({"config": {"alpha": args.alpha, "margin": args.margin,
                                          "resolution": args.resolution, "repo": str(_REPO)},
                               "summaries": summaries}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[DONE] -> {out.resolve()}")


if __name__ == "__main__":
    main()
