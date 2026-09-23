#!/usr/bin/env python3
"""
test_pipeline_raw.py — 纯搜索层（raw）driver + viz（standalone）

统一 S0（maps 第一张图起点）× N 终点 × 4 引擎 + Poisson α 扫参 →
multi_dest_results_raw.json → 出图到 <out_dir>/。只依赖
unified_planner_raw.py（纯搜索层规划器），不含任何精化口径。
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from unified_planner_raw import PLANNER_VERSION, UnifiedPlanner, UConfig  # noqa: E402

# ═══════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════
IDE = dict(maps_dir="./maps",  # 地图 JSON 目录（相对本脚本）
           out_dir="results",  # 图输出根目录：results/refined/ 与 results/raw/
           resolution=0.3,  # SDF 栅格分辨率（米）
           safety_margin=0.3,  # 安全裕度（米）= danger radius
           base_alpha=1.0,  # 主对比中 Poisson-A* 的代表 α
           alpha_sweep=[0.25, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0], repeats=10,  # 计时重复次数（另有 1 次 warmup 不计时）
           fresh_cache=False,  # True = 清空 Poisson 磁盘缓存（重测首解耗时）
           dpi=600,  # 出图 DPI
           octile_h=True,  # octile 启发式（v7.1）：False = 退回 h≡0 Dijkstra 原口径
           octile_preh=True,  # True = h 按 goal 预计算查表（_dijkstra_jit_preh）；False = 逐次现算
           octile_ablation=True,  # True = 追加一轮 octile_h=False 的 4 引擎消融跑批，
           # 供 D 图画 octile on/off 配对点（计算量约 +40%）
           )

CACHE_DIR = HERE / ".uep_cache"
ENGINES = ["A* (alpha=0)", "Poisson-A* (alpha=1)", "FMM", "UPP"]  # driver 与 viz 共用
BASE_STRATEGIES = [(ENGINES[0], "plain"), (ENGINES[1], "poisson"), (ENGINES[2], "fmm"), (ENGINES[3], "upp")]
# D 图口径：各引擎"搜索段"stage 名（y 轴 = 这些 stage 的耗时和）
SEARCH_STAGES = {"plain": ["GridSearch_plain"], "poisson": ["GridSearch_poisson"], "fmm": ["GridSearch_fmm"],
                 "upp": ["UPP_preprocess", "UPP_search"]}


def _search_stage_ms(timer_report, engine):
    """从 timer_report 提取搜索段耗时（ms）；无记录（如 scipy 支路缺 stage）返回 None。"""
    total = 0.0
    found = False
    for st in SEARCH_STAGES.get(engine, []):
        rec = timer_report.get(st)
        if rec:
            total += rec["total_ms"]
            found = True
    return total if found else None


# ═══ driver 层 ═══

def load_maps(maps_dir):
    maps = []
    for p in sorted(Path(maps_dir).glob("*_map.json")):
        d = json.load(open(p, encoding="utf-8"))
        maps.append((p.stem, d, tuple(d["points"]["start"]["coordinates"]), tuple(d["points"]["end"]["coordinates"])))
    if not maps:
        raise SystemExit(f"no maps in {maps_dir}")
    return maps


def chain_metrics(planner, S, E, wps, margin):
    chain = [tuple(S)] + [tuple(p) for p in wps] + [tuple(E)]
    step = planner.sdf.res * 0.5
    xs, ys = [], []
    length = 0.0
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        length += d
        n = max(2, int(d / step))
        for t in np.linspace(0, 1, n + 1):
            xs.append(a[0] + t * (b[0] - a[0]))
            ys.append(a[1] + t * (b[1] - a[1]))
    sd = planner.sdf.sd_batch(np.array(xs), np.array(ys))
    turn = 0.0
    for i in range(1, len(chain) - 1):
        v1 = (chain[i][0] - chain[i - 1][0], chain[i][1] - chain[i - 1][1])
        v2 = (chain[i + 1][0] - chain[i][0], chain[i + 1][1] - chain[i][1])
        d1, d2 = math.hypot(*v1), math.hypot(*v2)
        if d1 > 1e-9 and d2 > 1e-9:
            c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
            turn += math.degrees(math.acos(c))
    euclid = math.hypot(E[0] - S[0], E[1] - S[1])
    return {"length_m": round(length, 3), "path_ratio": round(length / euclid, 4) if euclid > 1e-9 else None,
            "min_clearance_m": round(float(sd.min()), 3), "avg_clearance_m": round(float(sd.mean()), 3),
            "clearance_std_m": round(float(sd.std()), 3), "danger_violations": int((sd < margin).sum()),
            "turning_angle_deg": round(turn, 2), "n_points": len(chain)}


def run_strategy(planner, strategy_name, engine, S0, ends, margin, repeats):
    out = {}
    for i, E in enumerate(ends):
        planner.reset_run_state()
        r = planner.plan(S0, E, engine=engine)  # warmup：一次性成本不计入
        times, search_times, wps, mets = [], [], None, None
        for _ in range(repeats):
            planner.reset_run_state()
            try:
                r = planner.plan(S0, E, engine=engine)
            except Exception as ex:
                print(f"    [warn] {strategy_name} E{i + 1} rep failed: {ex}", flush=True)
                continue
            if r["metrics"]["success"]:
                wps = r["waypoints"]
                mets = r["metrics"]
                tr = r["timer_report"]
                tk = f"Total_plan_{engine}"
                if tk in tr:
                    times.append(tr[tk]["total_ms"])
                sms = _search_stage_ms(tr, engine)
                if sms is not None:
                    search_times.append(sms)
        rec = {"success": wps is not None}
        if wps is not None:
            rec.update(chain_metrics(planner, S0, E, wps, margin))
            rec["waypoints"] = [[round(x, 3), round(y, 3)] for x, y in wps]
            if mets:
                # 搜索代价 KPI：确定性量，取最后一次成功运行的值即可
                rec["expanded_nodes"] = mets.get("expanded_nodes")
                rec["search_pops"] = mets.get("search_pops")
            t = np.array(times)
            rec["time_ms"] = {"mean": round(float(t.mean()), 1), "std": round(float(t.std()), 1),
                              "min": round(float(t.min()), 1), "max": round(float(t.max()), 1), "n": int(len(t))}
            if search_times:
                st = np.array(search_times)
                rec["search_ms"] = {"mean": round(float(st.mean()), 2), "std": round(float(st.std()), 2),
                                    "n": int(len(st))}
        out[i] = rec
        print(f"    E{i + 1}: success={rec['success']} PR={rec.get('path_ratio', '-')} "
              f"t={rec.get('time_ms', {}).get('mean', '-')}ms", flush=True)
    return out


def run_raw(ide, octile_h=None, only_base=False, suffix=""):
    """跑纯搜索层口径：规划 + 落盘，返回结果 JSON 路径。
    octile_h=None 用 ide 配置；only_base=True 只跑 4 个基础引擎（octile 消融用）。"""
    octile = ide["octile_h"] if octile_h is None else octile_h
    maps = load_maps(HERE / ide["maps_dir"])
    S0 = maps[0][2]
    ends = [m[3] for m in maps]
    print(f"\n{'=' * 60}\n[RAW{suffix}] maps={len(maps)} S0={S0} octile_h={octile}")
    for i, E in enumerate(ends):
        print(f"  E{i + 1} = {E}")
    
    cfg = UConfig(sdf_resolution=ide["resolution"], safety_margin=ide["safety_margin"], alpha=ide["base_alpha"],
                  use_cache=True, cache_dir=str(CACHE_DIR), verbose=False, octile_h=octile,
                  octile_preh=bool(ide.get("octile_preh", False)))
    t0 = time.perf_counter()
    planner = UnifiedPlanner(maps[0][1], cfg)
    sdf_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    _ = planner.poisson
    poisson_ms = (time.perf_counter() - t0) * 1000
    
    blob = {"config": {"resolution": ide["resolution"], "safety_margin": ide["safety_margin"],
                       "base_alpha": ide["base_alpha"], "alpha_sweep": ide["alpha_sweep"], "repeats": ide["repeats"],
                       "octile_h": bool(octile), "octile_preh": bool(ide.get("octile_preh", False)),
                       "planner_version": PLANNER_VERSION}, "start": list(S0), "ends": [list(e) for e in ends],
            "preprocessing": {"sdf_build_ms": round(sdf_ms, 1), "poisson_solve_ms": round(poisson_ms, 1),
                              "note": "one-time per map excluded from query time not amortized"}, "strategies": {}}
    for name, engine in BASE_STRATEGIES:
        print(f"[run] {name} (cfg.alpha={planner.cfg.alpha:g})", flush=True)
        blob["strategies"][name] = run_strategy(planner, name, engine, S0, ends, ide["safety_margin"], ide["repeats"])
    if not only_base:
        for a in ide["alpha_sweep"]:
            name = f"Poisson-A* sweep alpha={a:g}"
            print(f"[run] {name}", flush=True)
            planner.cfg.alpha = a
            blob["strategies"][name] = run_strategy(planner, name, "poisson", S0, ends, ide["safety_margin"],
                                                    ide["repeats"])
    res_path = HERE / f"multi_dest_results_raw{suffix}.json"
    json.dump(blob, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[RAW{suffix} DONE] -> {res_path}")
    return res_path


# ═══ viz 层 ═══

def render_figures(res_path, maps_dir, figs_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPoly
    from matplotlib.ticker import (FixedLocator, FormatStrFormatter, FuncFormatter, LogLocator, NullFormatter,
                                   NullLocator)
    
    RES = json.load(open(res_path, encoding="utf-8"))
    CFG = RES["config"]
    FIGS = Path(figs_dir)
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
         "mathtext.fontset": "stix", "axes.unicode_minus": False, "axes.titlesize": 12, "axes.labelsize": 11,
         "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"})
    
    S0, ENDS = tuple(RES["start"]), [tuple(e) for e in RES["ends"]]
    ND = len(ENDS)
    # 口径标签：raw 与 refined 是两套机制效果，每张图标题显式区分
    DLAB = [f"E{i + 1}" for i in range(ND)]
    GEO = json.load(open(sorted(Path(maps_dir).glob("*_map.json"))[0], encoding="utf-8"))
    OBST, W, H = GEO["obstacles"]["items"], GEO["map"]["width"], GEO["map"]["height"]
    ECOL = {ENGINES[0]: "#D55E00", ENGINES[1]: "#0072B2", ENGINES[2]: "#009E73", ENGINES[3]: "#CC79A7"}
    ELAB = {e: e.replace("alpha", "α") for e in ENGINES}  # 图例显示名（JSON key 保持英文）
    DCOL = [plt.cm.tab10(i) for i in range(ND)]
    DCOL_SOFT = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    g = lambda s, i: RES["strategies"][s].get(str(i))
    
    def turn_rates(chain, ds=0.3):
        """均匀弧长重采样 Δs 后计算转角指标，返回 (mean_rate °/m, rms_rate °/m, p90 |Δθ| °)。
        单位长度公平比较：与路径长短、航点密度无关。"""
        L = 0.0
        segs = []
        for a, b in zip(chain[:-1], chain[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            segs.append(d)
            L += d
        if L < 1e-9 or len(chain) < 3:
            return np.nan, np.nan, np.nan
        P = []
        acc = 0.0
        k = 0
        for t in np.arange(0, L, ds):
            while k < len(segs) - 1 and acc + segs[k] < t:
                acc += segs[k]
                k += 1
            u = (t - acc) / (segs[k] if segs[k] > 1e-9 else 1.0)
            a, b = chain[k], chain[k + 1]
            P.append((a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])))
        P.append(chain[-1])
        P = np.array(P)
        d = np.diff(P, axis=0)
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        dt = np.abs(np.diff(ang))
        dt = np.minimum(dt, 360 - dt)
        return float(dt.mean() / ds), float(np.sqrt((dt ** 2).mean()) / ds), float(np.percentile(dt, 90))
    
    XLAB = [f"E{i + 1}" for i in range(ND)]
    EUCLID = [math.hypot(ENDS[i][0] - S0[0], ENDS[i][1] - S0[1]) for i in range(ND)]
    
    def draw_map(ax):
        ax.add_patch(plt.Rectangle((0, 0), W, H, fc="white", ec="black", lw=1.2, zorder=0))
        for o in OBST:
            if o["shape_type"] == "circle":
                ax.add_patch(Circle(tuple(o["center"]), o["radius"], fc="#BFBFBF", ec="#404040", lw=0.8, zorder=1))
            elif o["shape_type"] == "polygon":
                ax.add_patch(MplPoly(o["vertices"], closed=True, fc="#BFBFBF", ec="#404040", lw=0.8, zorder=1))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
    
    def footnote(fig, y=0.004):
        p = RES.get("preprocessing", {})
        n_total = n_succ = danger_total = 0
        for sdata in RES["strategies"].values():
            for i in range(ND):
                r = sdata.get(str(i))
                if not r:
                    continue
                n_total += 1
                if r.get("success"):
                    n_succ += 1
                    danger_total += r.get("danger_violations", 0)
        arb_note = ""
        fig.text(0.01, y + 0.016, f"Note: one-time preprocessing — SDF {p.get('sdf_build_ms', '-')} ms, "
                                  f"Poisson field {p.get('poisson_solve_ms', '-')} ms (cached, shared, not amortized). "
                                  f"Query times: mean of {CFG['repeats']} runs (1 warmup excluded). "
                                  f"UPP includes per-call S-field rebuild (faithful v5 baseline).", fontsize=8,
                 color="#444444")
        fig.text(0.01, y, f"Danger radius = safety margin = {CFG['safety_margin']} m. "
                          f"Success rate {n_succ}/{n_total} total danger violations = {danger_total}.{arb_note}",
                 fontsize=8, color="#444444")
    
    def plot_chain(ax, rec, i, color, lw=1.7, ms=3.2):
        pts = [S0] + [tuple(p) for p in rec["waypoints"]] + [ENDS[i]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=color, lw=lw, alpha=0.85, zorder=4)
        ax.plot(xs[1:-1], ys[1:-1], "o", color=color, ms=ms, zorder=5)
        return pts
    
    # ── A ──
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for i, ax in enumerate(axes.flat):
        draw_map(ax)
        ax.plot(*S0, "o", ms=9, mfc="#00B050", mec="black", mew=1.1, zorder=10)
        for eng in ENGINES:
            r = g(eng, i)
            if not r or not r.get("success"): continue
            pts = [S0] + [tuple(p) for p in r["waypoints"]] + [ENDS[i]]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=ECOL[eng], lw=2.6, alpha=0.9, zorder=4)
        ax.plot(*ENDS[i], "*", ms=16, mfc="#FFD966", mec="black", mew=1.1, zorder=11)
        ax.set_title(f"E{i + 1}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    h = [plt.Line2D([], [], color=ECOL[e], lw=3.0, label=ELAB[e]) for e in ENGINES]
    h.append(
        plt.Line2D([], [], color="none", marker="*", ms=15, mfc="#FFD966", mec="black", mew=1.1, label="Destination"))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=11.5, frameon=True, fancybox=False, edgecolor="black",
               framealpha=1.0, bbox_to_anchor=(0.5, 0.008))
    fig.suptitle("Multi-Destination Planning on One Map — Unified Start F1, All Planners per Destination", fontsize=15)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    fig.savefig(FIGS / "A_trajectories.png", dpi=dpi)
    plt.close(fig)
    
    # ── A2 ──
    ALIST = [("0 (standard A*)", ENGINES[0])] + [(f"{a:g}", f"Poisson-A* sweep alpha={a:g}") for a in
                                                 CFG["alpha_sweep"]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for k, (title, sname) in enumerate(ALIST):
        ax = axes.flat[k]
        draw_map(ax)
        ax.plot(*S0, "o", ms=8, mfc="#00B050", mec="black", mew=1.0, zorder=10)
        prs = []
        for i in range(ND):
            r = g(sname, i)
            if not r or not r.get("success"):
                continue
            plot_chain(ax, r, i, DCOL[i], lw=1.4, ms=2.6)
            ax.plot(*ENDS[i], "*", ms=10, mfc=DCOL[i], mec="black", mew=0.8, zorder=9)
            prs.append(r["path_ratio"])
        if title == "0 (standard A*)":
            ax.set_title(r"$\alpha = 0$ (standard A*)", fontsize=12)
        else:
            ax.set_title(rf"$\alpha = {title}$", fontsize=12)
        if prs:
            ax.text(0.02, 0.02, f"mean PR = {np.mean(prs):.3f}", transform=ax.transAxes, fontsize=9, color="#333333",
                    zorder=12)
        ax.set_xticks([])
        ax.set_yticks([])
    h = [plt.Line2D([], [], color=DCOL[i], lw=1.6, label=DLAB[i]) for i in range(ND)]
    fig.legend(handles=h, loc="upper center", ncol=ND, fontsize=11, frameon=True, fancybox=False, edgecolor="black",
               framealpha=1.0, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(r"Poisson-A* $\alpha$-Sweep: from Standard A* ($\alpha=0$) toward Voronoi-like Detour "
                 r"($\alpha\to\infty$)", fontsize=15)
    fig.tight_layout(rect=[0, 0.01, 1, 0.925])
    fig.savefig(FIGS / "A2_poisson_alpha_sweep.png", dpi=dpi)
    plt.close(fig)
    
    # ── B ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cells = [("path_ratio", "Path Ratio", "bar"), ("time_ms", "Query Time (ms, log)", "bar_err"),
             ("clearance_min", "Clearance (m)", "combo"), ("n_points", "Waypoints (n)", "bar"),
             ("turn_mean", "Mean Turning Rate (°/m)", "turn"), ("turn_rms", "RMS Turning Rate (°/m)", "turn")]
    x = np.arange(ND)
    wdt = 0.8 / len(ENGINES)
    for ax, (key, label, kind) in zip(axes.flat, cells):
        for j, eng in enumerate(ENGINES):
            vals, errs, mins = [], [], []
            for i in range(ND):
                r = g(eng, i)
                if not r or not r.get("success"):
                    vals.append(np.nan)
                    errs.append(0)
                    mins.append(np.nan)
                    continue
                if kind == "bar_err":
                    vals.append(r["time_ms"]["mean"])
                    errs.append(r["time_ms"]["std"])
                elif kind == "combo":
                    vals.append(r["avg_clearance_m"])
                    mins.append(r["min_clearance_m"])
                elif kind == "turn":
                    ch = [S0] + [tuple(p) for p in r["waypoints"]] + [ENDS[i]]
                    tm, tr, _ = turn_rates(ch)
                    vals.append(tm if key == "turn_mean" else tr)
                else:
                    v = r.get(key)
                    vals.append(v if v is not None else np.nan)
                    errs.append(0)
            pos = x + j * wdt - 0.4 + wdt / 2
            ax.bar(pos, vals, wdt, yerr=errs if kind == "bar_err" else None, capsize=3 if kind == "bar_err" else 0,
                   color=ECOL[eng], alpha=0.9, edgecolor="black", lw=0.4)
            if kind == "combo":
                ax.plot(pos, mins, "D", color="black", ms=4.5, zorder=5, label="Min clearance" if j == 0 else None)
        if kind == "combo":
            ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
        if key == "time_ms":
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=80))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(FixedLocator([10, 50, 100, 500]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(x)
        ax.set_xticklabels(XLAB)
        ax.set_title(label, fontsize=11.5)
        ax.grid(axis="y", alpha=0.3, which="both")
    h = [plt.Rectangle((0, 0), 1, 1, fc=ECOL[e], ec="black", lw=0.4) for e in ENGINES]
    fig.legend(handles=h, labels=[ELAB[e] for e in ENGINES], loc="lower center", ncol=4, fontsize=11, frameon=True,
               fancybox=False, edgecolor="black", framealpha=1.0, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Per-Destination Metrics by Planner (6 destinations, unified start)", fontsize=15)
    fig.tight_layout(rect=[0, 0.065, 1, 0.96])
    fig.savefig(FIGS / "B_statistics.png", dpi=dpi)
    plt.close(fig)
    
    # ── C ──
    AKPI = [("path_ratio", "Path Ratio"), ("clearance_combo", "Clearance (m)"), ("time_ms", "Query Time (ms)"),
            ("n_points", "Waypoints (n)"), ("turn_mean", "Mean Turning Rate (°/m)"),
            ("turn_rms", "RMS Turning Rate (°/m)")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    xa = np.arange(len(ALIST))
    wdt = 0.8 / ND
    for ax, (key, label) in zip(axes.flat, AKPI):
        means = []
        for a_idx, (title, sname) in enumerate(ALIST):
            col, mins = [], []
            for i in range(ND):
                r = g(sname, i)
                if not (r and r.get("success")):
                    col.append(np.nan)
                    mins.append(np.nan)
                    continue
                if key == "clearance_combo":
                    col.append(r["avg_clearance_m"])
                    mins.append(r["min_clearance_m"])
                elif key == "time_ms":
                    col.append(r["time_ms"]["mean"])
                elif key in ("turn_mean", "turn_rms"):
                    ch = [S0] + [tuple(p) for p in r["waypoints"]] + [ENDS[i]]
                    tm, tr, _ = turn_rates(ch)
                    col.append(tm if key == "turn_mean" else tr)
                else:
                    col.append(r.get(key, np.nan))
            col = np.array(col, float)
            with np.errstate(invalid="ignore"):
                means.append(np.nanmean(col))
            pos = xa[a_idx] + np.arange(ND) * wdt - 0.4 + wdt / 2
            ax.bar(pos, col, wdt, color=DCOL_SOFT, alpha=0.95, edgecolor="#666666", lw=0.4)
            if key == "clearance_combo":
                ax.plot(pos, mins, "D", color="black", ms=3.5, zorder=5, label="Min clearance" if a_idx == 0 else None)
        ax.plot(xa, means, "k-", lw=1.6, marker="o", ms=6, mfc="white", mec="black", zorder=6)
        if key == "clearance_combo":
            ax.legend(loc="upper left", fontsize=9, frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
        ax.set_xticks(xa)
        ax.set_xticklabels([t.replace(" (standard A*)", "\n(A*)") for t, _ in ALIST], fontsize=9.5)
        ax.set_title(label, fontsize=12)
        ax.grid(axis="y", alpha=0.3, which="both")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=DCOL_SOFT[i], ec="#666666", lw=0.3) for i in range(ND)]
    handles.append(plt.Line2D([], [], color="black", lw=1.6, marker="o", ms=6, mfc="white", mec="black"))
    fig.legend(handles=handles, labels=DLAB + ["Mean over destinations"], loc="lower center", ncol=ND + 1, fontsize=10,
               frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Effect of Poisson Shaping Strength α on Path-Quality KPIs (α=0 is standard A*)", fontsize=15)
    fig.tight_layout(rect=[0, 0.065, 1, 0.96])
    fig.savefig(FIGS / "C_alpha_kpi.png", dpi=dpi)
    plt.close(fig)
    
    # ── D：octile 消融（1×2）——OFF 斜纹柱为衬底、ON 实心柱同轴前景（略窄），
    #        实心柱顶以上的斜纹露出部分即"启发式削减的扩展量/时间" ──
    comp_path = res_path.with_name(res_path.stem + "_octileoff.json")
    COMP = None
    if comp_path.exists():
        try:
            COMP = json.load(open(comp_path, encoding="utf-8"))
        except Exception:
            COMP = None
    ABL_ENG = [ENGINES[0], ENGINES[1]]  # 仅 A*/Poisson 受 octile 影响（numba 搜索支路）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    if COMP is not None:
        _c0, _c1 = RES.get("config", {}), COMP.get("config", {})
        _bad = [k for k in ("resolution", "safety_margin", "base_alpha") if k in _c0 and k in _c1 and _c0[k] != _c1[k]]
        if _bad:
            fig.text(0.5, 0.905, "CONFIG MISMATCH with _octileoff.json: " + ", ".join(
                f"{k} {_c0[k]} vs {_c1[k]}" for k in _bad) + " — delete both multi_dest_results_raw*.json and re-run.",
                     color="red", fontsize=11, ha="center", fontweight="bold")
    x = np.arange(ND)
    wdt = 0.30
    
    def _fmt_num(v):
        return f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}"
    
    def _fmt_ms(v):
        return f"{v:g}"
    
    for ax, (key, label) in zip(axes, [("expanded_nodes", "Expanded Nodes (settled)"),
                                       ("search_ms", "Search-Stage Time (ms)")]):
        fmt = _fmt_num if key == "expanded_nodes" else _fmt_ms
        for j, eng in enumerate(ABL_ENG):
            vals, vals_off = [], []
            for i in range(ND):
                r = g(eng, i)
                v = None
                if r and r.get("success"):
                    v = (r.get("search_ms") or {}).get("mean") if key == "search_ms" else r.get(key)
                vals.append(v if v is not None else np.nan)
                vo = None
                if COMP is not None:
                    rc = COMP["strategies"].get(eng, {}).get(str(i))
                    if rc and rc.get("success"):
                        vo = (rc.get("search_ms") or {}).get("mean") if key == "search_ms" else rc.get(key)
                vals_off.append(vo if vo is not None else np.nan)
            pos = x + (j - 0.5) * (wdt + 0.08)
            ax.bar(pos, vals_off, wdt, facecolor="white", edgecolor=ECOL[eng], lw=1.4, hatch="///", zorder=2)
            ax.bar(pos, vals, wdt * 0.86, color=ECOL[eng], alpha=0.95, edgecolor="black", lw=0.5, zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(x)
        ax.set_xticklabels(XLAB)
        ax.set_xlabel("Destination")
        ax.set_ylabel(label + ", log", fontsize=11)
        ax.set_title(f"octile ablation — {label}", fontsize=12)
        ax.grid(axis="y", alpha=0.3, which="major")
    h = [plt.Rectangle((0, 0), 1, 1, fc=ECOL[e], ec="black", lw=0.4) for e in ENGINES[:2]]
    h.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#666666", lw=1.4, hatch="///"))
    fig.legend(handles=h, labels=[ELAB[e] for e in ENGINES[:2]] + ["octile OFF (h≡0 Dijkstra)"], loc="lower center",
               ncol=3, fontsize=11, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Search Effort vs. Time — octile ablation per Destination", fontsize=15)
    fig.text(0.01, 0.082,
             "Settled = |visited| at goal pop (deterministic). Hatched = octile OFF baseline; solid bar (slightly "
             "narrower, in front) = octile ON — the exposed hatched cap is the reduction from the octile heuristic.",
             fontsize=9, color="#333333")
    fig.text(0.01, 0.052, "octile OFF applies only to A*/Poisson (numba search branch); UPP/FMM unaffected.",
             fontsize=9, color="#333333")
    fig.tight_layout(rect=[0, 0.125, 1, 0.93])
    fig.savefig(FIGS / "D_search_effort.png", dpi=dpi)
    plt.close(fig)
    
    # ── E：单位路径长度的时间成本（Discussion 补充分析，断轴显示）──
    # 断轴边界由数据自适应：小段 [0, max_small×1.18]，大段 [min_big×0.88, max_big×1.12]，
    # 避免数值贴边/越界（大小值分界取 1.0 ms/m，与数据分布间隙一致）。
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5), sharex="col",
                            gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.06})
    x = np.arange(ND)
    wdt = 0.8 / len(ENGINES)
    d = 0.012
    rh = 1.0 / 3.2
    for j, (key, label) in enumerate([("time_ms", "Query Time per Path Length (ms/m)"),
                                      ("search_ms", "Search-Stage Time per Path Length (ms/m)")]):
        axt, axb = axs[0, j], axs[1, j]
        vals = []
        for k, eng in enumerate(ENGINES):
            ev = []
            for i in range(ND):
                r = g(eng, i)
                v = None
                if r and r.get("success"):
                    if key == "time_ms":
                        v = r["time_ms"]["mean"] / r["length_m"]
                    else:
                        sm = (r.get("search_ms") or {}).get("mean")
                        v = sm / r["length_m"] if sm is not None else None
                ev.append(v if v is not None else np.nan)
            vals.append(ev)
            pos = x + k * wdt - 0.4 + wdt / 2
            axt.bar(pos, ev, wdt, color=ECOL[eng], alpha=0.9, edgecolor="black", lw=0.4)
            axb.bar(pos, ev, wdt, color=ECOL[eng], alpha=0.9, edgecolor="black", lw=0.4)
        flat = [v for ev in vals for v in ev if not np.isnan(v)]
        small = [v for v in flat if v <= 1.0]
        big = [v for v in flat if v > 1.0]
        lo_top = (max(small) * 1.18) if small else 1.0
        hi_bot = (min(big) * 0.88) if big else 4.0
        hi_top = (max(big) * 1.12) if big else 6.0
        axt.set_ylim(hi_bot, hi_top)
        axb.set_ylim(0, lo_top)
        for k, ev in enumerate(vals):
            pos_k = x + k * wdt - 0.4 + wdt / 2
            for p, v in zip(pos_k, ev):
                if np.isnan(v):
                    continue
                if v <= 1.0:
                    axb.text(p, min(v * 1.12, lo_top * 0.96), f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                             color="#333333")
                else:
                    axt.text(p, min(v * 1.03, hi_top * 0.95), f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                             color="#333333")
        axt.set_title(label, fontsize=12)
        axb.set_xlabel("Destination")
        axb.set_ylabel("ms/m", fontsize=11)
        axb.set_xticks(x)
        axb.set_xticklabels(XLAB)
        axt.tick_params(axis="x", which="both", length=0)
        plt.setp(axt.get_xticklabels(), visible=False)
        axt.spines["bottom"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axt.plot((-d, +d), (-d * rh * 2.2, +d * rh * 2.2), transform=axt.transAxes, color="k", clip_on=False, lw=1.2)
        axt.plot((1 - d, 1 + d), (-d * rh * 2.2, +d * rh * 2.2), transform=axt.transAxes, color="k", clip_on=False,
                 lw=1.2)
        axb.plot((-d, +d), (1 - d / rh / 2.2, 1 + d / rh / 2.2), transform=axb.transAxes, color="k", clip_on=False,
                 lw=1.2)
        axb.plot((1 - d, 1 + d), (1 - d / rh / 2.2, 1 + d / rh / 2.2), transform=axb.transAxes, color="k",
                 clip_on=False, lw=1.2)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=ECOL[e], ec="black", lw=0.4) for e in ENGINES],
               labels=[ELAB[e] for e in ENGINES], loc="lower center", ncol=4, fontsize=11, frameon=True, fancybox=False,
               edgecolor="black", framealpha=1.0, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Time Cost per Unit Path Length — cross-destination fair comparison", fontsize=15)
    fig.text(0.01, 0.155, "Caveat: search cost is not a linear function of distance (it includes terrain-dependent "
                          "exploration); ms/m is an amortized figure of merit, not a physical law.", fontsize=9,
             color="#333333")
    fig.text(0.01, 0.118, "Broken y-axis: lower segment holds values ≤ 1 ms/m (A*/Poisson/FMM), upper segment holds "
                          "UPP (≈3-10 ms/m); segment bounds auto-fit to data. UPP includes per-call S-field rebuild "
                          "(UPP_preprocess); other engines: numba search only.", fontsize=9, color="#333333")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.24, hspace=0.06)
    fig.savefig(FIGS / "E_time_per_meter.png", dpi=dpi)
    plt.close(fig)
    print(f"  [FIGS] -> {FIGS}")


# ═══ 一键入口 ═══

def main(ide=IDE):
    if ide["fresh_cache"]:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print("[fresh-cache] removed .uep_cache")
    res = run_raw(ide)
    if ide["octile_ablation"]:
        run_raw(ide, octile_h=False, only_base=True, suffix="_octileoff")
    render_figures(res, HERE / ide["maps_dir"], HERE / ide["out_dir"], ide["dpi"])
    print(f"\n[ALL DONE] figures in {HERE / ide['out_dir']}/")


if __name__ == "__main__":
    main()
