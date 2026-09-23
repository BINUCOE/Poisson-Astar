#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════
# unified_planner_raw.py —— 纯搜索层（raw）路径规划器  v7.1+raw
#
# 开源/论文配套版本：pipeline 仅含最优搜索层，不含任何精化逻辑：
#   SDF 几何基底（UnifiedSDFMap）→ Poisson 场解算与边权整形（PoissonField）
#   → octile 启发式栅格搜索（GridSearch：plain / poisson / fmm；UPPSearch）
#   → 抽稀折线 + 弦安全校验 / 格点回填（_safe_grid_waypoints）
#
# 血缘说明：本文件由内部开发版 unified_planner.py（v7.1，raw/refined 双口径）
# 剔除精化层（Refiner / HomotopyArbiter / 同伦仲裁 / Fallback L1-L3 / 脊线居中
# / use_refiner 开关）而来，搜索层源码逐字一致，规划结果逐位相同。
#
# 有记录的框架级取舍（对所有引擎一致）:
#   F-a. 共享禁切角 8 连通图（UPP 官方允许切角，safety_margin 已滤除
#        贴障格点，差异可忽略）；F-b. S/E 由 nearest_free_cell 吸附；
#   F-c. float64（官方 float32）、S 场滑窗求和（官方 fftconvolve）。
#
# v7.1 搜索代价 KPI 口径：_dijkstra_jit / _upp_astar_faithful 返回
# (settled, pops)——settled = visited 集合大小（文献标准口径，确定性量），
# pops = 出堆总次数（含 stale，debug 用）；经 plan() metrics 导出
# expanded_nodes / search_pops；FMM 记可达自由格数；scipy 参照支路记 None。
# ═══════════════════════════════════════════════════════════════

PLANNER_VERSION = "v7.1+raw"
import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _cs_dijkstra
from scipy.sparse.linalg import spsolve

# v4: numba JIT 可选加速（环境有则启用，无则静默回退）
_HAS_NUMBA = False
try:
    import numba
    from numba import njit, prange
    
    _HAS_NUMBA = True
except Exception:
    pass


def _maybe_jit(*args, **kwargs):
    """可选 JIT：有 numba 则 @njit，否则无操作装饰器。"""
    
    def decorator(func):
        if _HAS_NUMBA:
            return njit(*args, **kwargs)(func)
        return func
    
    return decorator


# ═══════════════════════════════════════════════════════════════
# v7: 统一 numba 搜索核 —— 忠诚复现 UPP（safeplan 官方实现）+ 同基底 Dijkstra
# ═══════════════════════════════════════════════════════════════

# ── 旧环境兼容垫片（Python 3.7 / numpy < 1.20）───────────────────
try:
    from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view
except ImportError:  # numpy < 1.20（py3.7 常配 1.17–1.19）：as_strided 等价实现
    from numpy.lib.stride_tricks import as_strided as _as_strided
    
    
    def _sliding_window_view(x, window_shape):
        """2D 方形滑窗视图；窗口元素与官方版完全一致（下游经 np.where
        物化为连续数组，视图步长不影响归约结果与位序）。"""
        Wy, Wx = window_shape
        ny, nx = x.shape
        shape = (ny - Wy + 1, nx - Wx + 1, Wy, Wx)
        strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
        return _as_strided(x, shape=shape, strides=strides)

try:
    _default_rng = np.random.default_rng
except AttributeError:  # numpy < 1.17：RandomState 提供相同的 normal/uniform 接口
    def _default_rng(seed=None):
        return np.random.RandomState(seed)

_upp_astar = None
_dijkstra_jit = None
if _HAS_NUMBA:
    @njit(cache=True)
    def _heap3_push(hf, hg, hn, hs, f, g, n):
        """三元组 (f, g, node) 最小堆入堆；与 heapq 元组比较序一致。"""
        i = hs
        hs += 1
        while i > 0:
            p = (i - 1) // 2
            if (hf[p] < f or (hf[p] == f and (hg[p] < g or (hg[p] == g and hn[p] <= n)))):
                break
            hf[i] = hf[p];
            hg[i] = hg[p];
            hn[i] = hn[p]
            i = p
        hf[i] = f;
        hg[i] = g;
        hn[i] = n
        return hs
    
    
    @njit(cache=True)
    def _heap3_pop(hf, hg, hn, hs):
        """出堆最小元，返回 (f, g, n, 新大小)。顺序与 heapq.heappop 一致。"""
        f = hf[0];
        g = hg[0];
        n = hn[0]
        hs -= 1
        if hs > 0:
            hf[0] = hf[hs];
            hg[0] = hg[hs];
            hn[0] = hn[hs]
            i = 0
            while True:
                l = 2 * i + 1
                r = l + 1
                m = i
                if l < hs and (
                        hf[l] < hf[m] or (hf[l] == hf[m] and (hg[l] < hg[m] or (hg[l] == hg[m] and hn[l] < hn[m])))):
                    m = l
                if r < hs and (
                        hf[r] < hf[m] or (hf[r] == hf[m] and (hg[r] < hg[m] or (hg[r] == hg[m] and hn[r] < hn[m])))):
                    m = r
                if m == i:
                    break
                tf = hf[i];
                hf[i] = hf[m];
                hf[m] = tf
                tg = hg[i];
                hg[i] = hg[m];
                hg[m] = tg
                tn = hn[i];
                hn[i] = hn[m];
                hn[m] = tn
                i = m
        return f, g, n, hs
    
    
    @njit(cache=True)
    def _upp_astar_faithful(indptr, indices, unit_w, safety, nx, sn, gn, alpha0, beta0, eps, goal_tol, patience, bmin,
                            bmax, bdec, brec, amin, amax, adec, arec, tol_ang, turn_target, turn_win, adaptive_beta,
                            adaptive_alpha):
        """UPP 忠诚 numba 核（复现 safeplan/algos/upp.py 搜索循环，逐条对应）：

        - 全程格点单位：unit_w = 步长（1 / √2），safety 无量纲；
        - 堆元组 (f, g, node)，f 在入堆时以当时的 α/β 固化（官方语义：堆内旧 f 不刷新）；
        - 每次出堆（含 stale 重复元与 goal 元）先按到目标 L2 距离进度更新 β、
          按运动方向夹角滑窗更新 α，随后 goal 判定，最后 visited 去重——
          与官方 plan() 的顺序严格一致；
        - β：Δd < −goal_tol 乘 brec（封顶 bmax）；Δd > goal_tol 乘 bdec（封底 bmin）；
          停滞计数 ≥ patience 乘 bdec；α：滑窗累计 (转角−turn_target)，
          超 ±tol_ang 时乘 arec/adec 并复位窗口（起点节点无父节点，累计 −turn_target）。
        """
        N = indptr.shape[0] - 1
        g = np.full(N, np.inf)
        came = np.full(N, -1, np.int64)
        visited = np.zeros(N, np.uint8)
        cap = 2 * indices.shape[0] + 8  # 每次松弛至多一次入堆
        hf = np.empty(cap, np.float64)
        hg = np.empty(cap, np.float64)
        hn = np.empty(cap, np.int64)
        hs = 0
        pops = 0
        settled = 0
        gy = gn // nx
        gx = gn % nx
        alpha = alpha0  # 官方初值 alphaBase=0.5，由滑窗机制在搜索中调整
        beta = beta0
        stalled = 0
        prev_dist = -1.0
        turn_sum = 0.0
        turn_iter = 0
        two_pi = 2.0 * math.pi
        g[sn] = 0.0
        # 起点入堆：h0 = α·l1+(1−α)·l∞+β·S（官方 heuristic(self.start, self.goal)）
        cy = sn // nx
        cx = sn % nx
        dy = gy - cy
        if dy < 0:
            dy = -dy
        dx = gx - cx
        if dx < 0:
            dx = -dx
        h0 = alpha0 * (dy + dx) + (1.0 - alpha0) * (dy if dy > dx else dx) + beta * safety[sn]
        hs = _heap3_push(hf, hg, hn, hs, h0, 0.0, sn)
        found = False
        
        while hs > 0:
            _, g_u, u, hs = _heap3_pop(hf, hg, hn, hs)
            pops += 1
            uy = u // nx
            ux = u % nx
            # ── β 自适应（每次出堆，含 stale；官方顺序第一步）──
            dy = gy - uy
            dx = gx - ux
            cur_dist = math.sqrt(dy * dy + dx * dx)
            if prev_dist < 0.0:
                prev_dist = cur_dist
            delta = cur_dist - prev_dist
            if adaptive_beta:
                if delta < -goal_tol:
                    stalled = 0
                    nb = beta * brec
                    if nb < bmax:
                        beta = nb
                elif delta > goal_tol:
                    stalled = 0
                    nb = beta * bdec
                    if nb > bmin:
                        beta = nb
                else:
                    stalled += 1
                    if stalled >= patience:
                        nb = beta * bdec
                        if nb > bmin:
                            beta = nb
                        stalled = 0
            prev_dist = cur_dist
            # ── α 自适应（每次出堆；官方顺序第二步）──
            if adaptive_alpha:
                turn_angle = 0.0
                if came[u] >= 0:
                    pv = came[u]
                    mvx = ux - pv % nx
                    mvy = uy - pv // nx
                    gvx = gx - ux
                    gvy = gy - uy
                    m_norm = math.sqrt(mvx * mvx + mvy * mvy)
                    gnrm = math.sqrt(gvx * gvx + gvy * gvy)
                    if m_norm > 1e-6 and gnrm > 1e-6:
                        raw = (math.atan2(gvy, gvx) - math.atan2(mvy, mvx) + math.pi) % two_pi - math.pi
                        if raw < 0.0:
                            raw = -raw
                        turn_angle = raw
                turn_sum += turn_angle - turn_target
                turn_iter += 1
                if turn_iter >= turn_win:
                    if turn_sum > tol_ang:
                        na = alpha * arec
                        if na < amax:
                            alpha = na
                    elif turn_sum < -tol_ang:
                        na = alpha * adec
                        if na > amin:
                            alpha = na
                    turn_sum = 0.0
                    turn_iter = 0
            # ── goal 判定（官方第三步：先于 visited 检查）──
            if u == gn:
                found = True
                break
            if visited[u]:
                continue
            visited[u] = 1
            settled += 1
            gu = g[u]
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if visited[v]:
                    continue
                ng = gu + unit_w[k]
                if ng < g[v]:
                    came[v] = u
                    g[v] = ng
                    vy = v // nx
                    vx = v % nx
                    ddy = gy - vy
                    if ddy < 0:
                        ddy = -ddy
                    ddx = gx - vx
                    if ddx < 0:
                        ddx = -ddx
                    hh = alpha * (ddy + ddx) + (1.0 - alpha) * (ddy if ddy > ddx else ddx) + beta * safety[v]
                    hs = _heap3_push(hf, hg, hn, hs, ng + hh, ng, v)
        return found, came, settled, pops
    
    
    @njit(cache=True)
    def _dijkstra_jit(indptr, indices, weights, sn, gn, nx, res, use_h):
        """同基底 A*/Dijkstra：plain / poisson 引擎的 numba 计时参照，
        与 UPP 共用同一 (f, g, node) 堆与 CSR 图，消除 C/Python 语言偏差。

        use_h=True 时启用 octile 可采纳启发式 h = (max + (sqrt(2)-1)*min) * res：
        poisson 边权倍率 1+α·e^(−βu) ≥ 1 ⇒ 相邻格 h 差 ≤ 边权（启发式一致），
        首次弹出 goal 即最优，路径代价与 Dijkstra 逐位一致；实测 settled
        节点数降为 1/1.4~1/4.7（6 图），plain/poisson 同核同启发式。
        返回 (found, came, settled, pops)：settled = visited 集合大小
        （确定性的搜索代价 KPI），pops = 出堆总次数（含 stale，debug 用）。"""
        N = indptr.shape[0] - 1
        g = np.full(N, np.inf)
        came = np.full(N, -1, np.int64)
        visited = np.zeros(N, np.uint8)
        cap = 2 * indices.shape[0] + 8
        hf = np.empty(cap, np.float64)
        hg = np.empty(cap, np.float64)
        hn = np.empty(cap, np.int64)
        hs = 0
        pops = 0
        settled = 0
        gx = gn % nx
        gy = gn // nx
        sdy = abs(sn // nx - gy)
        sdx = abs(sn % nx - gx)
        h0 = sdy if sdy > sdx else sdx
        h0 += 0.4142135623730951 * (sdy if sdy < sdx else sdx)
        h0 *= res
        g[sn] = 0.0
        hs = _heap3_push(hf, hg, hn, hs, h0, 0.0, sn)
        found = False
        while hs > 0:
            _, g_u, u, hs = _heap3_pop(hf, hg, hn, hs)
            pops += 1
            if u == gn:
                found = True
                break
            if visited[u]:
                continue
            visited[u] = 1
            settled += 1
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if visited[v]:
                    continue
                ng = g_u + weights[k]
                if ng < g[v]:
                    g[v] = ng
                    came[v] = u
                    if use_h:
                        dy = abs(v // nx - gy)
                        dx = abs(v % nx - gx)
                        hh = dy if dy > dx else dx
                        hh += 0.4142135623730951 * (dy if dy < dx else dx)
                        hs = _heap3_push(hf, hg, hn, hs, ng + hh * res, ng, v)
                    else:
                        hs = _heap3_push(hf, hg, hn, hs, ng, ng, v)
        return found, came, settled, pops
    
    
    @njit(cache=True)
    def _dijkstra_jit_preh(indptr, indices, weights, sn, gn, nx, res, H):
        """octile 启发式预计算版：h 数组 H 在搜索前一次性向量化构建（按 goal 缓存于
        sdf._h_cache，同一次批跑内 50 次重复仅建一次），松弛时查表而非逐次现算。
        与 _dijkstra_jit(use_h=True) 语义逐位一致——h 对一次查询是静态的，
        只依赖 (v, gn)。实验用途：量化"逐次重算 h"的实现层开销。"""
        N = indptr.shape[0] - 1
        g = np.full(N, np.inf)
        came = np.full(N, -1, np.int64)
        visited = np.zeros(N, np.uint8)
        cap = 2 * indices.shape[0] + 8
        hf = np.empty(cap, np.float64)
        hg = np.empty(cap, np.float64)
        hn = np.empty(cap, np.int64)
        hs = 0
        pops = 0
        settled = 0
        g[sn] = 0.0
        hs = _heap3_push(hf, hg, hn, hs, H[sn], 0.0, sn)
        found = False
        while hs > 0:
            _, g_u, u, hs = _heap3_pop(hf, hg, hn, hs)
            pops += 1
            if u == gn:
                found = True
                break
            if visited[u]:
                continue
            visited[u] = 1
            settled += 1
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if visited[v]:
                    continue
                ng = g_u + weights[k]
                if ng < g[v]:
                    g[v] = ng
                    came[v] = u
                    hs = _heap3_push(hf, hg, hn, hs, ng + H[v], ng, v)
        return found, came, settled, pops


def _jit_warmup():
    """微图触发 numba 编译（cache=True 存盘后后续进程近零开销）。"""
    if not _HAS_NUMBA:
        return
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1], dtype=np.int64)
    unit_w = np.array([1.0], dtype=np.float64)
    safety = np.zeros(2, dtype=np.float64)
    _upp_astar_faithful(indptr, indices, unit_w, safety, 2, 0, 1, 0.5, 0.5, 0.01, 0.1, 20, 0.1, 2.0, 0.97, 1.05, 0.05,
                        0.95, 0.97, 1.05, math.pi, math.radians(15.0), 10, True, True)
    _dijkstra_jit(indptr, indices, unit_w, 0, 1, 2, 1.0, False)
    _dijkstra_jit(indptr, indices, unit_w, 0, 1, 2, 1.0, True)
    _dijkstra_jit_preh(indptr, indices, unit_w, 0, 1, 2, 1.0, np.array([0.5, 0.2]))


def _save_with_retry(fig, path, dpi, bbox_inches=None, tries=3):
    """fig.savefig 的容错封装：FUSE/portal 类挂载上，冷进程首次写大 PNG
    偶发 ENOENT（句柄已打开、写入中途失败），短延迟重试即可成功。"""
    import time as _time
    for i in range(tries):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            return
        except OSError:
            if i == tries - 1:
                raise
            _time.sleep(0.3 * (i + 1))


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
from matplotlib.gridspec import GridSpec


def _setup_cjk_font():
    """有中文字体则启用（消除缺字形警告），没有则静默保持默认。"""
    try:
        from matplotlib import font_manager
        prefs = (
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC",
            "Arial Unicode MS")
        have = {f.name for f in font_manager.fontManager.ttflist}
        for p in prefs:
            if p in have:
                plt.rcParams["font.sans-serif"] = [p, "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_setup_cjk_font()


# ═══════════════════════════════════════════════════════════════
# Config（刻意收敛到少量旋钮）
# ═══════════════════════════════════════════════════════════════

@dataclass
class UPPParams:
    """UPP 官方超参数（来源：github.com/jatinarora30/safeplan，
    safeplan/algos/upp.py 默认值 + README run1.json）。逐条照搬，不做调优；
    adaptive_beta / adaptive_alpha 整体关闭可复现论文常值参数口径
    （α=0.5, β=0.5 为对比实验值；几何感知初始化仍保留，见 upp.py plan()）。
    """
    
    # Algorithm 1 / 初始化基值
    alpha_base: float = 0.5
    beta_base: float = 10.0  # β_init = clip(βBase·ρ·σ/(μ+ε), βMin, βMax)
    radius_base: float = 1.0  # R_init = clip(round(RBase·(μ+σ)), RMin, RMax)
    epsilon: float = 0.01  # 安全场核 1/(‖Δ‖∞+ε) 与初始化分母共用
    # β 自适应（Algorithm 2 / 式 5）
    beta_min: float = 0.1
    beta_max: float = 2.0
    beta_decay: float = 0.97
    beta_recovery: float = 1.05
    beta_patience: int = 20  # K_β：停滞计数阈值
    goal_tol: float = 0.1  # τ_goal：进度/停滞判定容差（格）
    # α 自适应（Algorithm 2 / 式 6）
    alpha_min: float = 0.05
    alpha_max: float = 0.95
    alpha_decay: float = 0.97
    alpha_recovery: float = 1.05
    tol_angular_deg: float = 180.0  # τ_ang：转角累计触发阈值
    turn_target_deg: float = 15.0  # θ_tar：目标转角（度）
    turn_window: int = 10  # K_α：转角滑窗长度
    # R 初始化边界
    radius_min: int = 1
    radius_max: int = 5
    # 消融开关（论文对比实验为常值参数；机制检验用）
    adaptive_beta: bool = True
    adaptive_alpha: bool = True


@dataclass
class UConfig:
    sdf_resolution: float = 0.3  # SDF/搜索栅格分辨率（米）
    safety_margin: float = 0.1  # 安全裕度（米）
    alpha: float = 0.25  # 泊松整形强度（0 = 退化为标准 A*）
    beta: Optional[float] = None  # None → 2/u_mean 自适应
    fmm_c0: float = 3.0  # FMM 速度常数：speed = sd/(sd+c0)
    decimate_step: int = 8  # 栅格路径抽稀步长（格）
    use_cache: bool = True  # Poisson 场磁盘缓存（地图未变时跳过求解）
    cache_dir: str = "./.uep_cache"  # 缓存目录
    verbose: bool = True
    # v4 新增可视化配置
    viz_dpi: int = 110  # 可视化 DPI（原 160，降低可大幅提速）
    skip_viz: bool = False  # 仅计算不画图（扫参用）
    use_numba: bool = True  # 是否启用 numba JIT（搜索核加速）
    # octile 启发式开关（numba A* 路径）：True → plain/poisson 搜索用 octile 距离作一致
    # 可采纳启发式，settled 节点数降为 1/1.4~1/4.7（6 图实测），路径代价与 Dijkstra
    # 逐位一致，KPI 完全可比；False 退回 h≡0 Dijkstra（原口径）。scipy 参照支路不受影响。
    octile_h: bool = True
    # h 预计算开关（实验）：True → _dijkstra_jit_preh，h 数组按 goal 建一次查表；
    # False → 原实现，每次松弛现算 h。仅 octile_h=True 且 numba 支路时生效。
    octile_preh: bool = False
    upp: UPPParams = field(default_factory=UPPParams)  # v7: UPP 官方超参数（safeplan 默认）


# ═══════════════════════════════════════════════════════════════
# 时间分析器
# ═══════════════════════════════════════════════════════════════

class StageTimer:
    """分阶段计时器，用于精细化时间成本分析。"""
    
    def __init__(self):
        self.records: Dict[str, List[float]] = {}
        self._stack: List[Tuple[str, float]] = []
    
    def start(self, name: str):
        self._stack.append((name, time.perf_counter()))
    
    def end(self, name: str = None):
        if not self._stack:
            return
        if name is None:
            name, t0 = self._stack.pop()
        else:
            idx = None
            for i, (n, _) in enumerate(reversed(self._stack)):
                if n == name:
                    idx = len(self._stack) - 1 - i
                    break
            if idx is None:
                return
            name, t0 = self._stack.pop(idx)
        dt = (time.perf_counter() - t0) * 1000
        self.records.setdefault(name, []).append(dt)
        return dt
    
    def report(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in self.records.items():
            arr = np.array(v)
            out[k] = {"count": len(v), "total_ms": round(float(arr.sum()), 2), "mean_ms": round(float(arr.mean()), 2),
                      "min_ms": round(float(arr.min()), 2), "max_ms": round(float(arr.max()), 2),
                      "std_ms": round(float(arr.std()), 2), }
        return out
    
    def reset(self):
        self.records.clear()
        self._stack.clear()


# ═══════════════════════════════════════════════════════════════
# Layer 0: UnifiedSDFMap —— 唯一几何内核
# ═══════════════════════════════════════════════════════════════

class UnifiedSDFMap:
    """PIL 栅格化（rect/circle/polygon 统一）+ EDT 连续 SDF + 双线性插值。"""
    
    def __init__(self, map_data: Dict, resolution: float = 0.3, timer: Optional[StageTimer] = None):
        self._raw = map_data
        self.width = float(map_data["map"]["width"])
        self.height = float(map_data["map"]["height"])
        self.obstacles = map_data["obstacles"]["items"]
        self.res = float(resolution)
        self.nx = max(2, int(self.width / self.res) + 1)
        self.ny = max(2, int(self.height / self.res) + 1)
        self.timer = timer or StageTimer()
        # v3: 缓存与复用钩子（由 UnifiedPlanner 按 cfg 开启）
        self.map_hash = hashlib.sha1(
            json.dumps({"map": map_data.get("map"), "obstacles": map_data.get("obstacles")}, sort_keys=True,
                       default=str).encode("utf-8")).hexdigest()[:16]  # v6-P3: geometry-only hash, points excluded
        self.use_cache = False
        self.cache_dir = "./.uep_cache"
        self._graph_cache: Dict[float, Tuple] = {}  # safety_margin -> 栅格图结构
        self._bg_cache: Dict[str, np.ndarray] = {}  # cmap -> 预烘焙背景 RGBA
        # v4: 边权乘子缓存（按 mode 缓存，避免 GridSearch 每次重建时重复计算 np.exp）
        self._mult_cache: Dict[str, np.ndarray] = {}
        # v8: CSR / jit-CSR 缓存提升到 sdf 级（原为 GridSearch 实例级，每次 plan() 都
        # 随 GridSearch 销毁，同图重复规划时每次查询重复装配 ~200 万边 CSR，约 100–200ms）。
        # 键不变：(margin, mode, alpha, beta, fmm_c0)，跨引擎/跨终点/重复运行共享。
        self._csr_cache: Dict[Tuple, csr_matrix] = {}
        self._jit_csr_cache: Dict[Tuple, tuple] = {}
        self._h_cache: Dict[Tuple, np.ndarray] = {}  # octile_preh：按 goal 缓存预计算 h 场
        
        self.timer.start("SDF_rasterize")
        self._mask = self._rasterize()
        self.timer.end("SDF_rasterize")
        
        self.timer.start("SDF_edt")
        free = (self._mask > 128).astype(np.uint8)
        self.grid = (distance_transform_edt(free) - distance_transform_edt(1 - free)) * self.res
        gy, gx = np.gradient(self.grid)
        self._gx = gx / self.res
        self._gy = gy / self.res
        self.timer.end("SDF_edt")
    
    def _rasterize(self) -> np.ndarray:
        img = Image.new('L', (self.nx, self.ny), 255)
        draw = ImageDraw.Draw(img)
        s = 1.0 / self.res
        for obs in self.obstacles:
            st = obs.get("shape_type", "rect")
            if st == "rect":
                x1, y1 = obs.get("x1", 0), obs.get("y1", 0)
                x2, y2 = obs.get("x2", x1), obs.get("y2", y1)
                draw.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], fill=0)
            elif st == "circle":
                cx, cy, r = obs["center"][0] * s, obs["center"][1] * s, obs["radius"] * s
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
            elif st == "polygon":
                vs = [(v[0] * s, v[1] * s) for v in obs["vertices"]]
                if len(vs) >= 3:
                    draw.polygon(vs, fill=0)
        return np.array(img, dtype=np.uint8)
    
    def sd(self, x: float, y: float) -> float:
        """世界坐标 SDF（米），越界返回负值。"""
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return min(x, y, self.width - x, self.height - y)
        ix, iy = x / self.res, y / self.res
        x0, y0 = int(ix), int(iy)
        if x0 >= self.nx - 1 or y0 >= self.ny - 1:
            return float(self.grid[min(y0, self.ny - 1), min(x0, self.nx - 1)])
        sx, sy = ix - x0, iy - y0
        g = self.grid
        return float(
            g[y0, x0] * (1 - sx) * (1 - sy) + g[y0, x0 + 1] * sx * (1 - sy) + g[y0 + 1, x0] * (1 - sx) * sy + g[
                y0 + 1, x0 + 1] * sx * sy)
    
    def sd_batch(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        ix, iy = xs / self.res, ys / self.res
        x0 = np.floor(ix).astype(int)
        y0 = np.floor(iy).astype(int)
        sx, sy = ix - x0, iy - y0
        out = (x0 < 0) | (y0 < 0) | (x0 >= self.nx - 1) | (y0 >= self.ny - 1)
        x0 = np.clip(x0, 0, self.nx - 2)
        y0 = np.clip(y0, 0, self.ny - 2)
        g = self.grid
        v = (g[y0, x0] * (1 - sx) * (1 - sy) + g[y0, x0 + 1] * sx * (1 - sy) + g[y0 + 1, x0] * (1 - sx) * sy + g[
            y0 + 1, x0 + 1] * sx * sy)
        return np.where(out, -np.inf, v)
    
    def segment_min_sd(self, a, b, step: Optional[float] = None) -> Tuple[float, float]:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        step = step or self.res * 0.5
        n = max(2, int(d / step))
        ts = np.linspace(0, 1, n + 1)
        v = self.sd_batch(a[0] + ts * (b[0] - a[0]), a[1] + ts * (b[1] - a[1]))
        return float(np.min(v)), float(np.mean(v))
    
    def chain_min_sd(self, chain) -> Tuple[float, float]:
        mn, sm, cnt = float('inf'), 0.0, 0
        for i in range(len(chain) - 1):
            m, a = self.segment_min_sd(chain[i], chain[i + 1])
            mn = min(mn, m)
            sm += a
            cnt += 1
        return mn, (sm / cnt if cnt else 0.0)
    
    def snap_to_ridge(self, p, iters=8, step=0.8):
        x, y = float(p[0]), float(p[1])
        for _ in range(iters):
            ix, iy = x / self.res, y / self.res
            x0, y0 = int(ix), int(iy)
            if x0 < 0 or y0 < 0 or x0 >= self.nx - 1 or y0 >= self.ny - 1:
                break
            sx, sy = ix - x0, iy - y0
            gx = (self._gx[y0, x0] * (1 - sx) * (1 - sy) + self._gx[y0, x0 + 1] * sx * (1 - sy) + self._gx[
                y0 + 1, x0] * (1 - sx) * sy + self._gx[y0 + 1, x0 + 1] * sx * sy)
            gy = (self._gy[y0, x0] * (1 - sx) * (1 - sy) + self._gy[y0, x0 + 1] * sx * (1 - sy) + self._gy[
                y0 + 1, x0] * (1 - sx) * sy + self._gy[y0 + 1, x0 + 1] * sx * sy)
            n = math.hypot(gx, gy)
            if n < 1e-6:
                break
            nx_, ny_ = x + step * gx / n, y + step * gy / n
            if not (0 <= nx_ <= self.width and 0 <= ny_ <= self.height):
                break
            x, y = nx_, ny_
        return round(x, 3), round(y, 3)
    
    def world_to_cell(self, p) -> Tuple[int, int]:
        return int(np.clip(p[1] / self.res, 0, self.ny - 1)), int(np.clip(p[0] / self.res, 0, self.nx - 1))
    
    def cell_to_world(self, c) -> Tuple[float, float]:
        return (c[1] + 0.5) * self.res, (c[0] + 0.5) * self.res
    
    def nearest_free_cell(self, c, free: np.ndarray, max_r=80) -> Optional[Tuple[int, int]]:
        y, x = c
        if free[y, x]:
            return c
        for r in range(1, max_r):
            for dy in range(-r, r + 1):
                for dx in (-r, r):
                    for yy, xx in ((y + dy, x + dx), (y + dx, x + dy)):
                        if 0 <= yy < self.ny and 0 <= xx < self.nx and free[yy, xx]:
                            return yy, xx
        return None


# ═══════════════════════════════════════════════════════════════
# Layer 1a: PoissonField —— 只做边权整形，不做下降场
# ═══════════════════════════════════════════════════════════════

class PoissonField:
    def __init__(self, sdf: UnifiedSDFMap, timer: Optional[StageTimer] = None):
        self.timer = timer or StageTimer()
        self.timer.start("Poisson_solve")
        t0 = time.perf_counter()
        if not self._try_load_cache(sdf):
            self._solve(sdf)
            self._save_cache(sdf)
        self.solve_ms = (time.perf_counter() - t0) * 1000
        self.timer.end("Poisson_solve")
    
    def _solve(self, sdf: UnifiedSDFMap):
        """向量化装配（矩阵与原逐格循环实现逐元相同）+ spsolve 直接法。

        注：原实现对角 deg 恒为 4（deg+=1 在边界判断之外），即地图边界也按
        Dirichlet 墙处理，此处保持一致。CG 迭代法在该规模实测慢于直接法，弃用。
        """
        occ = sdf.grid <= 0
        ny, nx = sdf.ny, sdf.nx
        free = ~occ
        nf = int(free.sum())
        idx = -np.ones((ny, nx), dtype=np.int64)
        idx[free] = np.arange(nf)
        ar = np.arange(nf)
        rows_l, cols_l, val_l = [ar], [ar], [np.full(nf, 4.0)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sh = np.full((ny, nx), -1, np.int64)
            if dr == -1:
                sh[1:, :] = idx[:-1, :]
            elif dr == 1:
                sh[:-1, :] = idx[1:, :]
            elif dc == -1:
                sh[:, 1:] = idx[:, :-1]
            else:
                sh[:, :-1] = idx[:, 1:]
            d = sh[free]
            m = d >= 0
            rows_l.append(ar[m])
            cols_l.append(d[m])
            val_l.append(np.full(int(m.sum()), -1.0))
        A = csr_matrix((np.concatenate(val_l), (np.concatenate(rows_l), np.concatenate(cols_l))), shape=(nf, nf))
        self.u = np.zeros((ny, nx))
        self.u[free] = spsolve(A, np.ones(nf))
        self.u_mean = float(self.u[self.u > 0].mean()) if np.any(self.u > 0) else 1.0
    
    def _cache_path(self, sdf: UnifiedSDFMap) -> Optional[Path]:
        if not getattr(sdf, 'use_cache', False) or not getattr(sdf, 'map_hash', None):
            return None
        return Path(getattr(sdf, 'cache_dir', './.uep_cache')) / f"poisson_{sdf.map_hash}_{sdf.res:.4g}.npz"
    
    def _try_load_cache(self, sdf: UnifiedSDFMap) -> bool:
        p = self._cache_path(sdf)
        if p is None or not p.exists():
            return False
        try:
            z = np.load(p)
            u = z['u']
            if u.shape != (sdf.ny, sdf.nx):
                return False
            self.u = u
            self.u_mean = float(z['u_mean'])
            return True
        except Exception:
            return False
    
    def _save_cache(self, sdf: UnifiedSDFMap):
        p = self._cache_path(sdf)
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, u=self.u, u_mean=self.u_mean)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Layer 1b/2: 统一栅格搜索（整形 A* / FMM 导航函数）
# ═══════════════════════════════════════════════════════════════

_DIRS = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)), (0, 1, 1.0), (1, 1, math.sqrt(2)), (1, 0, 1.0),
         (1, -1, math.sqrt(2)), (0, -1, 1.0)]


class GridSearch:
    def __init__(self, sdf: UnifiedSDFMap, cfg: UConfig, poisson: Optional[PoissonField] = None,
                 timer: Optional[StageTimer] = None):
        self.sdf = sdf
        self.cfg = cfg
        self.free = sdf.grid > cfg.safety_margin
        self.u = poisson.u if poisson else None
        self.beta = cfg.beta if cfg.beta is not None else (
            2.0 / poisson.u_mean if poisson and poisson.u_mean > 0 else 1.0)
        self.timer = timer or StageTimer()
        # v8: CSR / jit-CSR 缓存托管在 sdf（跨查询共享，见 UnifiedSDFMap.__init__）；
        # 实例属性保留为引用，兼容可能的外部访问；键 = (margin, mode, alpha, beta, fmm_c0)
        self._csr_cache = sdf._csr_cache
        self._jit_csr_cache = sdf._jit_csr_cache  # v7: numba 同基底 Dijkstra 的 CSR 数组缓存
    
    def _mult_grid(self, mode) -> np.ndarray:
        """边权乘子网格（从 sdf._mult_cache 获取，避免 GridSearch 每次重建时重复计算）。"""
        # v6-P1: 键必须含 alpha/beta/fmm_c0 —— 乘子网格随这些参数变化，
        # 原仅以 mode 作键会在同进程 alpha 扫参时复用首个 alpha 的场（静默错误）
        key = (mode, round(float(self.cfg.alpha), 6), round(float(self.beta), 6), round(float(self.cfg.fmm_c0), 6))
        m = self.sdf._mult_cache.get(key)
        if m is None:
            if mode == 'poisson':
                g = 1.0 + self.cfg.alpha * np.exp(-self.beta * self.u)
            elif mode == 'fmm':
                sd = self.sdf.grid
                g = (sd + self.cfg.fmm_c0) / np.maximum(sd, 1e-6)
            else:
                g = np.ones((self.sdf.ny, self.sdf.nx))
            m = np.ascontiguousarray(g).ravel()
            self.sdf._mult_cache[key] = m
        return m
    
    def _graph_struct(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """8 连通禁切角栅格图（COO 三元组），按 safety_margin 缓存于 sdf，每图只建一次。"""
        key = float(self.cfg.safety_margin)
        cache = self.sdf._graph_cache
        if key in cache:
            return cache[key]
        ny, nx, res = self.sdf.ny, self.sdf.nx, self.sdf.res
        pad = np.pad(self.free, 1, constant_values=False)
        ys, xs = np.nonzero(self.free)
        pys, pxs = ys + 1, xs + 1
        src = ys.astype(np.int64) * nx + xs
        rows_l, cols_l, step_l = [], [], []
        for dy, dx, w in _DIRS:
            cy, cx = pys + dy, pxs + dx
            ok = pad[cy, cx]
            if dy != 0 and dx != 0:  # 禁切角：两正交邻格均须自由
                ok = ok & pad[pys + dy, pxs] & pad[pys, pxs + dx]
            rows_l.append(src[ok])
            cols_l.append((cy[ok] - 1).astype(np.int64) * nx + (cx[ok] - 1))
            step_l.append(np.full(int(ok.sum()), w * res))
        struct = (np.concatenate(rows_l), np.concatenate(cols_l), np.concatenate(step_l), ny * nx)
        cache[key] = struct
        return struct
    
    def search(self, start_c, goal_c, mode='poisson', dijkstra=False):
        """scipy Dijkstra（参照实现，h≡0 全树）或 numba 同基底 A*（use_numba 且非 FMM 时，
        cfg.octile_h=True 启用 octile 启发式；h 一致 ⇒ 代价与 scipy Dijkstra 逐位一致）。
        副作用：self.last_expanded / self.last_pops 记录本次搜索代价（settled/pops），
        scipy 参照支路除 FMM 全树口径（可达自由格数）外均记 None。"""
        self.last_expanded: Optional[int] = None
        self.last_pops: Optional[int] = None
        self.timer.start(f"GridSearch_{mode}")
        ny, nx = self.sdf.ny, self.sdf.nx
        s = self.sdf.nearest_free_cell(start_c, self.free)
        g_ = self.sdf.nearest_free_cell(goal_c, self.free)
        if s is None or g_ is None:
            self.timer.end(f"GridSearch_{mode}")
            return None, None
        rows, cols, steps, N = self._graph_struct()
        mult = self._mult_grid(mode)
        # v4: CSR 缓存键
        cache_key = (
            float(self.cfg.safety_margin), mode, float(self.cfg.alpha), float(self.beta), float(self.cfg.fmm_c0))
        # v8: 缓存托管在 sdf（跨查询共享）；GridSearch 实例只持有引用
        G = self.sdf._csr_cache.get(cache_key)
        if G is None:
            G = csr_matrix((steps * mult[cols], (rows, cols)), shape=(N, N))
            self.sdf._csr_cache[cache_key] = G
        sn, gn = s[0] * nx + s[1], g_[0] * nx + g_[1]
        if (_HAS_NUMBA and self.cfg.use_numba and _dijkstra_jit is not None and not dijkstra):
            j = self.sdf._jit_csr_cache.get(cache_key)
            if j is None:
                j = (np.ascontiguousarray(G.indptr, dtype=np.int64), np.ascontiguousarray(G.indices, dtype=np.int64),
                     np.ascontiguousarray(G.data, dtype=np.float64))
                self.sdf._jit_csr_cache[cache_key] = j
            if self.cfg.octile_h and self.cfg.octile_preh:
                hk = ("hfield", int(gn))
                H = self.sdf._h_cache.get(hk)
                if H is None:
                    ny_, nx_ = self.sdf.ny, self.sdf.nx
                    gy_, gx_ = gn // nx_, gn % nx_
                    dy_ = np.abs(np.arange(ny_)[:, None] - gy_)
                    dx_ = np.abs(np.arange(nx_)[None, :] - gx_)
                    H = np.ascontiguousarray(
                        (np.maximum(dy_, dx_) + 0.4142135623730951 * np.minimum(dy_, dx_)).ravel() * self.sdf.res,
                        dtype=np.float64)
                    self.sdf._h_cache[hk] = H
                found, came, settled, pops = _dijkstra_jit_preh(j[0], j[1], j[2], sn, gn, nx, self.sdf.res, H)
            else:
                found, came, settled, pops = _dijkstra_jit(j[0], j[1], j[2], sn, gn, nx, self.sdf.res,
                                                           bool(self.cfg.octile_h))
            self.last_expanded, self.last_pops = int(settled), int(pops)
            if not found:
                self.timer.end(f"GridSearch_{mode}")
                return None, None
            cells = [g_]
            node = gn
            while node != sn:
                node = int(came[node])
                if node < 0:
                    self.timer.end(f"GridSearch_{mode}")
                    return None, None
                cells.append((node // nx, node % nx))
            cells.reverse()
            self.timer.end(f"GridSearch_{mode}")
            return cells, None
        dist, pred = _cs_dijkstra(G, directed=True, indices=sn, return_predecessors=True)
        T = dist.reshape(ny, nx) if dijkstra else None
        if dijkstra:
            # FMM 全树口径：导航函数的搜索代价 = 可达自由格数（场方法的算法本性）
            self.last_expanded = int(np.isfinite(dist).sum())
        if not np.isfinite(dist[gn]):
            self.timer.end(f"GridSearch_{mode}")
            return None, T
        cells = [g_]
        node = gn
        while node != sn:
            p = int(pred[node])
            if p < 0:
                self.timer.end(f"GridSearch_{mode}")
                return None, T
            node = p
            cells.append((node // nx, node % nx))
        cells.reverse()
        self.timer.end(f"GridSearch_{mode}")
        return cells, T
    
    def fmm_descent(self, T: np.ndarray, start_c, goal_c, max_steps=20000):
        self.timer.start("FMM_descent")
        ny, nx = T.shape
        res = self.sdf.res
        finite = np.isfinite(T)
        T_safe = np.where(finite, T, np.max(T[finite]) * 2.0 if finite.any() else 1e9)
        with np.errstate(invalid='ignore'):
            gy, gx = np.gradient(T_safe)
        
        def interp(F, y, x):
            y = min(max(y, 0), ny - 1.001)
            x = min(max(x, 0), nx - 1.001)
            y0, x0 = int(y), int(x)
            sy, sx = y - y0, x - x0
            return (F[y0, x0] * (1 - sx) * (1 - sy) + F[y0, x0 + 1] * sx * (1 - sy) + F[y0 + 1, x0] * (1 - sx) * sy + F[
                y0 + 1, x0 + 1] * sx * sy)
        
        y, x = float(start_c[0]), float(start_c[1])
        pts = [self.sdf.cell_to_world((start_c[0], start_c[1]))]
        stall = 0
        last_T = T[start_c]
        for _ in range(max_steps):
            if math.hypot(y - goal_c[0], x - goal_c[1]) < 1.0:
                break
            dy = interp(gy, y, x)
            dx = interp(gx, y, x)
            n = math.hypot(dy, dx)
            if n < 1e-9 or not np.isfinite(n):
                stall += 1
                if stall > 50:
                    break
                y += (goal_c[0] - y) * 0.01
                x += (goal_c[1] - x) * 0.01
                continue
            y -= 0.6 * dy / n
            x -= 0.6 * dx / n
            t = interp(T_safe, y, x)
            stall = stall + 1 if t > last_T - 1e-7 else 0
            last_T = min(t, last_T)
            if stall > 50:
                break
            pts.append(self.sdf.cell_to_world((y, x)))
        pts.append(self.sdf.cell_to_world((goal_c[0], goal_c[1])))
        self.timer.end("FMM_descent")
        return pts


# ═══════════════════════════════════════════════════════════════
# Layer 1c: UPP (Unified Path Planner) —— 忠诚复现基线
# ═══════════════════════════════════════════════════════════════

class UPPSearch:
    """UPP 忠诚复现：A* + 自适应加权启发式 + 卷积安全场。

    复现来源：J. K. Arora et al., "Balancing Safety and Optimality in Robot
    Path Planning: Algorithm and Metric", Robotics and Autonomous Systems, 2026
    （arXiv:2505.23197），官方实现 github.com/jatinarora30/safeplan。

    逐条对应官方 upp.py 的语义：
      • h = α·l₁+(1−α)·l∞+β·S，全程格点单位（步长 1/√2，不乘 res）；
      • S 卷积核 1/(‖Δ‖∞+ε)，1≤‖Δ‖∞≤R，中心 exclude，障碍格点 S≡0；
      • β_init/R_init 几何感知（μ/σ 取自由空间 EDT，ρ 障碍密度）；
      • 每次出堆（含 stale 重复元）先更新 β（到目标 L2 距离进度）、
        再更新 α（运动方向夹角滑窗），然后 goal 判定、visited 去重；
      • f 在入堆时固化（堆内旧 f 不刷新）；堆元组 (f, g, node)。

    有记录的框架级取舍（对所有引擎一致，对 UPP 不偏不倚）：
      F-a. 搜索在框架共享的禁切角 8 连通图上进行（官方允许切角；框架
           统一禁切角且 safety_margin 已滤除贴障格点，差异可忽略）；
      F-b. S/E 由 nearest_free_cell 吸附（官方要求输入自由格点）；
      F-c. float64（官方 float32）；S 场滑窗求和（官方 fftconvolve，
           加法顺序差异 ~1e-7）；以上不改变机制，仅影响 tie 情形的
           浮点末位；
    """
    
    def __init__(self, sdf: UnifiedSDFMap, cfg: UConfig, timer: Optional[StageTimer] = None):
        self.sdf = sdf
        self.cfg = cfg
        self.params = cfg.upp
        self.timer = timer or StageTimer()
        self.ny, self.nx = sdf.ny, sdf.nx
        self.N = self.ny * self.nx
        
        self.timer.start("UPP_preprocess")
        # 公平性补记：几何初始化 + 安全场构建计入 stage 计时（与 Poisson_solve 同口径）
        
        # ── 与图结构对齐的 S/E 吸附掩膜（沿用 F1）──
        gs = GridSearch(sdf, cfg, timer=timer)
        self.free_upp = gs.free
        
        # ── Algorithm 1：几何感知初始化（官方 plan() 内计算，与 S/E 无关，
        #    故提到构造期一次；μ/σ 官方取格点单位 EDT，这里换算 /res）──
        p = self.params
        occ = (sdf.grid <= 0)
        rho = float(occ.sum()) / float(occ.size)
        free_d = sdf.grid[~occ]
        mu = float(free_d.mean()) / sdf.res
        sigma = float(free_d.std()) / sdf.res
        self.beta0 = min(max(p.beta_base * rho * sigma / (mu + p.epsilon), p.beta_min), p.beta_max)
        self.R = int(min(max(round(p.radius_base * (mu + sigma)), p.radius_min), p.radius_max))
        
        # ── CSR 邻接（与 GridSearch 共用图结构；unit_w = 步长，格点单位）──
        key = float(cfg.safety_margin)
        csr_cache = getattr(sdf, "_upp_csr_cache", None)
        if csr_cache is None:
            csr_cache = {}
            sdf._upp_csr_cache = csr_cache
        if key in csr_cache:
            self._indptr, self._indices, self._unit_w = csr_cache[key]
        else:
            rows, cols, steps, N = gs._graph_struct()
            g = csr_matrix((steps, (rows, cols)), shape=(N, N))
            self._indptr = np.ascontiguousarray(g.indptr, dtype=np.int64)
            self._indices = np.ascontiguousarray(g.indices, dtype=np.int64)
            self._unit_w = np.ascontiguousarray(g.data / sdf.res, dtype=np.float64)  # 米 → 格
            csr_cache[key] = (self._indptr, self._indices, self._unit_w)
        
        # ── S(n) 静态场（官方 fftconvolve；滑窗实现，语义一致）──
        self._safety = self._build_safety()
        self.timer.end("UPP_preprocess")
        
        self._jit = bool(_HAS_NUMBA and cfg.use_numba and _upp_astar_faithful is not None)
        if self._jit:
            _jit_warmup()
    
    def _build_safety(self) -> np.ndarray:
        """S(n)：Σ 𝟙_O(n+Δ)/(‖Δ‖∞+ε)，1≤‖Δ‖∞≤R，中心 exclude，障碍格点 S≡0。

        内部格点滑窗 + 边界逐格，求和语义与官方卷积一致（加法顺序差异
        ~1e-7，不改变机制）。v7.1：按需求不启用任何缓存（内存/磁盘），
        每次构造直接重建，预处理成本全量计入 UPP_preprocess。
        """
        sdf = self.sdf
        p = self.params
        ny, nx, R = self.ny, self.nx, self.R
        eps = p.epsilon
        obs = (sdf.grid <= 0)
        S = np.zeros((ny, nx))
        dy = np.arange(-R, R + 1).reshape(-1, 1)
        dx = np.arange(-R, R + 1).reshape(1, -1)
        d_inf = np.maximum(np.abs(dy), np.abs(dx)).astype(np.float64)
        kernel = np.where((d_inf > 0) & (d_inf <= R), 1.0 / (d_inf + eps), 0.0)
        W = 2 * R + 1
        if ny > 2 * R and nx > 2 * R:
            padded = np.pad(obs, R, constant_values=False)
            win = _sliding_window_view(padded, (W, W))
            # reshape 成 (n, W²) 沿末轴归约，保证结合顺序稳定
            s_int = (win[R:ny - R, R:nx - R] * kernel).reshape(-1, W * W).sum(axis=1)
            S[R:ny - R, R:nx - R] = s_int.reshape(ny - 2 * R, nx - 2 * R)
        interior = np.zeros((ny, nx), dtype=bool)
        if ny > 2 * R and nx > 2 * R:
            interior[R:ny - R, R:nx - R] = True
        for cy, cx in map(tuple, np.argwhere(~interior)):
            acc = 0.0
            for oy in range(-R, R + 1):
                yy = cy + oy
                if yy < 0 or yy >= ny:
                    continue
                for ox in range(-R, R + 1):
                    if oy == 0 and ox == 0:
                        continue
                    xx = cx + ox
                    if xx < 0 or xx >= nx:
                        continue
                    if obs[yy, xx]:
                        acc += 1.0 / (max(abs(oy), abs(ox)) + eps)
            S[cy, cx] = acc
        # 官方 preB：自由格且 S_sum>0 保留，其余置 0（含全部障碍格点）
        S = np.where((~obs) & (S > 0), S, 0.0)
        return np.ascontiguousarray(S.ravel(), dtype=np.float64)
    
    def search(self, start_c, goal_c) -> Optional[List[Tuple[int, int]]]:
        """UPP 搜索。返回格点坐标列表 [(y,x), ...]。"""
        self.timer.start("UPP_search")
        nx = self.nx
        s = self.sdf.nearest_free_cell(start_c, self.free_upp)
        g = self.sdf.nearest_free_cell(goal_c, self.free_upp)
        if s is None or g is None:
            self.timer.end("UPP_search")
            return None
        sn = s[0] * nx + s[1]
        gn = g[0] * nx + g[1]
        p = self.params
        deg = math.pi / 180.0
        if self._jit:
            found, came, settled, pops = _upp_astar_faithful(self._indptr, self._indices, self._unit_w, self._safety,
                                                             nx, sn, gn, p.alpha_base, self.beta0, p.epsilon,
                                                             p.goal_tol, p.beta_patience, p.beta_min, p.beta_max,
                                                             p.beta_decay, p.beta_recovery, p.alpha_min, p.alpha_max,
                                                             p.alpha_decay, p.alpha_recovery, p.tol_angular_deg * deg,
                                                             p.turn_target_deg * deg, p.turn_window, p.adaptive_beta,
                                                             p.adaptive_alpha)
        else:
            found, came, settled, pops = self._search_py(sn, gn)
        self.last_expanded, self.last_pops = int(settled), int(pops)
        if not found:
            self.timer.end("UPP_search")
            return None
        cells = []
        node = gn
        while node != sn:
            cells.append((node // nx, node % nx))
            node = int(came[node])
            if node < 0:
                self.timer.end("UPP_search")
                return None
        cells.append((sn // nx, sn % nx))
        cells.reverse()
        self.timer.end("UPP_search")
        return cells
    
    def _search_py(self, sn: int, gn: int):
        """纯 Python 忠诚镜像：与 numba 核 _upp_astar_faithful 逐条对应，
        --no-numba / 无 numba 环境时启用。两者在测试地图上断言路径一致。"""
        import heapq
        nx = self.nx
        N = self.N
        p = self.params
        deg = math.pi / 180.0
        tol_ang = p.tol_angular_deg * deg
        turn_target = p.turn_target_deg * deg
        g_score = np.full(N, np.inf)
        came = np.full(N, -1, np.int64)
        visited = np.zeros(N, dtype=np.uint8)
        safety = self._safety
        gy, gx_ = gn // nx, gn % nx
        two_pi = 2.0 * math.pi
        alpha = p.alpha_base
        beta = self.beta0
        stalled = 0
        prev_dist = None
        turn_sum = 0.0
        turn_iter = 0
        
        def h_u(n):
            dy = abs(gy - n // nx)
            dx = abs(gx_ - n % nx)
            return alpha * (dy + dx) + (1.0 - alpha) * (dy if dy > dx else dx) + beta * safety[n]
        
        g_score[sn] = 0.0
        heap = [(h_u(sn), 0.0, sn)]
        pops = 0
        settled = 0
        while heap:
            _, g_u, u = heapq.heappop(heap)
            pops += 1
            uy, ux = u // nx, u % nx
            # ── β 自适应（每次出堆，含 stale；与官方一致）──
            cur_dist = math.hypot(gy - uy, gx_ - ux)
            if prev_dist is None:
                prev_dist = cur_dist
            delta = cur_dist - prev_dist
            if p.adaptive_beta:
                if delta < -p.goal_tol:
                    stalled = 0
                    beta = min(beta * p.beta_recovery, p.beta_max)
                elif delta > p.goal_tol:
                    stalled = 0
                    beta = max(beta * p.beta_decay, p.beta_min)
                else:
                    stalled += 1
                    if stalled >= p.beta_patience:
                        beta = max(beta * p.beta_decay, p.beta_min)
                        stalled = 0
            prev_dist = cur_dist
            # ── α 自适应（每次出堆；与官方一致）──
            if p.adaptive_alpha:
                turn_angle = 0.0
                if came[u] >= 0:
                    pv = int(came[u])
                    mvx = ux - pv % nx
                    mvy = uy - pv // nx
                    gvx = gx_ - ux
                    gvy = gy - uy
                    m_norm = math.hypot(mvx, mvy)
                    g_norm = math.hypot(gvx, gvy)
                    if m_norm > 1e-6 and g_norm > 1e-6:
                        raw = (math.atan2(gvy, gvx) - math.atan2(mvy, mvx) + math.pi) % two_pi - math.pi
                        turn_angle = abs(raw)
                turn_sum += turn_angle - turn_target
                turn_iter += 1
                if turn_iter >= p.turn_window:
                    if turn_sum > tol_ang:
                        alpha = min(alpha * p.alpha_recovery, p.alpha_max)
                    elif turn_sum < -tol_ang:
                        alpha = max(alpha * p.alpha_decay, p.alpha_min)
                    turn_sum = 0.0
                    turn_iter = 0
            # ── goal 判定先于 visited 检查（官方顺序）──
            if u == gn:
                return True, came, settled, pops
            if visited[u]:
                continue
            visited[u] = 1
            settled += 1
            for k in range(self._indptr[u], self._indptr[u + 1]):
                v = int(self._indices[k])
                if visited[v]:
                    continue
                ng = g_u + self._unit_w[k]
                if ng < g_score[v]:
                    came[v] = u
                    g_score[v] = ng
                    heapq.heappush(heap, (ng + h_u(v), ng, v))
        return False, came, settled, pops


# ═══════════════════════════════════════════════════════════════
# UnifiedPlanner —— 纯搜索层主管线 + 精细化计时
# ═══════════════════════════════════════════════════════════════

def validate_path(sdf: UnifiedSDFMap, chain, margin: float) -> Tuple[bool, float, float]:
    mn, mean = sdf.chain_min_sd(chain)
    return mn >= margin, mn, mean


class UnifiedPlanner:
    def __init__(self, map_data: Dict, cfg: Optional[UConfig] = None):
        self.cfg = cfg or UConfig()
        self.timer = StageTimer()
        self.timer.start("SDF_build")
        self.sdf = UnifiedSDFMap(map_data, self.cfg.sdf_resolution, timer=self.timer)
        self.timer.end("SDF_build")
        self.sdf.use_cache = self.cfg.use_cache
        self.sdf.cache_dir = self.cfg.cache_dir
        self._poisson: Optional[PoissonField] = None
        self.rng = _default_rng(42)
        self.log: List[str] = []
        self._coarse_stats: Dict[str, Dict] = {}  # engine -> {"expanded": int|None, "pops": int|None}
    
    def reset_run_state(self, seed: int = 42):
        """同一 planner 跨引擎复用时，重置每次运行的随机源/日志/计时器。
        重播种 42 使各引擎的随机候选序列与 v2 各自新建 planner 时完全一致。"""
        self.rng = _default_rng(seed)
        self.log = []
        self.timer.reset()
        self._coarse_stats = {}
    
    def _log(self, msg):
        self.log.append(msg)
        if self.cfg.verbose:
            print(f"  [UEP] {msg}")
    
    @property
    def poisson(self) -> PoissonField:
        if self._poisson is None:
            self._poisson = PoissonField(self.sdf, timer=self.timer)
            self._log(f"Poisson 场求解 {self._poisson.solve_ms:.0f}ms, u_mean={self._poisson.u_mean:.1f}")
        return self._poisson
    
    def _safe_grid_waypoints(self, cells):
        """弦校验回填：抽稀折线逐弦 segment_min_sd 校验，
        凡 < safety_margin 的弦沿原始格点序列回填中间格（相邻格连线即原始安全栅格链），
        修复 decimate_step 直弦切角导致的负 clearance。FMM 为连续下降点路径，不经此函数。"""
        margin = self.cfg.safety_margin
        k = self.cfg.decimate_step
        idxs = list(range(0, len(cells), k))
        if idxs[-1] != len(cells) - 1:
            idxs.append(len(cells) - 1)
        pts = []
        for a_i, b_i in zip(idxs[:-1], idxs[1:]):
            a = self.sdf.cell_to_world(cells[a_i])
            if not pts or pts[-1] != a:
                pts.append(a)
            bw = self.sdf.cell_to_world(cells[b_i])
            m, _ = self.sdf.segment_min_sd(a, bw)
            if m < margin:
                for j in range(a_i + 1, b_i):
                    pts.append(self.sdf.cell_to_world(cells[j]))
        last = self.sdf.cell_to_world(cells[-1])
        if pts[-1] != last:
            pts.append(last)
        return pts
    
    def _coarse(self, S, E, engine: str) -> Optional[List[Tuple[float, float]]]:
        gs = GridSearch(self.sdf, self.cfg, self.poisson if engine == 'poisson' else None, timer=self.timer)
        sc, gc = self.sdf.world_to_cell(S), self.sdf.world_to_cell(E)
        if engine == 'fmm':
            cells, T = gs.search(gc, sc, mode='fmm', dijkstra=True)
            self._coarse_stats['fmm'] = {"expanded": gs.last_expanded, "pops": gs.last_pops}
            if cells is None:
                return None
            pts = gs.fmm_descent(T, self.sdf.nearest_free_cell(sc, gs.free), self.sdf.nearest_free_cell(gc, gs.free))
            k = self.cfg.decimate_step
            return [pts[0]] + pts[k::k] + [pts[-1]]
        if engine == 'upp':
            upp = UPPSearch(self.sdf, self.cfg, timer=self.timer)
            cells = upp.search(sc, gc)
            self._coarse_stats['upp'] = {"expanded": getattr(upp, 'last_expanded', None),
                                         "pops": getattr(upp, 'last_pops', None)}
            if cells is None:
                return None
            return self._safe_grid_waypoints(cells)
        cells, _ = gs.search(sc, gc, mode='poisson' if engine == 'poisson' else 'plain')
        self._coarse_stats[engine] = {"expanded": gs.last_expanded, "pops": gs.last_pops}
        if cells is None:
            return None
        return self._safe_grid_waypoints(cells)
    
    def _full_pipeline(self, S, E, engine: str):
        """纯搜索层管线：粗规划（搜索 + 抽稀 + 弦校验/回填）→ 交付折线。
        返回 (pts, valid, min_sd)；粗规划失败返回 (None, False, -1.0)。"""
        coarse = self._coarse(S, E, engine)
        if coarse is None:
            return None, False, -1.0
        pts = [p for p in coarse if p != tuple(S) and p != tuple(E)]
        valid, mn, _ = validate_path(self.sdf, [tuple(S)] + pts + [tuple(E)], self.cfg.safety_margin)
        return pts, valid, mn
    
    def plan(self, S, E, engine: str = 'poisson') -> Dict:
        self.timer.start(f"Total_plan_{engine}")
        t0 = time.perf_counter()
        S, E = tuple(S), tuple(E)
        sd_s, sd_e = self.sdf.sd(*S), self.sdf.sd(*E)
        if sd_s < 0 or sd_e < 0:
            raise ValueError(f"S/E 在障碍内: sd_s={sd_s:.2f}, sd_e={sd_e:.2f}")
        self._log(f"规划 {S} -> {E}, 引擎={engine}")
        
        pts, valid, mn = self._full_pipeline(S, E, engine)
        if pts is None:
            self._log("失败：S/E 不连通或无法保证安全")
            self.timer.end(f"Total_plan_{engine}")
            return {"waypoints": None, "metrics": {"success": False, "valid": False}, "engine_used": engine,
                    "log": self.log, "timer_report": self.timer.report()}
        
        chain = [S] + pts + [E]
        valid, min_sd, mean_sd = validate_path(self.sdf, chain, self.cfg.safety_margin)
        length = sum(
            math.hypot(chain[i + 1][0] - chain[i][0], chain[i + 1][1] - chain[i][1]) for i in range(len(chain) - 1))
        curvature = 0.0
        for i in range(1, len(chain) - 1):
            v1 = (chain[i][0] - chain[i - 1][0], chain[i][1] - chain[i - 1][1])
            v2 = (chain[i + 1][0] - chain[i][0], chain[i + 1][1] - chain[i][1])
            d1, d2 = math.hypot(*v1), math.hypot(*v2)
            if d1 > 1e-9 and d2 > 1e-9:
                c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
                curvature += math.acos(c)
        
        self.timer.end(f"Total_plan_{engine}")
        # 搜索代价 KPI：主引擎粗搜索的扩展量（settled=visited 集合大小，确定性量；
        # pops=出堆总次数含 stale，debug 用）。scipy 参照支路（无 numba）为 None；
        # FMM 为可达自由格数。
        cs = self._coarse_stats.get(engine, {})
        metrics = {"success": True, "valid": valid, "length_m": round(length, 3),
                   "path_ratio": round(length / (math.hypot(E[0] - S[0], E[1] - S[1]) or 1.0), 4),
                   "min_clearance_m": round(min_sd, 3), "mean_clearance_m": round(mean_sd, 3),
                   "curvature_rad": round(curvature, 3), "n_points": len(chain), "expanded_nodes": cs.get("expanded"),
                   "search_pops": cs.get("pops"), "time_ms": round((time.perf_counter() - t0) * 1000, 1), }
        self._log(f"完成: len={metrics['length_m']}, pr={metrics['path_ratio']}, "
                  f"minSD={metrics['min_clearance_m']}m, valid={valid}")
        return {"waypoints": pts, "metrics": metrics, "engine_used": engine, "log": self.log,
                "timer_report": self.timer.report()}


# ═══════════════════════════════════════════════════════════════
# 地图加载工具
# ═══════════════════════════════════════════════════════════════

def load_maps_from_dir(maps_dir: str, require_points: bool = True) -> List[
    Tuple[str, Dict, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]]:
    # require_points=False 时允许缺少 S/E 的地图进列表（配合 --pick-se 交互指定）
    p = Path(maps_dir)
    
    # 容错1：如果传入的是文件路径，自动取其所在目录
    if p.is_file() and p.suffix.lower() == '.json':
        print(f"  [INFO] 检测到传入的是文件路径，自动切换到所在目录: {p.parent}")
        p = p.parent
    
    # 容错2：支持相对路径和绝对路径，自动处理反斜杠
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    p = p.resolve()
    
    print(f"  [INFO] 解析后的绝对路径: {p}")
    print(f"  [INFO] 路径是否存在: {p.exists()}, 是否是目录: {p.is_dir()}")
    
    if not p.exists():
        raise FileNotFoundError(f"地图目录不存在: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"路径不是目录: {p}")
    
    json_files = list(p.glob("*.json"))
    print(f"  [INFO] 在目录中找到 {len(json_files)} 个 .json 文件")
    
    maps = []
    for json_path in sorted(json_files):
        print(f"  [INFO] 正在读取: {json_path.name}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] {json_path.name} 读取失败: {e}")
            continue
        
        # 兼容两种格式：含 points 的完整格式，以及简化格式
        pts = data.get("points", {})
        start = pts.get("start", {}).get("coordinates") if isinstance(pts, dict) else None
        end = pts.get("end", {}).get("coordinates") if isinstance(pts, dict) else None
        
        if start is None or end is None:
            if require_points:
                print(f"  [WARN] {json_path.name} 缺少起点/终点（points.start/end.coordinates），跳过")
                continue
            print(f"  [INFO] {json_path.name} 缺少起点/终点，将由 --pick-se 交互指定")
        
        name = json_path.stem
        obs_count = data.get("obstacles", {}).get("count", "?")
        maps.append((name, data, tuple(start), tuple(end)))
        print(f"  [LOAD] {name}: {start} -> {end}, obstacles={obs_count}")
    
    return maps


def _draw_obstacle(ax, obs):
    """按 shape_type 绘制障碍物（elp_v3 同款样式）。"""
    st = obs.get("shape_type", "rect")
    if st == "rect":
        ax.add_patch(
            Rectangle((obs["x1"], obs["y1"]), obs["x2"] - obs["x1"], obs["y2"] - obs["y1"], fc='#555555', ec='black',
                      lw=0.8, alpha=0.85, zorder=2))
    elif st == "circle":
        ax.add_patch(
            Circle(tuple(obs["center"]), obs["radius"], fc='#555555', ec='black', lw=0.8, alpha=0.85, zorder=2))
    elif st == "polygon":
        ax.add_patch(MplPolygon(obs["vertices"], closed=True, fc='#555555', ec='black', lw=0.8, alpha=0.85, zorder=2))


def _ensure_interactive_backend():
    """v7 模块级 matplotlib.use("Agg") 与交互点选冲突；需要时切回 GUI 后端。"""
    if matplotlib.get_backend().lower() == "agg":
        for cand in ("TkAgg", "Qt5Agg", "QtAgg", "WxAgg"):
            try:
                plt.switch_backend(cand)
                return
            except Exception:
                continue
        raise RuntimeError("当前环境无可用 GUI 后端，交互点选不可用；请去掉 --pick-se，或在带显示的环境运行")


def interactive_set_start_end(map_data: Dict, sdf: UnifiedSDFMap, S: Optional[Tuple[float, float]] = None,
                              E: Optional[Tuple[float, float]] = None, margin: float = 0.0) -> Tuple[
    Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """elp_v3 同款交互点选（移植 interactive_set_start_end）：

    按 [1] 进入起点模式，按 [2] 进入终点模式，单击放置；点击须满足
    sd >= margin（默认 0，即不得在障碍内/越界）。S/E 可预填（如地图 JSON
    提取值），两侧均可反复点击覆盖；[Enter]/q/Esc 或关闭窗口确认（v7.4 起
    单击不再自动关窗）。返回 (S, E)，未设置侧为 None。
    """
    _ensure_interactive_backend()
    w, h = map_data["map"]["width"], map_data["map"]["height"]
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        fig.canvas.manager.set_window_title("UnifiedPlanner - Pick S & E")
    except Exception:
        pass
    state = {"S": tuple(S) if S else None, "E": tuple(E) if E else None, "mode": None}
    
    def draw():
        ax.clear()
        ax.add_patch(Rectangle((0, 0), w, h, lw=2, ec="black", fc="#FAFAFA", zorder=1))
        for obs in map_data["obstacles"]["items"]:
            _draw_obstacle(ax, obs)
        if state["S"]:
            ax.scatter([state["S"][0]], [state["S"][1]], s=350, c="#00CC44", marker="o", ec="black", lw=2, zorder=10)
        if state["E"]:
            ax.scatter([state["E"][0]], [state["E"][1]], s=400, c="#FF3333", marker="*", ec="black", lw=2, zorder=10)
        ax.set_xlim(-2, w + 2)
        ax.set_ylim(-2, h + 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")
        title = {None: "按 [1]/[2] 切换起/终点模式，单击放置（入障/越界拒绝）；两侧可反复修改，[Enter] 确认",
                 "S": "[起点模式] 单击放置/覆盖起点（[Enter] 确认）", "E": "[终点模式] 单击放置/覆盖终点（[Enter] 确认）"}[
            state["mode"]]
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()
    
    def on_key(ev):
        if ev.key == "1":
            state["mode"] = "S"
            draw()
        elif ev.key == "2":
            state["mode"] = "E"
            draw()
        elif ev.key in ("enter", "return", "q", "escape"):
            # v7.4：显式确认才关闭——预填 S/E 时可反复覆盖修改两侧
            plt.close(fig)
    
    def on_click(ev):
        if ev.inaxes != ax or state["mode"] is None or ev.xdata is None:
            return
        x, y = round(ev.xdata, 2), round(ev.ydata, 2)
        sd = sdf.sd(x, y)
        if sd < margin:
            print("  [X] (%.2f, %.2f) 不安全（sd=%.2f < margin=%.2f），请重选" % (x, y, sd, margin))
            return
        state[state["mode"]] = (x, y)
        print("  [OK] %s = (%.2f, %.2f)，sd=%.2f（两侧均可继续修改，[Enter] 确认）" % (state["mode"], x, y, sd))
        state["mode"] = None
        draw()
    
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    draw()
    plt.show()
    return state["S"], state["E"]


def _rect(x1, y1, x2, y2, i=0):
    return {"uid": f"r{i}", "shape_type": "rect", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def make_map(width, height, obstacles):
    return {"map": {"width": width, "height": height}, "obstacles": {"count": len(obstacles), "items": obstacles},
            "points": {"start": {"coordinates": None}, "end": {"coordinates": None}}}


def test_maps():
    maps = []
    rng = _default_rng(7)
    obs = []
    for i in range(10):
        w, h = rng.uniform(4, 10), rng.uniform(3, 8)
        x, y = rng.uniform(10, 85 - w), rng.uniform(5, 55 - h)
        if abs(y + h / 2 - 30) < 6:
            continue
        obs.append(_rect(round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1), i))
    obs.append({"shape_type": "circle", "center": [50.0, 18.0], "radius": 4.0})
    obs.append({"shape_type": "polygon", "vertices": [[70, 40], [78, 44], [74, 52], [66, 48]]})
    maps.append(("A_open_field", make_map(100, 60, obs), (5.0, 55.0), (95.0, 5.0)))
    
    obs = [_rect(48, 0, 52, 26, 0), _rect(48, 34, 52, 60, 1), _rect(20, 40, 30, 47, 2), _rect(70, 10, 80, 17, 3)]
    maps.append(("B_narrow_gate", make_map(100, 60, obs), (8.0, 45.0), (92.0, 15.0)))
    
    obs = [_rect(15, 30, 44, 34, 0), _rect(47, 30, 58, 34, 1), _rect(66, 30, 85, 34, 2), _rect(30, 8, 40, 14, 3),
           _rect(60, 46, 70, 52, 4)]
    maps.append(("C_two_corridors", make_map(100, 60, obs), (5.0, 50.0), (95.0, 10.0)))
    
    obs = [_rect(30, 20, 32, 28, 0), _rect(30, 32, 32, 50, 1), _rect(30, 20, 70, 22, 2), _rect(30, 48, 70, 50, 3),
           _rect(68, 20, 70, 50, 4)]
    maps.append(("D_room_far_door", make_map(100, 60, obs), (65.0, 25.0), (95.0, 55.0)))
    return maps


# ═══════════════════════════════════════════════════════════════
# 可视化引擎
# ═══════════════════════════════════════════════════════════════

def plot_map_background(ax, sdf: UnifiedSDFMap, map_data: Dict, cmap='RdYlGn'):
    # v3: 背景 RGBA 预烘焙并缓存于 sdf，多子图复用；nearest 插值出图更快
    bg = sdf._bg_cache.get(cmap)
    if bg is None:
        norm = matplotlib.colors.Normalize(vmin=-2, vmax=5)
        bg = plt.get_cmap(cmap)(norm(sdf.grid))
        bg[..., 3] = 0.6  # alpha 预烘焙，等效原 imshow(alpha=0.6)
        sdf._bg_cache[cmap] = bg
    im = ax.imshow(bg, extent=[0, sdf.width, 0, sdf.height], origin='lower', interpolation='nearest')
    for o in map_data["obstacles"]["items"]:
        st = o.get("shape_type", "rect")
        if st == "rect":
            ax.add_patch(
                Rectangle((o["x1"], o["y1"]), o["x2"] - o["x1"], o["y2"] - o["y1"], fc='#333333', ec='black', lw=1.5,
                          alpha=0.9))
        elif st == "circle":
            ax.add_patch(Circle(o["center"], o["radius"], fc='#333333', ec='black', lw=1.5, alpha=0.9))
        else:
            ax.add_patch(MplPolygon(o["vertices"], closed=True, fc='#333333', ec='black', lw=1.5, alpha=0.9))
    return im


def compute_path_profile(sdf: UnifiedSDFMap, chain: List[Tuple[float, float]], n_samples=500):
    dists = [0.0]
    for i in range(len(chain) - 1):
        dists.append(dists[-1] + math.hypot(chain[i + 1][0] - chain[i][0], chain[i + 1][1] - chain[i][1]))
    total_len = dists[-1]
    if total_len < 1e-9:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    sample_ts = np.linspace(0, 1, n_samples)
    sample_dists = sample_ts * total_len
    
    xs, ys = [], []
    idx = 0
    for sd in sample_dists:
        while idx < len(dists) - 1 and dists[idx + 1] < sd:
            idx += 1
        if idx >= len(chain) - 1:
            xs.append(chain[-1][0])
            ys.append(chain[-1][1])
        else:
            seg_len = dists[idx + 1] - dists[idx]
            t = (sd - dists[idx]) / seg_len if seg_len > 1e-9 else 0.0
            xs.append(chain[idx][0] + t * (chain[idx + 1][0] - chain[idx][0]))
            ys.append(chain[idx][1] + t * (chain[idx + 1][1] - chain[idx][1]))
    
    xs = np.array(xs)
    ys = np.array(ys)
    sds = sdf.sd_batch(xs, ys)
    
    curvatures = np.zeros_like(xs)
    for i in range(1, len(xs) - 1):
        v1 = (xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        v2 = (xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        d1 = math.hypot(*v1)
        d2 = math.hypot(*v2)
        if d1 > 1e-9 and d2 > 1e-9:
            c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
            curvatures[i] = math.acos(c) / ((d1 + d2) / 2)
    
    return sample_dists, sds, curvatures


def visualize_single_map(name, map_data, S, E, results_by_engine, out_dir, cfg, show_all=False, sdf=None):
    if cfg.skip_viz:
        return None
    os.makedirs(out_dir, exist_ok=True)
    engines = list(results_by_engine.keys())
    n_eng = len(engines)
    
    if show_all:
        fig = plt.figure(figsize=(6 + n_eng * 6, 18))
        gs = GridSpec(3, n_eng + 1, figure=fig, width_ratios=[1.2] + [1.0] * n_eng)
    else:
        fig = plt.figure(figsize=(6 + n_eng * 5, 12))
        gs = GridSpec(2, n_eng + 1, figure=fig, width_ratios=[1.0] + [1.0] * n_eng)
    
    if sdf is None:
        sdf = UnifiedSDFMap(map_data, cfg.sdf_resolution)
    colors = {'plain': '#e74c3c', 'poisson': '#3498db', 'fmm': '#2ecc71'}
    
    # 行0: 路径对比总览
    ax_main = fig.add_subplot(gs[0, 0])
    im = plot_map_background(ax_main, sdf, map_data, cmap='RdYlGn')
    ax_main.plot(*S, 'go', ms=14, markeredgecolor='black', markeredgewidth=1.5, zorder=10, label='Start')
    ax_main.plot(*E, 'r*', ms=18, markeredgecolor='black', markeredgewidth=1.5, zorder=10, label='Goal')
    
    for eng in engines:
        res = results_by_engine[eng]
        if res["waypoints"]:
            chain = [S] + res["waypoints"] + [E]
            c = colors.get(eng, '#9b59b6')
            ax_main.plot([p[0] for p in chain], [p[1] for p in chain], '.-', color=c, lw=2.5, ms=4, alpha=0.85,
                         label=eng)
    ax_main.set_title(f"{name} - 多引擎路径对比", fontsize=13, fontweight='bold')
    ax_main.set_xlim(0, sdf.width)
    ax_main.set_ylim(0, sdf.height)
    ax_main.set_aspect('equal')
    ax_main.legend(loc='upper left', fontsize=9)
    _sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=matplotlib.colors.Normalize(vmin=-2, vmax=5))
    plt.colorbar(_sm, ax=ax_main, fraction=0.046, pad=0.04, label='SDF (m)')
    
    # 各引擎独立子图
    for c_idx, eng in enumerate(engines):
        res = results_by_engine[eng]
        m = res["metrics"]
        ax = fig.add_subplot(gs[0, c_idx + 1])
        im2 = plot_map_background(ax, sdf, map_data, cmap='RdYlGn')
        ax.plot(*S, 'go', ms=11, markeredgecolor='black', markeredgewidth=1.2, zorder=10)
        ax.plot(*E, 'r*', ms=15, markeredgecolor='black', markeredgewidth=1.2, zorder=10)
        
        if res["waypoints"]:
            chain = [S] + res["waypoints"] + [E]
            col = colors.get(eng, '#9b59b6')
            ax.plot([p[0] for p in chain], [p[1] for p in chain], '.-', color=col, lw=2.2, ms=4, alpha=0.9)
        
        title = f"{eng}\n"
        if m.get("success"):
            title += f"L={m.get('length_m', '-')}m  PR={m.get('path_ratio', '-')}\n"
            title += f"minSD={m.get('min_clearance_m', '-')}m  t={m.get('time_ms', '-')}ms"
        else:
            title += "FAILED"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, sdf.width)
        ax.set_ylim(0, sdf.height)
        ax.set_aspect('equal')
    
    # 行1: Clearance 剖面对比
    ax_clear = fig.add_subplot(gs[1, 0])
    for eng in engines:
        res = results_by_engine[eng]
        if res["waypoints"]:
            chain = [S] + res["waypoints"] + [E]
            dists, sds, _ = compute_path_profile(sdf, chain)
            col = colors.get(eng, '#9b59b6')
            ax_clear.plot(dists, sds, '-', color=col, lw=2, alpha=0.8, label=eng)
    ax_clear.axhline(cfg.safety_margin, color='red', ls='--', lw=1.5, alpha=0.7, label=f'margin={cfg.safety_margin}m')
    ax_clear.set_xlabel("Path distance (m)", fontsize=10)
    ax_clear.set_ylabel("Clearance (m)", fontsize=10)
    ax_clear.set_title("Clearance Profile", fontsize=11, fontweight='bold')
    ax_clear.legend(loc='upper right', fontsize=8)
    ax_clear.set_ylim(bottom=0)
    ax_clear.grid(True, alpha=0.3)
    
    for c_idx, eng in enumerate(engines):
        res = results_by_engine[eng]
        ax = fig.add_subplot(gs[1, c_idx + 1])
        if res["waypoints"]:
            chain = [S] + res["waypoints"] + [E]
            dists, sds, curvs = compute_path_profile(sdf, chain)
            col = colors.get(eng, '#9b59b6')
            ax.fill_between(dists, sds, alpha=0.3, color=col)
            ax.plot(dists, sds, '-', color=col, lw=2)
            ax2 = ax.twinx()
            ax2.plot(dists, curvs, '--', color='orange', lw=1.2, alpha=0.7, label='curvature')
            ax2.set_ylabel("Curvature (rad/m)", color='orange', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='orange')
        ax.axhline(cfg.safety_margin, color='red', ls='--', lw=1.2, alpha=0.6)
        ax.set_xlabel("Distance (m)", fontsize=9)
        ax.set_ylabel("Clearance (m)", fontsize=9)
        ax.set_title(f"{eng} - Profile", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 行2 (可选): 时间分解
    if show_all:
        ax_time = fig.add_subplot(gs[2, 0])
        stage_names = set()
        for eng in engines:
            tr = results_by_engine[eng].get("timer_report", {})
            stage_names.update(tr.keys())
        stage_names = sorted(stage_names)
        
        if stage_names:
            x = np.arange(len(stage_names))
            width = 0.8 / n_eng
            for i, eng in enumerate(engines):
                tr = results_by_engine[eng].get("timer_report", {})
                vals = [tr.get(s, {}).get("total_ms", 0) for s in stage_names]
                col = colors.get(eng, '#9b59b6')
                ax_time.bar(x + i * width - 0.4 + width / 2, vals, width, label=eng, color=col, alpha=0.8)
            ax_time.set_xticks(x)
            ax_time.set_xticklabels(stage_names, rotation=45, ha='right', fontsize=8)
            ax_time.set_ylabel("Time (ms)", fontsize=10)
            ax_time.set_title("Stage-wise Time Cost", fontsize=11, fontweight='bold')
            ax_time.legend(fontsize=8)
            ax_time.grid(True, alpha=0.3, axis='y')
        
        for c_idx, eng in enumerate(engines):
            res = results_by_engine[eng]
            ax = fig.add_subplot(gs[2, c_idx + 1])
            tr = res.get("timer_report", {})
            if tr:
                names = list(tr.keys())
                vals = [tr[n]["total_ms"] for n in names]
                wedges, texts, autotexts = ax.pie(vals, labels=None, autopct='%1.1f%%',
                                                  colors=plt.cm.Set3(np.linspace(0, 1, len(names))), startangle=90)
                for t in autotexts:
                    t.set_fontsize(7)
                ax.set_title(f"{eng} - Time Pie", fontsize=10)
                ax.legend(wedges, names, loc='center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7, frameon=False)
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{name}_report.png")
    _save_with_retry(plt.gcf(), fig_path, dpi=cfg.viz_dpi, bbox_inches='tight' if cfg.viz_dpi >= 140 else None)
    plt.close()
    print(f"  [VIZ] 可视化报告: {fig_path}")
    return fig_path


# ═══════════════════════════════════════════════════════════════
# 统计分析与导出
# ═══════════════════════════════════════════════════════════════

def export_csv(rows, out_dir):
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["map", "engine", "success", "valid", "length_m", "path_ratio", "min_clearance_m", "mean_clearance_m",
             "curvature_rad", "n_points", "time_ms"])
        for name, label, m in rows:
            if not m.get("success"):
                writer.writerow([name, label, False, False, "", "", "", "", "", "", ""])
            else:
                writer.writerow(
                    [name, label, m["success"], m["valid"], m["length_m"], m["path_ratio"], m["min_clearance_m"],
                     m["mean_clearance_m"], m["curvature_rad"], m["n_points"], m["time_ms"]])
    print(f"  [CSV] 统计表: {csv_path}")
    return csv_path


def export_json(full_results, out_dir):
    json_path = os.path.join(out_dir, "full_report.json")
    clean = {}
    for k, v in full_results.items():
        clean[k] = {eng: {"waypoints": res["waypoints"], "metrics": res["metrics"], "engine_used": res["engine_used"],
                          "log": res["log"], "timer_report": res.get("timer_report", {})} for eng, res in v.items()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] 详细报告: {json_path}")
    return json_path


def print_summary_table(rows):
    print("\n" + "=" * 120)
    print(f"{'map':<18}{'engine':<20}{'ok':<6}{'valid':<7}{'len(m)':<10}{'PR':<8}"
          f"{'minSD(m)':<10}{'meanSD(m)':<11}{'curv(rad)':<11}{'pts':<6}{'ms':<10}")
    print("-" * 120)
    for name, label, m in rows:
        if not m.get("success"):
            print(f"{name:<18}{label:<20}FAIL")
            continue
        print(f"{name:<18}{label:<20}{str(m['success']):<6}{str(m['valid']):<7}"
              f"{m['length_m']:<10.2f}{m['path_ratio']:<8.3f}{m['min_clearance_m']:<10.3f}"
              f"{m['mean_clearance_m']:<11.3f}{m['curvature_rad']:<11.3f}"
              f"{m['n_points']:<6}{m['time_ms']:<10.1f}")
    print("=" * 120)


def plot_global_comparison(rows, engines, out_dir):
    eng_stats = {}
    for _, label, m in rows:
        if not m.get("success"):
            continue
        if label not in eng_stats:
            eng_stats[label] = {"lengths": [], "prs": [], "min_sds": [], "times": [], "curvs": []}
        eng_stats[label]["lengths"].append(m["length_m"])
        eng_stats[label]["prs"].append(m["path_ratio"])
        eng_stats[label]["min_sds"].append(m["min_clearance_m"])
        eng_stats[label]["times"].append(m["time_ms"])
        eng_stats[label]["curvs"].append(m["curvature_rad"])
    
    if not eng_stats:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = [("lengths", "Length (m)", "mean"), ("prs", "Path Ratio", "mean"),
               ("min_sds", "Min Clearance (m)", "mean"), ("times", "Time (ms)", "mean"),
               ("curvs", "Curvature (rad)", "mean"), ("success_rate", "Success Rate", "rate")]
    
    colors = {'A* baseline (alpha=0)': '#e74c3c', 'Poisson-shaped A*': '#3498db', 'FMM nav-function': '#2ecc71'}
    
    for ax, (key, ylabel, agg) in zip(axes.flat, metrics):
        labels = []
        vals = []
        errs = []
        cols = []
        for label, data in eng_stats.items():
            if agg == "rate":
                total = sum(1 for r in rows if r[1] == label)
                succ = len(data["lengths"])
                v = succ / total * 100 if total > 0 else 0
                e = 0
            else:
                arr = np.array(data[key])
                v = arr.mean()
                e = arr.std()
            labels.append(label)
            vals.append(v)
            errs.append(e)
            cols.append(colors.get(label, '#9b59b6'))
        
        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, color=cols, alpha=0.8, capsize=4, edgecolor='black', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8, rotation=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "global_comparison.png")
    _save_with_retry(plt.gcf(), fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [VIZ] 全局对比图: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_batch(maps, engines, out_dir, cfg, show_all=False):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    full_results = {}
    
    for name, md, S, E in maps:
        print(f"\n[MAP] {name}: {S} -> {E}")
        full_results[name] = {}
        results_by_engine = {}
        try:
            planner = UnifiedPlanner(md, cfg)  # v3: 跨引擎复用 SDF/Poisson/图结构
        except Exception as ex:
            print(f"  [MAP ERROR] {name} 构建失败: {ex}")
            continue
        
        for eng, label in engines:
            print(f"  [ENGINE] {label} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                planner.reset_run_state()
                res = planner.plan(S, E, engine=eng)
            except Exception as ex:
                print(f"  ERROR: {ex}")
                res = {"waypoints": None, "metrics": {"success": False}, "log": [str(ex)], "engine_used": eng,
                       "timer_report": {}}
            dt = (time.perf_counter() - t0) * 1000
            m = res["metrics"]
            if m.get("success"):
                print(f"  OK  L={m.get('length_m', '-')}m  PR={m.get('path_ratio', '-')}  "
                      f"minSD={m.get('min_clearance_m', '-')}m  t={dt:.1f}ms")
            else:
                print(f"  FAILED")
            
            rows.append((name, label, m))
            full_results[name][eng] = res
            results_by_engine[eng] = res
        
        try:
            visualize_single_map(name, md, S, E, results_by_engine, out_dir, cfg, show_all=show_all, sdf=planner.sdf)
        except Exception as e:
            print(f"  [VIZ ERROR] {e}")
    
    print_summary_table(rows)
    export_csv(rows, out_dir)
    export_json(full_results, out_dir)
    
    try:
        plot_global_comparison(rows, engines, out_dir)
    except Exception as e:
        print(f"  [GLOBAL VIZ ERROR] {e}")
    
    return rows, full_results


# ═══════════════════════════════════════════════════════════════
# 运行配置
# ═══════════════════════════════════════════════════════════════

IDE = dict(maps_dir="./maps",  # 地图 JSON 目录；目录不存在时回退内置测试地图
           out_dir="results",  # 输出目录
           engine="all",  # all | plain | poisson | fmm | upp
           resolution=0.2,  # SDF 栅格分辨率（米）
           safety_margin=0.2,  # 安全裕度（米）；同时作为点选时的最小 sd 门槛
           alpha=1.0,  # 泊松整形强度
           pick_se=False,  # True → 每张地图弹窗交互点选 S/E（按 [1]/[2] 切换后单击；
           start=None,  # 手动起点 (x, y)，最高优先级（所有地图共用）；None 不覆盖
           end=None,  # 手动终点 (x, y)，同上
           viz_all=False,  # True → 可视化报告含时间分解行
           skip_viz=False,  # True → 只计算不画图（扫参用）
           )


def run(ide: Dict):
    """
    流程：构建配置 → 载图（pick_se 时允许缺 S/E 的 JSON 进入列表）→
    手动 S/E 全局覆盖 →（可选）逐图弹窗交互点选（预填现值，重选覆盖，
    关窗保留）→ run_batch 批处理出报告。
    """
    cfg = UConfig(sdf_resolution=ide["resolution"], safety_margin=ide["safety_margin"], alpha=ide["alpha"],
                  use_cache=True, verbose=True, skip_viz=ide["skip_viz"], use_numba=True)
    maps_dir = ide["maps_dir"]
    if maps_dir and Path(maps_dir).exists():
        print(f"[INFO] 地图目录: {maps_dir}")
        maps = load_maps_from_dir(maps_dir, require_points=not ide["pick_se"])
    else:
        if maps_dir:
            print(f"[WARN] 地图目录不存在: {maps_dir}，回退内置测试地图")
        else:
            print("[INFO] 使用内置测试地图")
        maps = test_maps()
    if not maps:
        print("[ERROR] 没有可用地图")
        return
    # 手动 S/E 全局覆盖（最高优先级）
    if ide["start"] is not None or ide["end"] is not None:
        maps = [(n, md, ide["start"] or S, ide["end"] or E) for n, md, S, E in maps]
        print(f"[INFO] 手动指定 S/E（全局）: {ide['start']} -> {ide['end']}")
    if ide["pick_se"]:
        picked = []
        for name, md, S, E in maps:
            print(f"\n[PICK] {name}: 现值 S={S} E={E}（[1]/[2] 切换，单击覆盖，可反复修改两侧，[Enter] 确认）")
            tmp = UnifiedSDFMap(md, ide["resolution"])
            Si, Ei = interactive_set_start_end(md, tmp, S=S, E=E, margin=ide["safety_margin"])
            picked.append((name, md, Si or S, Ei or E))
            del tmp
        maps = picked
    for name, md, S, E in maps:
        if S is None or E is None:
            print(f"[ERROR] {name} 起点/终点未设置，无法规划")
            return
    all_engines = [("plain", "A*"), ("poisson", "Poisson-A*"), ("fmm", "FMM-A*"), ("upp", "UPP")]
    if ide["engine"] == "all":
        engines = all_engines
    else:
        engines = [(e, l) for e, l in all_engines if e == ide["engine"]] or all_engines
    print(f"[INFO] 引擎: {[l for _, l in engines]}")
    print(f"[INFO] 输出目录: {ide['out_dir']}")
    run_batch(maps, engines, ide["out_dir"], cfg, show_all=ide["viz_all"])
    print(f"\n[DONE] 所有结果已保存至: {ide['out_dir']}/")


if __name__ == "__main__":
    run(IDE)
