from __future__ import annotations

import math
import random
import operator
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

CONST1 = "__CONST1__"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _rmse(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return float("inf")
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def _csi(y_true, y_prob, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_hat = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_hat == 1))
    fp = np.sum((y_true == 0) & (y_hat == 1))
    fn = np.sum((y_true == 1) & (y_hat == 0))
    den = tp + fp + fn
    return float(tp / den) if den > 0 else 0.0


def _norm01(a: np.ndarray) -> Tuple[np.ndarray, float, float]:
    a = np.asarray(a, dtype=float)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx == mn:
        return np.zeros_like(a, dtype=float), float(mn), float(mx)
    out = (a - mn) / (mx - mn)
    return np.clip(out, 0.0, 1.0), float(mn), float(mx)


def _inv_norm01(a01: np.ndarray, mn: float, mx: float) -> np.ndarray:
    a01 = np.asarray(a01, dtype=float)
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx == mn:
        return np.full_like(a01, np.nan, dtype=float)
    return a01 * (mx - mn) + mn


class EPLine:
    OPS = {"ADD": operator.add, "MULTIPLY": operator.mul}
    REL = {"<=": operator.le, ">": operator.gt}

    def __init__(self, predictor_names: Sequence[str], rng: random.Random):
        self.predictor_names = list(predictor_names)
        self.rng = rng
        self.v1 = rng.choice(self.predictor_names)
        self.v2 = rng.choice(self.predictor_names + [CONST1])
        self.v3 = rng.choice(self.predictor_names + [CONST1])
        self.v4 = rng.choice(self.predictor_names + [CONST1])
        self.v5 = rng.choice(self.predictor_names + [CONST1])
        self.c1 = rng.uniform(-1.0, 1.0)
        self.c2 = rng.uniform(-1.0, 1.0)
        self.c3 = rng.uniform(-1.0, 1.0)
        self.rel = rng.choice(list(self.REL.keys()))
        self.o1 = rng.choice(list(self.OPS.keys()))
        self.o2 = rng.choice(list(self.OPS.keys()))

    def copy(self) -> "EPLine":
        g = EPLine(self.predictor_names, self.rng)
        g.v1, g.v2, g.v3, g.v4, g.v5 = self.v1, self.v2, self.v3, self.v4, self.v5
        g.c1, g.c2, g.c3 = self.c1, self.c2, self.c3
        g.rel, g.o1, g.o2 = self.rel, self.o1, self.o2
        return g

    def used_variables(self) -> set[str]:
        return {v for v in [self.v1, self.v2, self.v3, self.v4, self.v5] if v != CONST1}

    def _val(self, row: Dict[str, float], name: str) -> float:
        if name == CONST1:
            return 1.0
        return _safe_float(row.get(name, np.nan))

    def evaluate(self, row: Dict[str, float]) -> float:
        a = self._val(row, self.v1)
        b = self._val(row, self.v2)
        if not (np.isfinite(a) and np.isfinite(b)):
            return 0.0
        if not self.REL[self.rel](a, b):
            return 0.0
        v3 = self._val(row, self.v3)
        v4 = self._val(row, self.v4)
        v5 = self._val(row, self.v5)
        if not (np.isfinite(v3) and np.isfinite(v4) and np.isfinite(v5)):
            return 0.0
        t1 = self.c1 * v3
        t2 = self.c2 * v4
        t3 = self.c3 * v5
        try:
            z = self.OPS[self.o1](t1, t2)
            z = self.OPS[self.o2](z, t3)
            return float(z)
        except Exception:
            return 0.0

    def mutate_2015(self, rng: random.Random) -> None:
        choices = [
            ("V1", 0.03125),
            ("REL", 0.03125),
            ("O1", 0.03125),
            ("O2", 0.03125),
            ("V2", 0.015625),
            ("V3", 0.015625),
            ("V4", 0.015625),
            ("V5", 0.015625),
            ("V2_TO_CONST1", 0.015625),
            ("V3_TO_CONST1", 0.015625),
            ("V4_TO_CONST1", 0.015625),
            ("V5_TO_CONST1", 0.015625),
            ("C1", 0.0833),
            ("C2", 0.0833),
            ("C3", 0.0833),
        ]
        label = rng.choices([k for k, _ in choices], weights=[w for _, w in choices], k=1)[0]
        if label == "V1":
            self.v1 = rng.choice(self.predictor_names)
        elif label == "V2":
            self.v2 = rng.choice(self.predictor_names + [CONST1])
        elif label == "V3":
            self.v3 = rng.choice(self.predictor_names + [CONST1])
        elif label == "V4":
            self.v4 = rng.choice(self.predictor_names + [CONST1])
        elif label == "V5":
            self.v5 = rng.choice(self.predictor_names + [CONST1])
        elif label == "V2_TO_CONST1":
            self.v2 = CONST1
        elif label == "V3_TO_CONST1":
            self.v3 = CONST1
        elif label == "V4_TO_CONST1":
            self.v4 = CONST1
        elif label == "V5_TO_CONST1":
            self.v5 = CONST1
        elif label == "REL":
            self.rel = rng.choice(list(self.REL.keys()))
        elif label == "O1":
            self.o1 = rng.choice(list(self.OPS.keys()))
        elif label == "O2":
            self.o2 = rng.choice(list(self.OPS.keys()))
        elif label == "C1":
            self.c1 = rng.uniform(-1.0, 1.0)
        elif label == "C2":
            self.c2 = rng.uniform(-1.0, 1.0)
        elif label == "C3":
            self.c3 = rng.uniform(-1.0, 1.0)

    def mutate_2018_2019(self, rng: random.Random) -> None:
        comp = rng.randrange(11)
        if comp == 0:
            self.v1 = rng.choice(self.predictor_names)
        elif comp == 1:
            self.v2 = rng.choice(self.predictor_names + [CONST1])
        elif comp == 2:
            self.v3 = rng.choice(self.predictor_names + [CONST1])
        elif comp == 3:
            self.v4 = rng.choice(self.predictor_names + [CONST1])
        elif comp == 4:
            self.v5 = rng.choice(self.predictor_names + [CONST1])
        elif comp == 5:
            self.rel = rng.choice(list(self.REL.keys()))
        elif comp == 6:
            self.o1 = rng.choice(list(self.OPS.keys()))
        elif comp == 7:
            self.o2 = rng.choice(list(self.OPS.keys()))
        elif comp == 8:
            self.c1 = rng.uniform(-1.0, 1.0)
        elif comp == 9:
            self.c2 = rng.uniform(-1.0, 1.0)
        else:
            self.c3 = rng.uniform(-1.0, 1.0)


class EPAlgorithm:
    def __init__(self, predictor_names: Sequence[str], n_lines: int, rng: random.Random, mode: str = "sum"):
        self.predictor_names = list(predictor_names)
        self.n_lines = int(n_lines)
        self.rng = rng
        self.mode = mode
        self.lines = [EPLine(self.predictor_names, rng) for _ in range(self.n_lines)]
        self.sex = rng.choice(["M", "F"])
        self.food_units = 0
        self.age = 0
        self.hunger = 0
        self.performance_train = -np.inf
        self.performance_val = -np.inf
        self.rmse_train = np.inf
        self.rmse_val = np.inf
        self.score = -np.inf

    def copy(self) -> "EPAlgorithm":
        c = EPAlgorithm(self.predictor_names, self.n_lines, self.rng, self.mode)
        c.lines = [g.copy() for g in self.lines]
        c.sex = self.sex
        c.food_units = self.food_units
        c.age = self.age
        c.hunger = self.hunger
        c.performance_train = self.performance_train
        c.performance_val = self.performance_val
        c.rmse_train = self.rmse_train
        c.rmse_val = self.rmse_val
        c.score = self.score
        return c

    def predict_row(self, row: Dict[str, float], baseline: float = 0.0) -> float:
        vals = [ln.evaluate(row) for ln in self.lines]
        if self.mode == "sum":
            out = baseline + float(np.sum(vals))
        elif self.mode == "fa":
            out = baseline + float(np.sum(vals))
        elif self.mode == "fb":
            out = baseline + vals[0] + vals[1] * vals[2] + vals[3] * vals[4]
        else:
            out = baseline + float(np.sum(vals))
        return float(out)

    def predict_array(self, rows: List[Dict[str, float]], baseline: Optional[np.ndarray] = None, clip01: bool = True) -> np.ndarray:
        if baseline is None:
            baseline = np.zeros(len(rows), dtype=float)
        out = np.fromiter((self.predict_row(rows[i], float(baseline[i])) for i in range(len(rows))), dtype=float, count=len(rows))
        if clip01:
            out = np.clip(out, 0.0, 1.0)
        return out

    def used_variables(self) -> set[str]:
        s = set()
        for ln in self.lines:
            s |= ln.used_variables()
        return s

    def copy_random_line_from(self, other: "EPAlgorithm", rng: random.Random) -> None:
        j = rng.randrange(self.n_lines)
        self.lines[j] = other.lines[j].copy()

    @staticmethod
    def crossover_2015(paternal: "EPAlgorithm", maternal: "EPAlgorithm", rng: random.Random) -> "EPAlgorithm":
        child = paternal.copy()
        for j in range(child.n_lines):
            if rng.random() < 0.5:
                child.lines[j].v1 = maternal.lines[j].v1
                child.lines[j].v2 = maternal.lines[j].v2
                child.lines[j].v3 = maternal.lines[j].v3
                child.lines[j].v4 = maternal.lines[j].v4
                child.lines[j].v5 = maternal.lines[j].v5
                child.lines[j].rel = maternal.lines[j].rel
                child.lines[j].o1 = maternal.lines[j].o1
                child.lines[j].o2 = maternal.lines[j].o2
        return child

    @staticmethod
    def transposition_2015(child: "EPAlgorithm", rng: random.Random) -> None:
        if child.n_lines < 2:
            return
        i = rng.randrange(child.n_lines)
        j = rng.randrange(child.n_lines)
        while j == i:
            j = rng.randrange(child.n_lines)
        src = child.lines[i]
        dst = child.lines[j]
        seg = rng.choice(["A", "B", "C", "D"])
        if seg == "A":
            dst.v1, dst.rel, dst.v2 = src.v1, src.rel, src.v2
        elif seg == "B":
            dst.c1, dst.v3, dst.o1 = src.c1, src.v3, src.o1
        elif seg == "C":
            dst.c2, dst.v4, dst.o2 = src.c2, src.v4, src.o2
        else:
            dst.c3, dst.v5 = src.c3, src.v5


def _rows_from_matrix(X: np.ndarray, predictor_names: Sequence[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i in range(X.shape[0]):
        rows.append({str(predictor_names[j]): float(X[i, j]) for j in range(X.shape[1])})
    return rows


def _normalize_matrix(X: np.ndarray, predictor_names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    X = np.asarray(X, dtype=float)
    Xn = np.empty_like(X, dtype=float)
    stats: Dict[str, Tuple[float, float]] = {}
    for j, name in enumerate(predictor_names):
        coln, mn, mx = _norm01(X[:, j])
        Xn[:, j] = coln
        stats[str(name)] = (mn, mx)
    return Xn, stats


def _apply_norm_stats(X: np.ndarray, predictor_names: Sequence[str], stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xn = np.empty_like(X, dtype=float)
    for j, name in enumerate(predictor_names):
        mn, mx = stats[str(name)]
        if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx == mn:
            Xn[:, j] = 0.0
        else:
            Xn[:, j] = np.clip((X[:, j] - mn) / (mx - mn), 0.0, 1.0)
    return Xn


@dataclass
class SingleSiteData:
    predictor_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: Optional[np.ndarray] = None

    def normalized(self):
        ytr, ymn, ymx = _norm01(self.y_train)
        Xtr, xstats = _normalize_matrix(self.X_train, self.predictor_names)
        Xv = _apply_norm_stats(self.X_val, self.predictor_names, xstats)
        yv = np.clip((self.y_val - ymn) / (ymx - ymn), 0.0, 1.0) if ymx != ymn else np.zeros_like(self.y_val, dtype=float)
        Xt = None
        if self.X_test is not None:
            Xt = _apply_norm_stats(self.X_test, self.predictor_names, xstats)
        return Xtr, ytr, Xv, yv, Xt, xstats, (ymn, ymx)


@dataclass
class Roebber2015Config:
    population_size: int = 10000
    n_lines: int = 10
    generations: int = 50
    max_mates_per_male: int = 10
    mutation_probability: float = 0.5
    transposition_probability: float = 0.5
    retain_top: int = 100
    random_state: int = 42


class Roebber2015EP:
    def __init__(self, config: Roebber2015Config = Roebber2015Config()):
        self.cfg = config
        self.rng = random.Random(config.random_state)
        self.survivors_: List[EPAlgorithm] = []
        self.y_stats_: Optional[Tuple[float, float]] = None
        self.x_stats_: Dict[str, Tuple[float, float]] = {}
        self.predictor_names_: List[str] = []

    def _score(self, alg: EPAlgorithm, rows_tr, ytr, rows_va, yva) -> None:
        ptr = alg.predict_array(rows_tr)
        pva = alg.predict_array(rows_va)
        alg.rmse_train = _rmse(ytr, ptr)
        alg.rmse_val = _rmse(yva, pva)
        alg.performance_train = -alg.rmse_train
        alg.performance_val = -alg.rmse_val
        alg.score = alg.performance_val

    def fit(self, data: SingleSiteData) -> "Roebber2015EP":
        Xtr, ytr, Xv, yv, _, xstats, ystats = data.normalized()
        rows_tr = _rows_from_matrix(Xtr, data.predictor_names)
        rows_va = _rows_from_matrix(Xv, data.predictor_names)
        self.predictor_names_ = list(data.predictor_names)
        self.x_stats_ = xstats
        self.y_stats_ = ystats

        pop = [EPAlgorithm(data.predictor_names, self.cfg.n_lines, self.rng, mode="sum") for _ in range(self.cfg.population_size)]
        archive: List[EPAlgorithm] = []

        for _ in range(self.cfg.generations):
            for ind in pop:
                self._score(ind, rows_tr, ytr, rows_va, yv)
            pop.sort(key=lambda z: z.rmse_train)
            archive.extend(sorted(pop, key=lambda z: z.rmse_val)[: self.cfg.retain_top])
            archive = sorted(archive, key=lambda z: z.rmse_val)[: self.cfg.retain_top]

            males = [z for z in pop if z.sex == "M"]
            females = [z for z in pop if z.sex == "F"]
            if not males or not females:
                for z in pop:
                    z.sex = self.rng.choice(["M", "F"])
                males = [z for z in pop if z.sex == "M"]
                females = [z for z in pop if z.sex == "F"]

            next_pop: List[EPAlgorithm] = [z.copy() for z in pop[: self.cfg.retain_top]]
            males = males[: self.cfg.retain_top]
            females = females[: self.cfg.retain_top]
            used_females = set()

            for m in males:
                candidates = [f for i, f in enumerate(females) if i not in used_females]
                if not candidates:
                    break
                n_mates = min(self.cfg.max_mates_per_male, len(candidates))
                mates = self.rng.sample(candidates, n_mates)
                for f in mates:
                    fi = females.index(f)
                    used_females.add(fi)
                    child = EPAlgorithm.crossover_2015(m, f, self.rng)
                    if self.rng.random() < self.cfg.mutation_probability:
                        child.lines[self.rng.randrange(child.n_lines)].mutate_2015(self.rng)
                    if self.rng.random() < self.cfg.transposition_probability:
                        EPAlgorithm.transposition_2015(child, self.rng)
                    child.sex = self.rng.choice(["M", "F"])
                    next_pop.append(child)
                    if len(next_pop) >= self.cfg.population_size:
                        break
                if len(next_pop) >= self.cfg.population_size:
                    break

            while len(next_pop) < self.cfg.population_size:
                child = EPAlgorithm(data.predictor_names, self.cfg.n_lines, self.rng, mode="sum")
                next_pop.append(child)
            pop = next_pop[: self.cfg.population_size]

        for ind in pop:
            self._score(ind, rows_tr, ytr, rows_va, yv)
        archive.extend(sorted(pop, key=lambda z: z.rmse_val)[: self.cfg.retain_top])
        self.survivors_ = sorted(archive, key=lambda z: z.rmse_val)[: self.cfg.retain_top]
        return self

    def predict(self, X: np.ndarray, max_members: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        Xn = _apply_norm_stats(X, self.predictor_names_, self.x_stats_)
        rows = _rows_from_matrix(Xn, self.predictor_names_)
        members = self.survivors_ if max_members is None else self.survivors_[:max_members]
        ens01 = np.vstack([m.predict_array(rows) for m in members])
        ymn, ymx = self.y_stats_
        ens = _inv_norm01(ens01, ymn, ymx)
        return np.nanmean(ens, axis=0), ens


@dataclass
class Roebber2018Config:
    population_size: int = 50
    eliminate: int = 10
    generations: int = 50
    n_lines: int = 5
    random_state: int = 42


def weighted_decay_bias_correction(forecast: np.ndarray, obs: np.ndarray) -> np.ndarray:
    forecast = np.asarray(forecast, dtype=float)
    obs = np.asarray(obs, dtype=float)
    bias = np.zeros_like(forecast, dtype=float)
    for t in range(1, len(forecast)):
        bias[t] = 0.95 * bias[t - 1] + 0.05 * (forecast[t] - obs[t])
    return forecast - bias


class Roebber2018SpatialEP:
    def __init__(self, config: Roebber2018Config = Roebber2018Config()):
        self.cfg = config
        self.rng = random.Random(config.random_state)
        self.best_: Dict[Tuple[int, int], EPAlgorithm] = {}
        self.x_stats_: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}
        self.y_stats_: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.predictor_names_: List[str] = []

    @staticmethod
    def _sigma_M(i: int, M: int) -> float:
        return float(1.0 + ((1.0 - i / 100.0) ** 3) * abs(math.cos(math.pi * i / 10.0)) * (M / 10.0 - 1.0))

    def _sample_remote_idx(self, nx: int, ny: int, ix: int, iy: int, gen: int) -> Tuple[int, int]:
        sx = self._sigma_M(gen, nx)
        sy = self._sigma_M(gen, ny)
        x = int(round(self.rng.gauss(ix, sx)))
        y = int(round(self.rng.gauss(iy, sy)))
        x = max(0, min(nx - 1, x))
        y = max(0, min(ny - 1, y))
        return x, y

    def _fit_point(self, Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray, yv: np.ndarray, ix: int, iy: int, nx: int, ny: int, grid_best: Dict[Tuple[int, int], EPAlgorithm], predictor_names: Sequence[str]) -> EPAlgorithm:
        Xtrn, xstats = _normalize_matrix(Xtr, predictor_names)
        ytrn, ymn, ymx = _norm01(ytr)
        Xvn = _apply_norm_stats(Xv, predictor_names, xstats)
        yvn = np.clip((yv - ymn) / (ymx - ymn), 0.0, 1.0) if ymx != ymn else np.zeros_like(yv, dtype=float)
        rows_tr = _rows_from_matrix(Xtrn, predictor_names)
        rows_va = _rows_from_matrix(Xvn, predictor_names)
        base_tr = Xtrn[:, predictor_names.index("RFV2_MEAN")] if "RFV2_MEAN" in predictor_names else np.zeros(Xtrn.shape[0])
        base_va = Xvn[:, predictor_names.index("RFV2_MEAN")] if "RFV2_MEAN" in predictor_names else np.zeros(Xvn.shape[0])

        pop = [EPAlgorithm(predictor_names, self.cfg.n_lines, self.rng, mode="sum") for _ in range(self.cfg.population_size)]
        best = None
        best_rmse = np.inf

        for gen in range(1, self.cfg.generations + 1):
            for ind in pop:
                ptr = ind.predict_array(rows_tr, base_tr)
                pva = ind.predict_array(rows_va, base_va)
                ind.rmse_train = _rmse(ytrn, ptr)
                ind.rmse_val = _rmse(yvn, pva)
            pop.sort(key=lambda z: z.rmse_train)
            if pop[0].rmse_val < best_rmse:
                best = pop[0].copy()
                best_rmse = pop[0].rmse_val

            survivors = [z.copy() for z in pop[:-self.cfg.eliminate]]
            offspring: List[EPAlgorithm] = []

            rx, ry = self._sample_remote_idx(nx, ny, ix, iy, gen)
            donor = grid_best.get((rx, ry), pop[0])
            first = donor.copy()
            first.lines[self.rng.randrange(first.n_lines)].mutate_2018_2019(self.rng)
            offspring.append(first)

            local_best = pop[: self.cfg.eliminate - 1]
            for donor in local_best:
                c = donor.copy()
                c.lines[self.rng.randrange(c.n_lines)].mutate_2018_2019(self.rng)
                offspring.append(c)

            pop = survivors + offspring

        self.x_stats_[(ix, iy)] = xstats
        self.y_stats_[(ix, iy)] = (ymn, ymx)
        return best if best is not None else pop[0]

    def fit(self, X_train: xr.DataArray, y_train: xr.DataArray, X_val: xr.DataArray, y_val: xr.DataArray) -> "Roebber2018SpatialEP":
        Xtr = X_train.transpose("T", "M", "Y", "X")
        Xva = X_val.transpose("T", "M", "Y", "X")
        Ytr = y_train.transpose("T", "Y", "X")
        Yva = y_val.transpose("T", "Y", "X")
        self.predictor_names_ = [str(v) for v in Xtr["M"].values.tolist()]
        ny, nx = Ytr.sizes["Y"], Ytr.sizes["X"]
        best_map: Dict[Tuple[int, int], EPAlgorithm] = {}

        for iy in range(ny):
            for ix in range(nx):
                Xtrp = Xtr.isel(Y=iy, X=ix).values
                Xvap = Xva.isel(Y=iy, X=ix).values
                ytrp = Ytr.isel(Y=iy, X=ix).values
                yvap = Yva.isel(Y=iy, X=ix).values
                m = np.isfinite(ytrp) & np.all(np.isfinite(Xtrp), axis=1)
                mv = np.isfinite(yvap) & np.all(np.isfinite(Xvap), axis=1)
                if np.sum(m) < 10 or np.sum(mv) < 10:
                    continue
                best_map[(ix, iy)] = self._fit_point(Xtrp[m], ytrp[m], Xvap[mv], yvap[mv], ix, iy, nx, ny, best_map, self.predictor_names_)
        self.best_ = best_map
        return self

    def predict(self, X_test: xr.DataArray) -> xr.DataArray:
        Xte = X_test.transpose("T", "M", "Y", "X")
        out = np.full((Xte.sizes["T"], Xte.sizes["Y"], Xte.sizes["X"]), np.nan, dtype=float)
        for (ix, iy), alg in self.best_.items():
            Xt = Xte.isel(Y=iy, X=ix).values
            m = np.all(np.isfinite(Xt), axis=1)
            if not np.any(m):
                continue
            Xn = _apply_norm_stats(Xt[m], self.predictor_names_, self.x_stats_[(ix, iy)])
            rows = _rows_from_matrix(Xn, self.predictor_names_)
            base = Xn[:, self.predictor_names_.index("RFV2_MEAN")] if "RFV2_MEAN" in self.predictor_names_ else np.zeros(Xn.shape[0])
            pred01 = alg.predict_array(rows, base)
            ymn, ymx = self.y_stats_[(ix, iy)]
            pred = _inv_norm01(pred01, ymn, ymx)
            out[m, iy, ix] = pred
        return xr.DataArray(out, coords={"T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]}, dims=("T", "Y", "X"), name="roebber2018")

    @staticmethod
    def bmc_weights_5(train_members: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        raw_levels = [0, 1, 2, 3, 4, 5]
        best_w = None
        best_logp = -np.inf
        median_ref = np.median(train_members, axis=1)
        n = len(obs)
        for w0 in raw_levels:
            for w1 in raw_levels:
                for w2 in raw_levels:
                    for w3 in raw_levels:
                        for w4 in raw_levels:
                            raw = np.array([w0, w1, w2, w3, w4], dtype=float)
                            if np.all(raw == 0):
                                continue
                            w = raw / raw.sum()
                            fc = np.sum(train_members * w[None, :], axis=1)
                            r = int(np.sum((fc - obs) ** 2 < (median_ref - obs) ** 2))
                            err = np.mean((fc - obs) ** 2 >= (median_ref - obs) ** 2)
                            eps = min(0.499999, max(1e-6, err))
                            logp = -np.log(4.0) + r * np.log(1.0 - eps) + (n - r) * np.log(eps)
                            if logp > best_logp:
                                best_logp = logp
                                best_w = w
        if best_w is None:
            best_w = np.ones(5) / 5.0
        sigma2 = float(np.mean(np.sum(best_w[None, :] * (train_members - obs[:, None]) ** 2, axis=1)))
        return best_w, sigma2


@dataclass
class Roebber2018AdaptiveConfig:
    n_lines: int = 5
    fast_trials: int = 10
    generations_initial: int = 50
    train_window: int = 730
    val_window: int = 365
    random_state: int = 42


class Roebber2018AdaptiveEP:
    def __init__(self, config: Roebber2018AdaptiveConfig = Roebber2018AdaptiveConfig()):
        self.cfg = config
        self.rng = random.Random(config.random_state)
        self.current_best_: Optional[EPAlgorithm] = None
        self.predictor_names_: List[str] = []
        self.x_stats_: Dict[str, Tuple[float, float]] = {}
        self.y_stats_: Optional[Tuple[float, float]] = None

    def _optimize_coefficients(self, alg: EPAlgorithm, rows_va: List[Dict[str, float]], yva: np.ndarray, baseline_va: np.ndarray) -> EPAlgorithm:
        best = alg.copy()
        best_rmse = np.inf
        for _ in range(self.cfg.fast_trials):
            c = alg.copy()
            for ln in c.lines:
                ln.c1 = self.rng.uniform(-1.0, 1.0)
                ln.c2 = self.rng.uniform(-1.0, 1.0)
                ln.c3 = self.rng.uniform(-1.0, 1.0)
            p = c.predict_array(rows_va, baseline_va)
            rmse = _rmse(yva, p)
            if rmse < best_rmse:
                best = c
                best_rmse = rmse
        return best

    def fit_initial(self, data: SingleSiteData) -> "Roebber2018AdaptiveEP":
        trainer = Roebber2015EP(Roebber2015Config(population_size=2000, n_lines=self.cfg.n_lines, generations=self.cfg.generations_initial, retain_top=1, random_state=self.cfg.random_state))
        trainer.fit(data)
        self.current_best_ = trainer.survivors_[0].copy()
        self.predictor_names_ = trainer.predictor_names_
        self.x_stats_ = trainer.x_stats_
        self.y_stats_ = trainer.y_stats_
        return self

    def step(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> EPAlgorithm:
        Xtr = _apply_norm_stats(X_train, self.predictor_names_, self.x_stats_)
        Xva = _apply_norm_stats(X_val, self.predictor_names_, self.x_stats_)
        ymn, ymx = self.y_stats_
        ytr = np.clip((y_train - ymn) / (ymx - ymn), 0.0, 1.0)
        yva_n = np.clip((y_val - ymn) / (ymx - ymn), 0.0, 1.0)
        rows_tr = _rows_from_matrix(Xtr, self.predictor_names_)
        rows_va = _rows_from_matrix(Xva, self.predictor_names_)
        base_tr = np.zeros(len(rows_tr))
        base_va = np.zeros(len(rows_va))
        cand = self.current_best_.copy()
        for ln in cand.lines:
            ln.mutate_2018_2019(self.rng)
        cand = self._optimize_coefficients(cand, rows_va, yva_n, base_va)
        prev_rmse = _rmse(yva_n, self.current_best_.predict_array(rows_va, base_va))
        new_rmse = _rmse(yva_n, cand.predict_array(rows_va, base_va))
        if new_rmse < prev_rmse:
            self.current_best_ = cand
        return self.current_best_.copy()


@dataclass
class Creature:
    alg: EPAlgorithm
    species: str
    x: int
    y: int


@dataclass
class Roebber2019Config:
    grid_size: int = 100
    prey_capacity: int = 5000
    predator_capacity: int = 5000
    init_prey: int = 5000
    init_pred: int = 1667
    generations: int = 70
    l_temperature: float = 36.2275
    c_temperature: float = 0.125
    diversity_threshold: float = 0.05
    random_state: int = 42


class Roebber2019PredatorPreyEP:
    def __init__(self, predictor_names: Sequence[str], config: Roebber2019Config = Roebber2019Config()):
        self.predictor_names = list(predictor_names)
        self.cfg = config
        self.rng = random.Random(config.random_state)
        self.prey: List[Creature] = []
        self.predators: List[Creature] = []
        self.top100_: List[EPAlgorithm] = []
        self.food_grid_: List[List[set[str]]] = []
        self.reference_rmse_: float = np.nan
        self.x_stats_: Dict[str, Tuple[float, float]] = {}
        self.y_stats_: Optional[Tuple[float, float]] = None

    def _a(self, metric: float) -> float:
        return max(0.25, 1.0 / (1.0 + math.exp(-self.cfg.l_temperature * (metric - 0.0294))))

    def _b(self, a: float) -> float:
        return self.cfg.c_temperature * (1.0 - a)

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        out = []
        for j in range(max(0, y - 1), min(self.cfg.grid_size, y + 2)):
            for i in range(max(0, x - 1), min(self.cfg.grid_size, x + 2)):
                out.append((i, j))
        return out

    def _spawn_food(self):
        food_vars = set(self.predictor_names[-7:] if len(self.predictor_names) >= 7 else self.predictor_names)
        food_vars.add(CONST1)
        g = []
        for _ in range(self.cfg.grid_size):
            row = []
            for _ in range(self.cfg.grid_size):
                k = self.rng.randint(1, min(5, len(food_vars)))
                row.append(set(self.rng.sample(list(food_vars), k)))
            g.append(row)
        self.food_grid_ = g

    def _init_population(self):
        self.prey = []
        self.predators = []
        for _ in range(self.cfg.init_prey):
            mode = self.rng.choice(["fa", "fb"])
            alg = EPAlgorithm(self.predictor_names, 5, self.rng, mode=mode)
            self.prey.append(Creature(alg, "prey", self.rng.randrange(self.cfg.grid_size), self.rng.randrange(self.cfg.grid_size)))
        for _ in range(self.cfg.init_pred):
            alg = EPAlgorithm(self.predictor_names, 5, self.rng, mode="fa")
            self.predators.append(Creature(alg, "predator", self.rng.randrange(self.cfg.grid_size), self.rng.randrange(self.cfg.grid_size)))

    def _evaluate(self, rows_tr, ytr, rows_va, yva, baseline_tr, baseline_va):
        ref = _rmse(ytr, baseline_tr)
        self.reference_rmse_ = ref
        for c in self.prey + self.predators:
            ptr = c.alg.predict_array(rows_tr, baseline_tr)
            pva = c.alg.predict_array(rows_va, baseline_va)
            c.alg.rmse_train = _rmse(ytr, ptr)
            c.alg.rmse_val = _rmse(yva, pva)
            metric = (ref - c.alg.rmse_train) / ref if ref > 0 else 0.0
            c.alg.performance_train = metric
            c.alg.performance_val = (ref - c.alg.rmse_val) / ref if ref > 0 else 0.0
            c.alg.score = c.alg.performance_val
        pool = sorted([c.alg.copy() for c in self.prey + self.predators], key=lambda z: z.rmse_val)
        self.top100_ = pool[:100]

    def _move_prey(self, c: Creature):
        a = self._a(c.alg.performance_train)
        nbrs = self._neighbors(c.x, c.y)
        if self.rng.random() < a:
            best_site = (c.x, c.y)
            best_food = -1
            for x, y in nbrs:
                if any((p.x == x and p.y == y) for p in self.predators):
                    continue
                food = self.food_grid_[y][x]
                score = len(food & c.alg.used_variables())
                if score > best_food:
                    best_food = score
                    best_site = (x, y)
            c.x, c.y = best_site
        else:
            c.x, c.y = self.rng.choice(nbrs)

    def _move_predator(self, c: Creature):
        a = self._a(c.alg.performance_train)
        nbrs = self._neighbors(c.x, c.y)
        if self.rng.random() < a:
            counts = []
            for x, y in nbrs:
                counts.append((sum((q.x == x and q.y == y) for q in self.prey), x, y))
            _, c.x, c.y = max(counts)
        else:
            c.x, c.y = self.rng.choice(nbrs)

    def _feeding(self):
        prey_map: Dict[Tuple[int, int], List[Creature]] = {}
        for q in self.prey:
            prey_map.setdefault((q.x, q.y), []).append(q)
        alive_prey = []
        for q in self.prey:
            food = self.food_grid_[q.y][q.x]
            if q.alg.used_variables().issubset(food):
                q.alg.hunger = 0
                alive_prey.append(q)
            else:
                q.alg.hunger += 1
                alive_prey.append(q)
        self.prey = alive_prey
        for p in self.predators:
            victims = prey_map.get((p.x, p.y), [])
            if victims:
                vic = victims[0]
                if vic in self.prey:
                    self.prey.remove(vic)
                    p.alg.food_units += 1
                    p.alg.hunger = 0
            else:
                p.alg.hunger += 1

    def _death(self):
        prey2 = []
        for q in self.prey:
            a = self._a(q.alg.performance_train)
            b = self._b(a)
            q.alg.age += 1
            dead = False
            if q.alg.hunger >= 5 and self.rng.random() < b:
                dead = True
            if q.alg.age >= 6 and self.rng.random() < b:
                dead = True
            if not dead:
                prey2.append(q)
        self.prey = prey2

        pred2 = []
        for p in self.predators:
            a = self._a(p.alg.performance_train)
            b = self._b(a)
            p.alg.age += 1
            dead = False
            if p.alg.food_units <= 0 and self.rng.random() < b:
                dead = True
            if p.alg.age >= 8 and self.rng.random() < b:
                dead = True
            if not dead:
                pred2.append(p)
        self.predators = pred2

    def _reproduce(self):
        new_prey: List[Creature] = []
        for q in self.prey:
            a = self._a(q.alg.performance_train)
            if len(self.prey) + len(new_prey) >= self.cfg.prey_capacity:
                break
            if q.alg.hunger == 0:
                c = q.alg.copy()
                if self.rng.random() < (1.0 - a):
                    line = c.lines[self.rng.randrange(c.n_lines)]
                    if self.rng.random() < 2.0 / 3.0:
                        line.mutate_2018_2019(self.rng)
                    else:
                        self.rng.choice(c.lines).mutate_2018_2019(self.rng)
                x, y = self.rng.choice(self._neighbors(q.x, q.y))
                new_prey.append(Creature(c, "prey", x, y))
        self.prey.extend(new_prey)

        new_pred: List[Creature] = []
        for p in self.predators:
            a = self._a(p.alg.performance_train)
            if len(self.predators) + len(new_pred) >= self.cfg.predator_capacity:
                break
            if p.alg.food_units >= 2:
                c = p.alg.copy()
                p.alg.food_units -= 2
                if self.rng.random() < (1.0 - a):
                    self.rng.choice(c.lines).mutate_2018_2019(self.rng)
                x, y = self.rng.choice(self._neighbors(p.x, p.y))
                new_pred.append(Creature(c, "predator", x, y))
        self.predators.extend(new_pred)

    def _learn(self):
        for species in [self.prey, self.predators]:
            for c in species:
                nbrs = self._neighbors(c.x, c.y)
                local = [q for q in species if (q.x, q.y) in nbrs]
                if not local:
                    continue
                best = max(local, key=lambda z: z.alg.performance_train)
                if best.alg.performance_train > c.alg.performance_train:
                    c.alg.copy_random_line_from(best.alg, self.rng)

    def fit(self, data: SingleSiteData) -> "Roebber2019PredatorPreyEP":
        Xtr, ytr, Xv, yv, _, xstats, ystats = data.normalized()
        self.x_stats_ = xstats
        self.y_stats_ = ystats
        rows_tr = _rows_from_matrix(Xtr, data.predictor_names)
        rows_va = _rows_from_matrix(Xv, data.predictor_names)
        base_tr = np.zeros(len(rows_tr))
        base_va = np.zeros(len(rows_va))
        self._spawn_food()
        self._init_population()

        for _ in range(self.cfg.generations):
            self._evaluate(rows_tr, ytr, rows_va, yv, base_tr, base_va)
            for q in self.prey:
                self._move_prey(q)
            for p in self.predators:
                self._move_predator(p)
            self._feeding()
            self._death()
            self._reproduce()
            self._learn()

        self._evaluate(rows_tr, ytr, rows_va, yv, base_tr, base_va)
        self.top100_ = sorted(self.top100_, key=lambda z: z.rmse_val)[:100]
        return self

    def _select_diverse_top5(self) -> List[EPAlgorithm]:
        ranked = sorted(self.top100_, key=lambda z: z.rmse_val)
        if not ranked:
            return []
        selected = [ranked[0]]
        if len(ranked) == 1:
            return selected
        preds = {}
        for k, alg in enumerate(ranked):
            preds[k] = alg
        avg_rmsd = 0.0
        pairs = 0
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                d = 0.0
                for li, lj in zip(ranked[i].lines, ranked[j].lines):
                    d += abs(li.c1 - lj.c1) + abs(li.c2 - lj.c2) + abs(li.c3 - lj.c3)
                avg_rmsd += d
                pairs += 1
        avg_rmsd = avg_rmsd / pairs if pairs > 0 else 0.0
        for alg in ranked[1:]:
            ok = True
            for s in selected:
                d = 0.0
                for li, lj in zip(alg.lines, s.lines):
                    d += abs(li.c1 - lj.c1) + abs(li.c2 - lj.c2) + abs(li.c3 - lj.c3)
                if d <= self.cfg.diversity_threshold * avg_rmsd:
                    ok = False
                    break
            if ok:
                selected.append(alg)
            if len(selected) == 5:
                break
        return selected[:5]

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xn = _apply_norm_stats(X, self.predictor_names, self.x_stats_)
        rows = _rows_from_matrix(Xn, self.predictor_names)
        top5 = self._select_diverse_top5()
        ens01 = np.vstack([alg.predict_array(rows, np.zeros(len(rows))) for alg in top5])
        ymn, ymx = self.y_stats_
        ens = _inv_norm01(ens01, ymn, ymx)
        return np.nanmean(ens, axis=0), ens


__all__ = [
    "CONST1",
    "EPLine",
    "EPAlgorithm",
    "SingleSiteData",
    "Roebber2015Config",
    "Roebber2015EP",
    "Roebber2018Config",
    "Roebber2018SpatialEP",
    "Roebber2018AdaptiveConfig",
    "Roebber2018AdaptiveEP",
    "Roebber2019Config",
    "Roebber2019PredatorPreyEP",
    "weighted_decay_bias_correction",
]

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def _split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    min_train_samples: int,
    min_val_samples: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]
    n = y.size
    if n < (min_train_samples + min_val_samples):
        return None
    n_val = max(min_val_samples, int(np.ceil(validation_fraction * n)))
    n_val = min(n_val, n - min_train_samples)
    if n_val < min_val_samples:
        return None
    split = n - n_val
    if split < min_train_samples:
        return None
    return X[:split], y[:split], X[split:], y[split:]


def _stack_train_predictors(X: xr.DataArray) -> np.ndarray:
    return X.transpose("T", "M", "Y", "X").values


def _stack_target(y: xr.DataArray) -> np.ndarray:
    return y.transpose("T", "Y", "X").values


def _ensure_xy_order(X: xr.DataArray, y: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    Xo = X.transpose("T", "M", "Y", "X")
    yo = y.transpose("T", "Y", "X")
    if not np.array_equal(Xo["T"].values, yo["T"].values):
        raise ValueError("X and y must share the same T coordinate")
    if not np.array_equal(Xo["Y"].values, yo["Y"].values):
        raise ValueError("X and y must share the same Y coordinate")
    if not np.array_equal(Xo["X"].values, yo["X"].values):
        raise ValueError("X and y must share the same X coordinate")
    return Xo, yo


def _maybe_time_from_test(y_mean: xr.DataArray, X_test: xr.DataArray) -> xr.DataArray:
    Xte = X_test.transpose("T", "M", "Y", "X")
    return y_mean.assign_coords(T=Xte["T"])


class WAS_RoebberBase:
    @staticmethod
    def tercile_thresholds_from_obs(
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        Y = Predictant.transpose("T", "Y", "X").sel(T=slice(str(clim_year_start), str(clim_year_end)))
        q33 = Y.quantile(1.0 / 3.0, dim="T")
        q66 = Y.quantile(2.0 / 3.0, dim="T")
        return q33, q66

    @staticmethod
    def tercile_probs_from_ensemble(
        ens: xr.DataArray,
        q33: xr.DataArray,
        q66: xr.DataArray,
    ) -> xr.DataArray:
        ens = ens.transpose("member", "T", "Y", "X")
        below = (ens < q33).mean(dim="member")
        above = (ens > q66).mean(dim="member")
        normal = 1.0 - below - above
        prob = xr.concat(
            [
                below.drop_vars("quantile", errors="ignore"),
                normal.drop_vars("quantile", errors="ignore"),
                above.drop_vars("quantile", errors="ignore"),
            ],
            dim="probability",
        )
        return prob.assign_coords(probability=["PB", "PN", "PA"]).transpose("probability", "T", "Y", "X")

    @staticmethod
    def tercile_probs_from_deterministic(
        pred: xr.DataArray,
        q33: xr.DataArray,
        q66: xr.DataArray,
    ) -> xr.DataArray:
        pb = xr.where(pred < q33, 1.0, 0.0)
        pa = xr.where(pred > q66, 1.0, 0.0)
        pn = 1.0 - pb - pa
        prob = xr.concat(
            [
                pb.drop_vars("quantile", errors="ignore"),
                pn.drop_vars("quantile", errors="ignore"),
                pa.drop_vars("quantile", errors="ignore"),
            ],
            dim="probability",
        )
        return prob.assign_coords(probability=["PB", "PN", "PA"]).transpose("probability", "T", "Y", "X")

    def forecast(
        self,
        Predictant: xr.DataArray,
        clim_year_start: int,
        clim_year_end: int,
        hindcast_det: xr.DataArray,
        hindcast_det_cross: xr.DataArray,
        Predictor_for_year: xr.DataArray,
        return_ensemble: bool = True,
        max_members: Optional[int] = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        y_mean, y_ens = self.compute_model(
            X_train=hindcast_det,
            y_train=Predictant,
            X_test=Predictor_for_year,
            return_ensemble=return_ensemble,
            max_members=max_members,
        )
        q33, q66 = self.tercile_thresholds_from_obs(Predictant, clim_year_start, clim_year_end)
        if y_ens is not None and y_ens.sizes.get("member", 0) > 0:
            prob = self.tercile_probs_from_ensemble(y_ens, q33, q66)
        else:
            prob = self.tercile_probs_from_deterministic(y_mean, q33, q66)
        return y_mean, prob


@dataclass
class WAS_RoebberComputeConfig:
    validation_fraction: float = 0.2
    min_train_samples: int = 10
    min_val_samples: int = 5
    parallel: bool = False
    backend: str = "processes"
    n_jobs: Optional[int] = None


def _point_worker_2015(args):
    Xtr, ytr, Xte, predictor_names, ep_cfg, was_cfg, point_seed, max_members = args
    split = _split_train_validation(
        Xtr,
        ytr,
        validation_fraction=was_cfg.validation_fraction,
        min_train_samples=was_cfg.min_train_samples,
        min_val_samples=was_cfg.min_val_samples,
    )
    if split is None:
        ntest = Xte.shape[0]
        return np.full(ntest, np.nan, dtype=float), np.full((1, ntest), np.nan, dtype=float)
    Xa, ya, Xv, yv = split
    cfg = Roebber2015Config(
        population_size=ep_cfg.population_size,
        n_lines=ep_cfg.n_lines,
        generations=ep_cfg.generations,
        max_mates_per_male=ep_cfg.max_mates_per_male,
        mutation_probability=ep_cfg.mutation_probability,
        transposition_probability=ep_cfg.transposition_probability,
        retain_top=ep_cfg.retain_top,
        random_state=int(point_seed),
    )
    model = Roebber2015EP(cfg)
    model.fit(SingleSiteData(list(predictor_names), Xa, ya, Xv, yv, Xte))
    mean_pred, ens = model.predict(Xte, max_members=max_members)
    return mean_pred.astype(float), ens.astype(float)


def _point_worker_2019(args):
    Xtr, ytr, Xte, predictor_names, pp_cfg, was_cfg, point_seed = args
    split = _split_train_validation(
        Xtr,
        ytr,
        validation_fraction=was_cfg.validation_fraction,
        min_train_samples=was_cfg.min_train_samples,
        min_val_samples=was_cfg.min_val_samples,
    )
    if split is None:
        ntest = Xte.shape[0]
        return np.full(ntest, np.nan, dtype=float), np.full((1, ntest), np.nan, dtype=float)
    Xa, ya, Xv, yv = split
    cfg = Roebber2019Config(
        grid_size=pp_cfg.grid_size,
        prey_capacity=pp_cfg.prey_capacity,
        predator_capacity=pp_cfg.predator_capacity,
        init_prey=pp_cfg.init_prey,
        init_pred=pp_cfg.init_pred,
        generations=pp_cfg.generations,
        l_temperature=pp_cfg.l_temperature,
        c_temperature=pp_cfg.c_temperature,
        diversity_threshold=pp_cfg.diversity_threshold,
        random_state=int(point_seed),
    )
    model = Roebber2019PredatorPreyEP(list(predictor_names), cfg)
    model.fit(SingleSiteData(list(predictor_names), Xa, ya, Xv, yv, Xte))
    mean_pred, ens = model.predict(Xte)
    return mean_pred.astype(float), ens.astype(float)


class WAS_Roebber2015EP(WAS_RoebberBase):
    def __init__(
        self,
        ep_config: Roebber2015Config = Roebber2015Config(),
        compute_config: WAS_RoebberComputeConfig = WAS_RoebberComputeConfig(),
    ):
        self.ep_config = ep_config
        self.compute_config = compute_config

    def _executor_map(self, worker, tasks):
        if not self.compute_config.parallel or len(tasks) <= 1:
            return list(map(worker, tasks))
        backend = self.compute_config.backend.lower()
        max_workers = self.compute_config.n_jobs
        if backend == "threads":
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                return list(ex.map(worker, tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(worker, tasks))

    def compute_model(
        self,
        X_train: xr.DataArray,
        y_train: xr.DataArray,
        X_test: xr.DataArray,
        return_ensemble: bool = True,
        max_members: Optional[int] = None,
    ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
        Xtr, Ytr = _ensure_xy_order(X_train, y_train)
        Xte = X_test.transpose("T", "M", "Y", "X")
        predictor_names = [str(v) for v in Xtr["M"].values.tolist()]
        Ttest, Yn, Xn = Xte.sizes["T"], Xte.sizes["Y"], Xte.sizes["X"]
        tasks = []
        points = []
        for iy in range(Yn):
            for ix in range(Xn):
                points.append((iy, ix))
                point_seed = int(self.ep_config.random_state + iy * Xn + ix)
                tasks.append((
                    Xtr.isel(Y=iy, X=ix).values,
                    Ytr.isel(Y=iy, X=ix).values,
                    Xte.isel(Y=iy, X=ix).values,
                    predictor_names,
                    self.ep_config,
                    self.compute_config,
                    point_seed,
                    max_members,
                ))
        results = self._executor_map(_point_worker_2015, tasks)
        mean_out = np.full((Ttest, Yn, Xn), np.nan, dtype=float)
        member_arrays = []
        maxm = 0
        for mean_pred, ens in results:
            maxm = max(maxm, 0 if ens is None else ens.shape[0])
            member_arrays.append(ens)
        ens_out = None
        if return_ensemble:
            ens_out = np.full((maxm, Ttest, Yn, Xn), np.nan, dtype=float)
        for (iy, ix), (mean_pred, ens) in zip(points, results):
            mean_out[:, iy, ix] = mean_pred
            if return_ensemble and ens_out is not None and ens is not None:
                ens_out[: ens.shape[0], :, iy, ix] = ens
        y_mean = xr.DataArray(
            mean_out,
            coords={"T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]},
            dims=("T", "Y", "X"),
            name="roebber2015_mean",
        )
        if not return_ensemble:
            return y_mean, None
        y_ens = xr.DataArray(
            ens_out,
            coords={"member": np.arange(maxm), "T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]},
            dims=("member", "T", "Y", "X"),
            name="roebber2015_ens",
        )
        return y_mean, y_ens


class WAS_Roebber2018SpatialEP(WAS_RoebberBase):
    def __init__(
        self,
        ep_config: Roebber2018Config = Roebber2018Config(),
        compute_config: WAS_RoebberComputeConfig = WAS_RoebberComputeConfig(parallel=False),
    ):
        self.ep_config = ep_config
        self.compute_config = compute_config
        self.model_: Optional[Roebber2018SpatialEP] = None

    def compute_model(
        self,
        X_train: xr.DataArray,
        y_train: xr.DataArray,
        X_test: xr.DataArray,
        return_ensemble: bool = True,
        max_members: Optional[int] = None,
    ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
        Xtr, Ytr = _ensure_xy_order(X_train, y_train)
        Xte = X_test.transpose("T", "M", "Y", "X")
        n_time = Xtr.sizes["T"]
        n_val = max(self.compute_config.min_val_samples, int(np.ceil(self.compute_config.validation_fraction * n_time)))
        n_val = min(n_val, n_time - self.compute_config.min_train_samples)
        if n_val < self.compute_config.min_val_samples:
            raise ValueError("Not enough training samples for Roebber2018SpatialEP validation split")
        split = n_time - n_val
        model = Roebber2018SpatialEP(
            Roebber2018Config(
                population_size=self.ep_config.population_size,
                eliminate=self.ep_config.eliminate,
                generations=self.ep_config.generations,
                n_lines=self.ep_config.n_lines,
                random_state=self.ep_config.random_state,
            )
        )
        model.fit(
            Xtr.isel(T=slice(0, split)),
            Ytr.isel(T=slice(0, split)),
            Xtr.isel(T=slice(split, None)),
            Ytr.isel(T=slice(split, None)),
        )
        self.model_ = model
        y_mean = model.predict(Xte)
        y_mean.name = "roebber2018_mean"
        if not return_ensemble:
            return y_mean, None
        y_ens = y_mean.expand_dims(member=[0]).transpose("member", "T", "Y", "X")
        y_ens.name = "roebber2018_ens"
        return y_mean, y_ens


class WAS_Roebber2019PredatorPreyEP(WAS_RoebberBase):
    def __init__(
        self,
        ep_config: Roebber2019Config = Roebber2019Config(),
        compute_config: WAS_RoebberComputeConfig = WAS_RoebberComputeConfig(),
    ):
        self.ep_config = ep_config
        self.compute_config = compute_config

    def _executor_map(self, worker, tasks):
        if not self.compute_config.parallel or len(tasks) <= 1:
            return list(map(worker, tasks))
        backend = self.compute_config.backend.lower()
        max_workers = self.compute_config.n_jobs
        if backend == "threads":
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                return list(ex.map(worker, tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(worker, tasks))

    def compute_model(
        self,
        X_train: xr.DataArray,
        y_train: xr.DataArray,
        X_test: xr.DataArray,
        return_ensemble: bool = True,
        max_members: Optional[int] = None,
    ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
        Xtr, Ytr = _ensure_xy_order(X_train, y_train)
        Xte = X_test.transpose("T", "M", "Y", "X")
        predictor_names = [str(v) for v in Xtr["M"].values.tolist()]
        Ttest, Yn, Xn = Xte.sizes["T"], Xte.sizes["Y"], Xte.sizes["X"]
        tasks = []
        points = []
        for iy in range(Yn):
            for ix in range(Xn):
                points.append((iy, ix))
                point_seed = int(self.ep_config.random_state + iy * Xn + ix)
                tasks.append((
                    Xtr.isel(Y=iy, X=ix).values,
                    Ytr.isel(Y=iy, X=ix).values,
                    Xte.isel(Y=iy, X=ix).values,
                    predictor_names,
                    self.ep_config,
                    self.compute_config,
                    point_seed,
                ))
        results = self._executor_map(_point_worker_2019, tasks)
        mean_out = np.full((Ttest, Yn, Xn), np.nan, dtype=float)
        member_arrays = []
        maxm = 0
        for mean_pred, ens in results:
            maxm = max(maxm, 0 if ens is None else ens.shape[0])
            member_arrays.append(ens)
        ens_out = None
        if return_ensemble:
            ens_out = np.full((maxm, Ttest, Yn, Xn), np.nan, dtype=float)
        for (iy, ix), (mean_pred, ens) in zip(points, results):
            mean_out[:, iy, ix] = mean_pred
            if return_ensemble and ens_out is not None and ens is not None:
                ens_out[: ens.shape[0], :, iy, ix] = ens
        y_mean = xr.DataArray(
            mean_out,
            coords={"T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]},
            dims=("T", "Y", "X"),
            name="roebber2019_mean",
        )
        if not return_ensemble:
            return y_mean, None
        y_ens = xr.DataArray(
            ens_out,
            coords={"member": np.arange(maxm), "T": Xte["T"], "Y": Xte["Y"], "X": Xte["X"]},
            dims=("member", "T", "Y", "X"),
            name="roebber2019_ens",
        )
        return y_mean, y_ens


__all__ += [
    "WAS_RoebberBase",
    "WAS_RoebberComputeConfig",
    "WAS_Roebber2015EP",
    "WAS_Roebber2018SpatialEP",
    "WAS_Roebber2019PredatorPreyEP",
]
