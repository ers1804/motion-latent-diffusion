#!/usr/bin/env python3
"""Statistical comparison of eval runs from their logs (item 8b, review_tracker.md).

test.py prints one metrics table per replication; the saved JSON keeps only
aggregates, so per-replication values are parsed from the log text.

Usage:
  python research/src/stats_tests.py <log_A> <log_B> [--metric Metrics/FID]
Compares metric distributions across replications: Welch's t-test (unequal
variance) + bootstrap CI of the difference (A - B). n is small (3), so treat
p-values as indicative; the bootstrap CI is the primary report.
"""
import argparse
import math
import random
import re
import sys

TABLE_ROW = re.compile(r"(Metrics/[A-Za-z_0-9]+)\s*[│|]\s*([0-9.]+)")
REP_MARK = re.compile(r"Evaluating (?:EgoMotionMetrics|MultiModality) - Replication (\d+)")


def parse_replications(log_path):
    """Return {metric: [per-replication values]} from a test.py log.

    Each replication prints a metrics table after its 'Replication k' marker.
    The final aggregate table (Metrics/FID/mean etc.) is excluded by keying on
    plain metric names (no /mean, /conf_interval suffixes).
    """
    text = open(log_path, errors="ignore").read()
    # Split log at replication markers; the table for rep k follows marker k.
    parts = REP_MARK.split(text)
    # parts = [pre, rep_idx, chunk, rep_idx, chunk, ...]; both the
    # EgoMotionMetrics and MultiModality markers match, so a replication can
    # contribute two chunks — the dict merge below handles that.
    per_rep = {}
    for i in range(1, len(parts) - 1, 2):
        rep = int(parts[i])
        chunk = parts[i + 1]
        for m, v in TABLE_ROW.findall(chunk):
            if "/" in m.replace("Metrics/", ""):  # skip /mean, /conf_interval rows
                continue
            per_rep.setdefault(m, {})[rep] = float(v)
    # keep metrics with >=2 reps, ordered by rep index
    out = {}
    for m, d in per_rep.items():
        vals = [d[k] for k in sorted(d)]
        if len(vals) >= 2:
            out[m] = vals
    return out


def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf") if ma != mb else 0.0, 0.0, ma - mb
    t = (ma - mb) / se
    # Welch–Satterthwaite dof
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    return t, dof, ma - mb


def t_sf(t, dof):
    """Two-sided p-value via incomplete beta (no scipy dependency)."""
    x = dof / (dof + t * t)
    # regularized incomplete beta I_x(dof/2, 1/2) using continued fraction
    a, b = dof / 2.0, 0.5

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-9, 1e-30
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x <= 0:
        ib = 0.0
    elif x >= 1:
        ib = 1.0
    else:
        bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
        if x < (a + 1.0) / (a + b + 2.0):
            ib = bt * betacf(a, b, x) / a
        else:
            ib = 1.0 - bt * betacf(b, a, 1.0 - x) / b
    return ib  # == P(|T|>t) for symmetric t


def bootstrap_diff(a, b, iters=20000, seed=0):
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--metric", default=None, help="e.g. Metrics/FID (default: all shared)")
    args = ap.parse_args()
    A, B = parse_replications(args.log_a), parse_replications(args.log_b)
    metrics = [args.metric] if args.metric else sorted(set(A) & set(B))
    if not metrics:
        sys.exit("no shared per-replication metrics found")
    print(f"A: {args.log_a} | B: {args.log_b}")
    for m in metrics:
        if m not in A or m not in B:
            print(f"{m}: missing in one log"); continue
        a, b = A[m], B[m]
        t, dof, diff = welch_t(a, b)
        p = t_sf(abs(t), dof)
        lo, hi = bootstrap_diff(a, b)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"{m:34s} A={sum(a)/len(a):7.3f} (n={len(a)})  B={sum(b)/len(b):7.3f} (n={len(b)})  "
              f"diff={diff:+7.3f}  Welch p={p:.3f}  boot95%[{lo:+.3f},{hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
