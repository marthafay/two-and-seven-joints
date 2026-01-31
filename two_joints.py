#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Martha Elias

"""
CAUTION
Deterministic modeling is vulnerable to unnatural distortions and algorithmically triggered
reactions. Independent safety and risk management strategies are essential.

DISCLAIMER (Research Only)
This repository contains a research prototype. It is provided for educational and research purposes
only. It does NOT constitute financial, investment, legal, medical, or any other professional
advice. No warranty is given. Use at your own risk. Before using any outputs to inform real-world
decisions, obtain advice from qualified professionals and perform independent verification.

Copyright (c) 2025 Martha Elias
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and limitations under
the License.

ADDITIONAL RESEARCH DISTRIBUTION CLAUSE
---------------------------------------
This software is distributed as part of an independent research project.
The author (Martha Elias) retains full authorship and reserves the unrestricted right to:
 • publish, archive, or make public updated or extended versions of this work in the future,
 • integrate parts of this work into future academic or open-source releases,
   even if earlier versions were distributed commercially.

Purchasers and licensees may freely use this work — including for commercial or applied purposes —
provided that proper attribution to the original author is maintained and the license terms are respected.
No exclusivity is granted to any buyer or institution. All versions remain part of the same continuous
research line and may be openly released by the author at any time.

Author: Elias, Martha
Version: v1.0 (October 2025)
I want ro get hired! Contact: marthaelias [at] protonmail [dot] com
"""
"""
two_joints.py — simplified Ξ demo with 2 joints
Re{Ξ}: evidence (coherence + reactivity − lag/drop)
Im{Ξ}: directed coupling (Odd × Chi + IAI)
Projection → execute / guard / stop / pivot

Dependencies: numpy (required), matplotlib (optional for --plot)
"""

import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

EPS = 1e-12
ACTIONS = ["execute","guard","stop","pivot"]
ANGLES  = np.deg2rad([0, 90, 180, 270])

# ----------------- Utilities -----------------

def hann(n: int) -> np.ndarray:
    if n < 8: return np.ones(n, float)
    t = np.linspace(0, np.pi, n)
    return np.clip(0.5*(1 - np.cos(t))*2.0, 0.05, 1.0)

def analytic(x: np.ndarray) -> np.ndarray:
    """FFT-Hilbert (nicht kausal, Fenster-basiert)"""
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.fft(x)
    H = np.zeros(n)
    if n % 2 == 0:
        H[0] = H[n//2] = 1.0
        H[1:n//2] = 2.0
    else:
        H[0] = 1.0
        H[1:(n+1)//2] = 2.0
    return np.fft.ifft(X * H)

def softmax(vals, T=0.6):
    v = np.asarray(vals, float) / max(T, 1e-6)
    v -= v.max()
    e = np.exp(v)
    return (e / (e.sum() + EPS)).tolist()

def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    wm = w.sum() + EPS
    mx = (w @ x) / wm; my = (w @ y) / wm
    sx = np.sqrt(((w*(x-mx)**2).sum())/wm + EPS)
    sy = np.sqrt(((w*(y-my)**2).sum())/wm + EPS)
    return float(np.clip(((w*(x-mx)*(y-my)).sum())/(wm*sx*sy + EPS), -1, 1))

# ----------------- Features -----------------

def plv_from_series(x: np.ndarray) -> float:
    """PLV einer reellen Serie via instant. Phase"""
    z = analytic(x)
    return float(np.abs(np.mean(np.exp(1j*np.angle(z)))))

def plv_pair(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    zu, zv = analytic(u), analytic(v)
    dphi = np.angle(zu * np.conj(zv))
    wn = w / (w.sum() + EPS)
    return float(np.clip(np.abs(np.sum(wn * np.exp(1j*dphi))), 0, 1))

def iai_pair(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    zu, zv = analytic(u), analytic(v)
    dphi = np.angle(zu * np.conj(zv))
    wn = w / (w.sum() + EPS)
    return float(np.clip(np.sum(wn * np.sin(dphi)), -1, 1))

def odd_chiral(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float,float,float]:
    xr = x[::-1]
    x_even = 0.5*(x + xr); x_odd = 0.5*(x - xr)
    E_even = float((w*(x_even**2)).sum())
    E_odd  = float((w*(x_odd**2)).sum())
    Odd = float(np.clip(E_odd / max(E_even + E_odd + EPS, EPS), 0, 1))
    dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0])
    num = float((w*(x*dy - y*dx)).sum())
    den = float((w*(x*x + y*y)).sum() + EPS)
    Chi = float(np.clip(num / den, -1, 1))
    return Odd, Chi, float(Odd*Chi)

# ----------------- Synthese (2 Gelenke) -----------------

def synth_2joints(fs=100, duration=30, seed=7):
    rng = np.random.default_rng(seed)
    n = int(fs*duration); t = np.arange(n)/fs
    f1, f2 = 0.7, 0.9
    # qd: Velocities, qdd: Accelerations
    qd1 = 0.7*np.sin(2*np.pi*f1*t + 0.2) + 0.08*rng.standard_normal(n)
    qd2 = 0.6*np.sin(2*np.pi*f2*t + 0.6) + 0.08*rng.standard_normal(n)
    qdd1 = np.gradient(qd1, 1/fs); qdd2 = np.gradient(qd2, 1/fs)
    # u: command torque (aggregiert), tau_meas: measured torque (mit Lag/Friktion)
    u = 0.9*np.sin(2*np.pi*0.8*t + 0.3) + 0.10*rng.standard_normal(n)
    tau_meas = np.zeros(n)
    alpha = 0.94
    for k in range(1, n):
        tau_meas[k] = alpha*tau_meas[k-1] + (1-alpha)*u[k] + 0.05*np.sign(qd1[k]+qd2[k]) \
                      + 0.06*rng.standard_normal()
    return t, qd1, qd2, qdd1, qdd2, u, tau_meas

# ----------------- Ξ-Berechnung -----------------

def compute_xi(qd1, qd2, qdd1, qdd2, u, tau_meas, fs=100, window=256, step=16,
               temp=0.6, kappa=1.0):
    n = len(u)
    W = int(window); S = int(step)
    w = hann(W)
    centers = []
    absXi, angXi, probs, actions = [], [], [], []
    # Diagnose
    PLV_v, PLV_uy, IAI_cs, rho_ud, lag_idx = [], [], [], [], []
    # voraggregieren
    ddqS = 0.5*(qdd1 + qdd2)

    # naive Lag-Schätzer (klein, schnell)
    def estimate_lag(a, b, maxlag):
        best, idx = 0.0, 0
        for L in range(-maxlag, maxlag+1):
            if L < 0:
                aa, bb = a[:L], b[-L:]
            elif L > 0:
                aa, bb = a[L:], b[:-L]
            else:
                aa, bb = a, b
            if len(aa) < 16: continue
            c = np.dot(aa-np.mean(aa), bb-np.mean(bb)) / ((np.linalg.norm(aa)+EPS)*(np.linalg.norm(bb)+EPS))
            if abs(c) > abs(best): best, idx = c, L
        return idx, best

    maxlag = int(0.12*fs)  # ~120ms
    for s in range(0, n - W + 1, S):
        sl = slice(s, s+W)
        # Re-Features
        plv_v = 0.5*(plv_from_series(qd1[sl]) + plv_from_series(qd2[sl]))
        rho   = abs(weighted_corr(u[sl], ddqS[sl], w))
        # Kopplung/Orientierung
        plv_uy = plv_pair(u[sl], tau_meas[sl], w)
        iai    = iai_pair(u[sl], tau_meas[sl], w)
        _, _, P = odd_chiral(u[sl], tau_meas[sl], w)
        # Lag
        Ls, _ = estimate_lag(u[sl], tau_meas[sl], maxlag=maxlag)
        Lidx = np.clip(abs(Ls)/max(1, maxlag), 0.0, 1.0)

        # Re{Ξ} und Im{Ξ} (einfaches Linearmodell)
        ReXi = 0.45*plv_v + 0.35*rho - 0.25*Lidx
        ImXi = 0.65*P + 0.35*iai
        Xi   = ReXi + 1j*ImXi

        # Softmax-Projektion auf Achsen (selbstvertrauensabhängige Temperatur)
        mag = float(np.abs(Xi)); ang = float(np.angle(Xi))
        T_eff = float(np.clip(temp * (1 - 0.4*(mag - 0.5)), 0.35, 1.25))
        vals = [mag*np.cos(ang - th) for th in ANGLES]
        p = softmax(vals, T=T_eff)
        a = ACTIONS[int(np.argmax(p))]

        centers.append(s + W//2)
        absXi.append(mag); angXi.append(ang); probs.append(p); actions.append(a)
        PLV_v.append(plv_v); PLV_uy.append(plv_uy); IAI_cs.append(iai); rho_ud.append(rho); lag_idx.append(Lidx)

    out = dict(
        centers=np.asarray(centers,int),
        absXi=np.asarray(absXi,float),
        angXi=np.asarray(angXi,float),
        probs=np.asarray(probs,float),
        action=np.array(actions,object),
        PLV_v=np.asarray(PLV_v,float),
        PLV_uy=np.asarray(PLV_uy,float),
        IAI_cs=np.asarray(IAI_cs,float),
        rho_ud=np.asarray(rho_ud,float),
        lag_idx=np.asarray(lag_idx,float),
    )
    return out

# ----------------- Szenarien -----------------

def scenario_orientation_flip(tau_meas, start, end):
    tau_meas[start:end] *= -1.0

def scenario_latency(u, lag_samples):
    if lag_samples <= 0: return
    u[lag_samples:] = u[:-lag_samples]
    u[:lag_samples] = 0.0

# ----------------- Plot -----------------

def plot_results(t, res, fs):
    if plt is None:
        print("Matplotlib fehlt: ohne Plot weiter 🤷")
        return
    P = res["probs"]; centers = res["centers"]/fs
    fig, axs = plt.subplots(4,1,figsize=(11,8), sharex=True)
    axs[0].plot(centers, res["absXi"], label="|Ξ|"); axs[0].grid(alpha=0.3); axs[0].legend()
    axs[1].plot(centers, res["PLV_v"], label="PLV_v")
    axs[1].plot(centers, res["PLV_uy"], label="PLV(u,τΣ)")
    axs[1].plot(centers, res["rho_ud"], label="|ρ(u,ddqΣ)|")
    axs[1].plot(centers, res["lag_idx"], label="LagIdx")
    axs[1].legend(); axs[1].grid(alpha=0.3)
    for i, lab in enumerate(ACTIONS):
        axs[2].plot(centers, P[:,i], label=lab)
    axs[2].set_ylim(-0.02,1.02); axs[2].legend(); axs[2].grid(alpha=0.3); axs[2].set_ylabel("Policy p")
    # Aktion als Index
    act_idx = np.array([ACTIONS.index(a) for a in res["action"]])
    axs[3].plot(centers, act_idx, drawstyle="steps-post"); axs[3].set_yticks(range(4)); axs[3].set_yticklabels(ACTIONS)
    axs[3].grid(alpha=0.3); axs[3].set_xlabel("Zeit (s)")
    plt.tight_layout(); plt.show()

# ----------------- CLI -----------------

def run_once(fs=100, duration=30, window=256, step=16, plot=False, scenario=None):
    t, qd1, qd2, qdd1, qdd2, u, tau_meas = synth_2joints(fs=fs, duration=duration)
    # Eventfenster
    n = len(t); ev_s = int(0.55*n); ev_e = int(0.70*n)

    if scenario == "flip":
        scenario_orientation_flip(tau_meas, ev_s, ev_e)
    elif scenario == "lag":
        lag_samp = max(1, int(0.05*fs))  # ~50ms
        scenario_latency(u, lag_samp)

    res = compute_xi(qd1, qd2, qdd1, qdd2, u, tau_meas, fs=fs, window=window, step=step)
    # Mini-Report
    hist = {}
    for a in res["action"]:
        hist[a] = hist.get(a,0) + 1
    print(f"Frames={len(res['centers'])} | mean|Ξ|={res['absXi'].mean():.3f} | actions={hist}")

    if plot:
        plot_results(t, res, fs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--scenario", type=str, default="", choices=["","flip","lag"],
                    help="''=Baseline, 'flip'=Orientierungsflip, 'lag'=Latency/Desync")
    args = ap.parse_args()
    run_once(fs=args.fs, duration=args.duration, window=args.window, step=args.step,
             plot=args.plot, scenario=args.scenario)

if __name__ == "__main__":
    main()