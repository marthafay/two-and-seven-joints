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
marthaelias [at] protonmail [dot] com
"""
# seven_joints_live.py
# research only. Apache 2.0. Keine Gewähr.

import time
import numpy as np

import matplotlib
matplotlib.use("MacOSX")  
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from collections import deque
from two_joints import compute_xi  
from matplotlib import animation
import os

EXPORT = True              # ⬅️ Toggle für Export
EXPORT_NAME = "xi_robotics_stream.mp4"
EXPORT_FPS = 30
EXPORT_FRAMES = []

# ------------------------
# Parameter
# ------------------------
J = 7
W = 256
step = 16
fs = 100
dt = 1.0 / fs
tmax = 10.0

# ------------------------
# Buffer anlegen
# ------------------------
names = ["qd", "qdd", "tau_cmd", "tau_meas"]
buffer = {k: deque(maxlen=W) for k in names}
results = {"t": [], "absXi": [], "policy": [], "action": []}

# ------------------------
# Synthese-Funktion: 7 Gelenke mit leicht unterschiedlicher Phase
# ------------------------
def synth_sample(t):
    f_base = 0.8
    qd = np.array([np.sin(2*np.pi*(f_base+0.05*j)*t + 0.3*j) for j in range(J)])
    qdd = np.array([2*np.pi*(f_base+0.05*j)*np.cos(2*np.pi*(f_base+0.05*j)*t + 0.3*j) for j in range(J)])
    tau_cmd = np.array([0.9*qd[j] + 0.05*np.random.randn() for j in range(J)])
    tau_meas = tau_cmd + 0.04*np.sin(2*np.pi*2*t) + 0.02*np.random.randn(J)
    # kleine Störung
    if 4.0 < t < 5.0:
        tau_meas *= -1.0
    return dict(qd=qd, qdd=qdd, tau_cmd=tau_cmd, tau_meas=tau_meas)

def update_buffers(sample):
    for k in buffer:
        buffer[k].append(sample[k])

def process_if_ready(t_now):
    if len(buffer["qd"]) < W:
        return None
    arrs = {k: np.stack(buffer[k], axis=1) for k in buffer}  # shape (J,W)

    # Für alte compute_xi-Version: Mitteln über alle Gelenke
    qd1, qd2 = arrs["qd"][0], arrs["qd"][1]
    qdd1, qdd2 = arrs["qdd"][0], arrs["qdd"][1]
    u = arrs["tau_cmd"].mean(axis=0)
    tau_meas = arrs["tau_meas"].mean(axis=0)

    res = compute_xi(qd1, qd2, qdd1, qdd2, u, tau_meas,
                     fs=fs, window=W, step=step)
    p = res["probs"][-1]
    a = res["action"][-1]
    mag = res["absXi"][-1]
    results["t"].append(t_now)
    results["absXi"].append(mag)
    results["policy"].append(p)
    results["action"].append(a)
    return mag, p, a

# ------------------------
# Plot vorbereiten
# ------------------------
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
(line_mag,) = axes[0].plot([], [], color="#1f77b4")
axes[0].set_ylabel("|Ξ|")
axes[0].grid(alpha=0.3)

labels = ["execute", "guard", "stop", "pivot"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
lines_policy = [axes[1].plot([], [], label=lab, color=c)[0] for lab, c in zip(labels, colors)]
axes[1].legend(loc="upper right")
axes[1].set_ylabel("Policy p")
axes[1].grid(alpha=0.3)

mapping = dict(execute=0, guard=1, stop=2, pivot=3)
(line_act,) = axes[2].step([], [], where="post", color="#333")
axes[2].set_yticks(range(4))
axes[2].set_yticklabels(labels)
axes[2].set_xlabel("Zeit (s)")
axes[2].set_title("Dominante Aktion")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------
# Simulation
# ------------------------
t = 0.0
while t < tmax:
    s = synth_sample(t)
    update_buffers(s)
    res = process_if_ready(t)
    if res:
        mag, p, a = res
        line_mag.set_data(results["t"], results["absXi"])
        for i in range(4):
            y = [pi[i] for pi in results["policy"]]
            lines_policy[i].set_data(results["t"], y)
        y_act = [mapping[x] for x in results["action"]]
        line_act.set_data(results["t"], y_act)
        tmax_dyn = max(5.0, t)
        axes[0].set_xlim(max(0, tmax_dyn-5), tmax_dyn)
        axes[0].set_ylim(0, max(0.4, np.nanmax(results["absXi"])*1.2))
        axes[1].set_xlim(max(0, tmax_dyn-5), tmax_dyn)
        axes[1].set_ylim(0, 1)
        axes[2].set_xlim(max(0, tmax_dyn-5), tmax_dyn)
        plt.pause(0.01)


        if EXPORT:
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            image = buf[:, :, :3]          # RGB, Alpha-Kanal verwerfen
            EXPORT_FRAMES.append(image.copy())
            
    t += dt
plt.ioff()
plt.show()

if EXPORT and EXPORT_FRAMES:
    print("🎞 Rendering export video...")

    h, w, _ = EXPORT_FRAMES[0].shape
    dpi = 100
    fig_export = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.axes([0,0,1,1])
    ax.set_axis_off()
    im = ax.imshow(EXPORT_FRAMES[0])

    def update_export(i):
        im.set_array(EXPORT_FRAMES[i])
        return [im]

    ani = animation.FuncAnimation(fig_export, update_export, frames=len(EXPORT_FRAMES), blit=True)
    
    writer = animation.FFMpegWriter(fps=EXPORT_FPS)
    ani.save(EXPORT_NAME, writer=writer)
    plt.close(fig_export)

    print(f"✅ Exported: {EXPORT_NAME} ({len(EXPORT_FRAMES)} frames at {EXPORT_FPS} fps)")