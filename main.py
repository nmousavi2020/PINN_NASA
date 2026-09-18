# =============================================================================
#
# Copyright (c) 2026 N. Mousavi
# All rights reserved.
#
# Description:
#     Physics-Informed Neural Network (PINN) workflow for bearing degradation
#     analysis using NASA bearing vibration data.
#
# Dataset:
#     NASA IMS Bearing Dataset
#
# Input:
#     sample/1st_test/
#
# Output:
#     plots/pinn_degradation.png
#
# =============================================================================

print()
print("╭───────────────────────────────────────────────────────────────╮")
print("│                    NASA Bearing — PINN                        │")
print("│                                                               │")
print("│        Physics-Informed Neural Network for Bearing            │")
print("│                  Degradation Analysis                         │")
print("│                                                               │")
print("│              Copyright © 2026 N. Mousavi                      │")
print("╰───────────────────────────────────────────────────────────────╯")

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================
# 1. Load NASA Bearing Data
# =========================

DATA_DIR = "sample/1st_test"

files = sorted(os.listdir(DATA_DIR))

time = []
rms = []

for i, filename in enumerate(files):
    path = os.path.join(DATA_DIR, filename)

    try:
        data = np.loadtxt(path)

        # NASA files contain multiple vibration channels
        signal = data[:, 0]

        # RMS feature
        value = np.sqrt(np.mean(signal ** 2))

        time.append(i)
        rms.append(value)

    except Exception:
        continue

time = np.array(time, dtype=np.float32)
rms = np.array(rms, dtype=np.float32)

# Normalize
t_min, t_max = time.min(), time.max()
y_min, y_max = rms.min(), rms.max()

t = (time - t_min) / (t_max - t_min)
y = (rms - y_min) / (y_max - y_min)

t_train = torch.tensor(t).reshape(-1, 1)
y_train = torch.tensor(y).reshape(-1, 1)


# =========================
# 2. PINN Model
# =========================

class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)


model = PINN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================
# 3. PINN Training
# =========================

for epoch in range(5000):

    t_physics = t_train.clone().requires_grad_(True)

    prediction = model(t_physics)

    # Data loss
    loss_data = torch.mean(
        (prediction - y_train) ** 2
    )

    # Physics loss:
    # degradation should generally increase
    dy_dt = torch.autograd.grad(
        prediction,
        t_physics,
        torch.ones_like(prediction),
        create_graph=True
    )[0]

    # Penalize negative degradation rate
    loss_physics = torch.mean(
        torch.relu(-dy_dt) ** 2
    )

    loss = loss_data + 0.1 * loss_physics

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss.item():.6f}"
        )


# =========================
# 4. Prediction
# =========================

with torch.no_grad():
    prediction = model(t_train).numpy().flatten()

# Convert back to original scale
prediction = prediction * (y_max - y_min) + y_min


# =========================
# 5. Plot
# =========================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    rms,
    label="Measured RMS",
    color="blue"
)

plt.plot(
    time,
    prediction,
    label="PINN Prediction",
    color="red",
    linewidth=2
)

plt.xlabel("Time / Measurement Index")
plt.ylabel("RMS Vibration")

plt.title("NASA Bearing Degradation using PINN")

plt.legend()
plt.grid(True)

os.makedirs("plots", exist_ok=True)

plt.savefig(
    "plots/pinn_degradation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
