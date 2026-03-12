"""
Noise-Simulation for SNA-QDL Protocol.

This script implements a rudimentary noise simulation for the SNA-QDL protocol
using Qiskit. The purpose of this program is to evaluate the robustness of the
protocol against noise in a simulated environment. The simulation records:

- Fidelity of the recovered message
- Distribution of errors via histograms
- Hamming distances and error heatmaps per bit and qudit

The simulation is executed using Qiskit Aer with a noise model derived from a
generic backend, providing insight into the behavior of the protocol under
hardware-like noise conditions. Results are printed to the console and visualized
through plots for further analysis.

Dependencies: may be installed via requirements.txt
Authors: Yauset Cabrera-Aparicio, Jorge García-Díaz, Pino Caballero-Gil (2026)
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library.standard_gates import TGate, SGate, ZGate

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
n = 5
d = 8
kappa = 2
total_qubits = 3 * n
shots = 4000

msg = np.random.randint(0, d, size=n)
key = np.random.randint(0, d, size=kappa)

# Example of an expansion matrix F, up to n = 8
F = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [2, 1]
]) % d

theta = np.dot(F, key) % d

print(f"Original Message: {msg}")
print(f"Secret Key: {key}")

phase_dict = {
    0: [],
    1: [TGate()],
    2: [SGate()],
    3: [SGate(), TGate()],
    4: [ZGate()],
    5: [ZGate(), TGate()],
    6: [ZGate(), SGate()],
    7: [ZGate(), SGate(), TGate()]
}

def apply_ftqc_phase(qc, qubit, k):
    for gate in phase_dict[k % 8]:
        qc.append(gate, [qubit])

def apply_qudit_phase(qc, start, theta, inverse=False):
    s = -1 if inverse else 1
    apply_ftqc_phase(qc, start,     s * theta)
    apply_ftqc_phase(qc, start + 1, s * 2 * theta)
    apply_ftqc_phase(qc, start + 2, s * 4 * theta)

qc = QuantumCircuit(total_qubits, total_qubits)

qft3  = QFT(3, do_swaps=True).to_gate()
iqft3 = QFT(3, do_swaps=True, inverse=True).to_gate()

# ALICE
for j in range(n):
    q = 3 * j
    for i, bit in enumerate(reversed(format(msg[j], "03b"))):
        if bit == "1":
            qc.x(q + i)
    qc.append(qft3, [q, q+1, q+2])
    apply_qudit_phase(qc, q, theta[j])

qc.barrier()

# BOB
for j in range(n):
    q = 3 * j
    apply_qudit_phase(qc, q, theta[j], inverse=True)
    qc.append(iqft3, [q, q+1, q+2])

qc.measure(range(total_qubits), range(total_qubits))

# Simulation
fake_backend = GenericBackendV2(num_qubits=total_qubits)
noise_model = NoiseModel.from_backend(fake_backend)

simulator = AerSimulator(
    method="statevector",
    noise_model=noise_model
)

compiled = transpile(qc, simulator, optimization_level=2)
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

expected_bits = ''.join(format(x, '03b') for x in reversed(msg))
success = counts.get(expected_bits, 0)
fidelity = success / shots

print("\n--- Results with simulated noise ---")
print(f"Expected: {expected_bits}")
print(f"Success: {success}/{shots}")
print(f"Fidelity: {100*fidelity:.2f}%")


# Visualization

def plot_sn_qdl_results(counts, expected_bits, n_top=10):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    BG        = "#F7F9FC"
    PANEL_BG  = "#FFFFFF"
    GRID_COL  = "#DDE3ED"
    TEXT_DARK = "#1A2340"
    TEXT_MID  = "#4A5568"
    SPINE_COL = "#CBD5E1"

    BAR_CORRECT = "#1E3A8A"
    BAR_ERROR   = "#93C5FD"

    rc = {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": SPINE_COL,
        "axes.labelcolor": TEXT_DARK,
        "axes.titlecolor": TEXT_DARK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_MID,
        "ytick.color": TEXT_MID,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
        "grid.color": GRID_COL,
        "grid.linewidth": 0.8,
        "legend.framealpha": 0.95,
        "legend.edgecolor": SPINE_COL,
        "legend.labelcolor": TEXT_DARK,
        "legend.fontsize": 10,
    }
    plt.rcParams.update(rc)

    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    top_labels = list(sorted_counts.keys())[:n_top]
    top_values = [v / shots for v in list(sorted_counts.values())[:n_top]]

    if expected_bits not in top_labels:
        top_labels.append(expected_bits)
        top_values.append(counts.get(expected_bits, 0) / shots)

    def _header(fig, title, subtitle, left=0.10, right=0.95):
        fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT_DARK, y=0.97)
        fig.text(0.5, 0.905, subtitle, ha="center", fontsize=10, color=TEXT_MID)
        fig.add_artist(plt.Line2D(
            [left, right], [0.888, 0.888],
            transform=fig.transFigure,
            color=SPINE_COL, linewidth=1.0
        ))


    # Main histogram
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.22)

    colors = [BAR_CORRECT if l == expected_bits else BAR_ERROR for l in top_labels]
    bars = ax.bar(
        top_labels, top_values,
        color=colors, edgecolor=SPINE_COL,
        linewidth=0.8, width=0.6
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.003,
            f"{h*100:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=TEXT_MID
        )

    ax.set_xlabel("Recovered Message (Bit version)", fontsize=11, labelpad=8)
    ax.set_ylabel("Probability", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(top_values) + 0.12)
    ax.tick_params(length=0, pad=5)
    ax.set_xticklabels(top_labels, rotation=45, ha="right", family="monospace", fontsize=7.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    legend_elements = [
        Patch(facecolor=BAR_CORRECT, edgecolor=SPINE_COL, label="Correct Message"),
        Patch(facecolor=BAR_ERROR,   edgecolor=SPINE_COL, label="Wrongful Communications")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    _header(fig,
            "Probabilistic Distribution of Errors with Simulated Noise for SNA-QDL",
            f"$n={n}$,   $d={d}$,   shots $={shots}$")

    plt.savefig("principal_hist.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.show()


    # Hamming distance
    ham_dist = {}
    for bitstr, cnt in counts.items():
        d_h = sum(a != b for a, b in zip(bitstr, expected_bits))
        ham_dist[d_h] = ham_dist.get(d_h, 0) + cnt

    max_d = max(ham_dist.keys())
    hd_x = list(range(max_d + 1))
    hd_y = [ham_dist.get(x, 0) / shots for x in hd_x]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.14)

    bar_colors = [
        BAR_CORRECT if x == 0 else plt.cm.Blues_r(0.15 + 0.65 * x / max_d)
        for x in hd_x
    ]

    bars = ax.bar(
        hd_x, hd_y,
        color=bar_colors,
        edgecolor=SPINE_COL,
        linewidth=0.8,
        width=0.6
    )

    for bar, val in zip(bars, hd_y):
        if val > 0.002:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + 0.003,
                f"{val*100:.1f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=TEXT_MID
            )

    ax.set_xlabel("Hamming Distance w.r.t correct message", fontsize=11, labelpad=8)
    ax.set_ylabel("Probability", fontsize=11, labelpad=8)
    ax.set_xticks(hd_x)
    ax.set_ylim(0, max(hd_y) + 0.10)
    ax.tick_params(length=0, pad=5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    legend_elements = [
        Patch(facecolor=BAR_CORRECT, edgecolor=SPINE_COL, label="With no error (d = 0)"),
        Patch(facecolor="#60A5FA", edgecolor=SPINE_COL, label="With errors (d ≥ 1)")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    _header(fig,
            "Error Distribution for Hamming Distance Metric",
            f"$n={n}$,   $d={d}$,   shots $={shots}$")

    plt.savefig("hamming_distance.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.show()


    # Error heatmap

    n_bits = len(expected_bits)
    bit_errors = np.zeros(n_bits)

    for bitstr, cnt in counts.items():
        if bitstr != expected_bits:
            for pos in range(n_bits):
                if bitstr[pos] != expected_bits[pos]:
                    bit_errors[pos] += cnt

    bit_error_rate = bit_errors / shots

    bits_per_qudit = 3
    n_qudits = n_bits // bits_per_qudit
    matrix = bit_error_rate.reshape(n_qudits, bits_per_qudit)

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
    fig.subplots_adjust(left=0.10, right=0.93, top=0.78, bottom=0.16)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_heatmap",
        ["#EFF6FF", "#BFDBFE", "#60A5FA", "#2563EB", "#1E3A8A"]
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)

    for i in range(n_qudits):
        for j in range(bits_per_qudit):
            val = matrix[i, j]
            txt_color = "white" if val > matrix.max() * 0.55 else TEXT_DARK
            ax.text(
                j, i, f"{val*100:.1f}%",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=txt_color
            )

    ax.set_xticks(range(bits_per_qudit))
    ax.set_xticklabels([f"Bit {j}" for j in range(bits_per_qudit)], fontsize=10)
    ax.set_yticks(range(n_qudits))
    ax.set_yticklabels([f"Qudit {i}" for i in range(n_qudits)], fontsize=10)
    ax.tick_params(length=0, pad=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Error rate", fontsize=10)
    cbar.ax.yaxis.set_tick_params(labelsize=9)

    ax.set_xlabel("Bit position in Qudit", fontsize=11, labelpad=8)
    ax.set_ylabel("Qudit", fontsize=11, labelpad=8)

    _header(fig,
            "Error per bit y qudit (Heatmap)",
            f"$n={n}$,   $d={d}$,   shots $=5000$")

    plt.savefig("heatmap_errors.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.show()

plot_sn_qdl_results(counts, expected_bits)