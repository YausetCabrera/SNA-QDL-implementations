"""
Real-Backend Noise Simulation for SNA-QDL Protocol.

This script performs a noise simulation of the SNA-QDL protocol using a real IBM
Quantum backend through Qiskit Runtime. The purpose of this program is to evaluate
the effects of realistic hardware noise on the SNA-QDL protocol. It measures:

- Fidelity of the recovered message
- Distribution of errors across different qudits
- Visual representation of probabilistic outcomes via histograms

The simulation is executed by connecting to an IBM Quantum processor backend and
leveraging Qiskit Aer’s simulator with backend-specific noise characteristics. The
circuit is transpiled for the target backend to account for connectivity and gate
constraints, and measurements are collected using the Sampler primitive of the
Qiskit Runtime service.

Results provide insights into the robustness and error patterns of the protocol
under realistic quantum hardware conditions, serving as a benchmark for fidelity
without any active error correction.

Dependencies: Qiskit, Qiskit Aer, Qiskit IBM Runtime, NumPy, Matplotlib
Authors: Yauset Cabrera-Aparicio, Jorge García-Díaz, Pino Caballero-Gil (2026)
"""

import numpy as np
import warnings
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import TGate, SGate, ZGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=DeprecationWarning)

service = QiskitRuntimeService(
    channel="ibm_cloud",
    token="9GwdH1QMaQUZMlY-cy4Q60pZ8-djWc1HBLPU4LT4t28P",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/3e3cb62a55634dfc8315f0de08e16991:a6e46b8d-6343-4694-be44-61ccfa8bb190::"
)

# It is possible to change the backend to another one availaible. You may consult the list by uncommenting the next lines
backends = service.backends()
for backend in backends:
    print(backend.name)

real_chip = service.backend("ibm_fez")
backend = AerSimulator.from_backend(real_chip)
print(f"Conected to IBM! Using simulator {real_chip.name}")

# Configuration
n = 5
d = 8
kappa = 2
total_qubits = 3 * n
shots = 4000

msg = np.random.randint(0, d, size=n)
key = np.random.randint(0, d, size=kappa)

F = np.tile(np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 1]]), (int(np.ceil(n/8)), 1))[:n] # Example for n <= 8
theta = np.dot(F, key) % d

print(f"Original Message: {msg}")

phase_dict = {
    0: [], 1: [TGate()], 2: [SGate()], 3: [SGate(), TGate()],
    4: [ZGate()], 5: [ZGate(), TGate()], 6: [ZGate(), SGate()],
    7: [ZGate(), SGate(), TGate()]
}

def apply_ftqc_phase(qc, qubit, k):
    for gate in phase_dict[int(k) % 8]:
        qc.append(gate, [qubit])

def apply_qudit_phase(qc, start, theta_val, inverse=False):
    reps = 7 if inverse else 1
    for _ in range(reps):
        apply_ftqc_phase(qc, start, theta_val)
        apply_ftqc_phase(qc, start + 1, 2 * theta_val)
        apply_ftqc_phase(qc, start + 2, 4 * theta_val)


# Circuits construction
qc = QuantumCircuit(total_qubits)
qft3 = QFT(3, do_swaps=True).to_gate()
iqft3 = QFT(3, do_swaps=True, inverse=True).to_gate()

for j in range(n):
    q = 3 * j
    for i, bit in enumerate(reversed(format(msg[j], "03b"))):
        if bit == "1": qc.x(q + i)
    qc.append(qft3, [q, q+1, q+2])
    apply_qudit_phase(qc, q, theta[j])

qc.barrier()

for j in range(n):
    q = 3 * j
    apply_qudit_phase(qc, q, theta[j], inverse=True)
    qc.append(iqft3, [q, q+1, q+2])

qc.measure_all()

print("Transpiling circuit...")
compiled_qc = transpile(qc, backend=backend, optimization_level=2)

print("Initializing computation...")
sampler = Sampler(mode=backend)
job = sampler.run([compiled_qc], shots=shots)
result = job.result()

pub_result = result[0]
counts_data = pub_result.data.meas.get_counts()
counts = {k: v for k, v in counts_data.items()}


# Visualization and metrics
expected_bits = ''.join(format(x, '03b') for x in reversed(msg))
success = counts.get(expected_bits, 0)
print(f"\n--- RESULTS ---")
print(f"Expected Message: {expected_bits}")
print(f"Fidelity: {(success/shots)*100:.2f}%")

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