"""
SNA-QDL noise robustness benchmark — Pauli, Amplitude Damping, or both.

Three noise modes (select via NOISE_MODE below):

  'pauli'   — Depolarizing effect, corresponding to a particular choice of
                  E(rho) = sum_{a,b} p_{a,b} X^a Z^b rho Z^{-b} X^{-a}
              Approximated on 3 qubits via depolarizing error on 'id' gates.
                  p_1q = 1 - (1 - p_err)^(1/3)

  'damping' — Common amplitude damping channel, with amplitude coefficient 
              as the average of gamma_j ~ N(p_err, gamma_std), clipped to [0,1], 
              one draw per qudit level and circuit build. Implemented via 
              amplitude_damping_error on each qubit of the qudit.

  'both'    — Serial composition: damping channel first, then Pauli channel.
              Both noise hooks are injected between Alice and Bob; Qiskit Aer
              composes the errors automatically on the 'id' gates.

Theoretical quantities reported:

  Pauli    — Delta = 1 - p_err (symmetric channel, exact)
             Bound = n*(d-1)*exp(-T*Delta^2/2)   [Hoeffding]

  Damping  — Fidelity F simeq 1 - (1/d) * sum_{r = 0:d-1} gamma_r
             Reported as preservation probability in theory.

  both     — Both quantities reported side by side.

Simulation: MPS method (Qiskit Aer). Circuit compiled once per n, reused
across T values. SHOTS shots used as T repetitions for majority-vote decoding.

Dependencies: qiskit, qiskit-aer, tqdm, matplotlib, numpy
Authors: Yauset Cabrera-Aparicio, Jorge García-Díaz, Pino Caballero-Gil (2026)
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    QuantumError,
)


# Edit parameters here

d          = 8                          # qudit dimension
N_VALUES   = [100, 500, 1000]           # message lengths (qudits), be aware of computational limitations
T_VALUES   = [1, 2, 5, 10, 20]          # majority-vote repetitions
SHOTS      = 400                        # shots per (n, T) scenario — each shot := one former repetition

# Noise mode: 'pauli' | 'damping' | 'both'
NOISE_MODE = 'both'

# Shared error magnitude
P_ERR      = 0.03

# Standard deviation for gamma_j ~ N(P_ERR, GAMMA_STD) in damping channel
GAMMA_STD  = 0.1


def _sample_gammas(d: int, p_err: float, std: float) -> np.ndarray:
    """
    Compute gamma_j ~ N(p_err, std) for j = 0, ..., d-1 and clip to [0, 1].
    These are the per-level damping rates for the cascade amplitude damping
    channel Lambda_gamma acting on a d-dimensional qudit. May be used in the
    future for not-simplistic approaches for damping modeling.
    """
    
    rng = np.random.default_rng(seed=42)
    gammas = rng.normal(p_err, std, size=d)
    return np.clip(gammas, 0.0, 1.0)


def _damping_fidelity(gammas: np.ndarray, d: int) -> float:
    """
    Compute the state-preservation probability for the generalized amplitude
    damping channel (Equation derived in the theoretical analysis):

        F simeq 1 - (1/d) * sum_{r = 0:d-1} gamma_r

    This quantity is independent of the message symbol j (as shown in the
    analysis) and the fidelity of the state received after noise effect.
    """

    sum_gam  = np.sum(gammas[1:])
    return 1- sum_gam / d


def build_noise_model(
    mode: str,
    p_err: float,
    gammas: np.ndarray,
) -> NoiseModel:
    """
    Build a Qiskit NoiseModel attached to the standard 'id' gate.

    All noise is composed into a single QuantumError per mode and attached
    to the standard 'id' gate (Qiskit 2.x does not allow custom gate names
    in basis_gates; using 'id' avoids that restriction).

    Composition order for mode='both' (serial):
        error = amplitude_damping.compose(depolarizing)
    i.e. damping is applied first, then Pauli — physically consistent with
    decoherence preceding stochastic Pauli errors.

    Effective rates:
      Pauli   — p_1q = 1 - (1 - p_err)^(1/3)
      Damping — gamma_eff = mean(gamma_j)   [ps: mean over qudit levels]
    """

    nm = NoiseModel()

    if mode == 'pauli':
        p_1q = float(np.clip(1.0 - (1.0 - p_err) ** (1.0 / 3.0), 0.0, 1.0))
        error = depolarizing_error(p_1q, 1)

    elif mode == 'damping':
        gamma_eff = float(np.clip(np.mean(gammas), 0.0, 1.0))
        error = amplitude_damping_error(gamma_eff)

    else:
        gamma_eff = float(np.clip(np.mean(gammas), 0.0, 1.0))
        p_1q      = float(np.clip(1.0 - (1.0 - p_err) ** (1.0 / 3.0), 0.0, 1.0))
        error = amplitude_damping_error(gamma_eff).compose(
                    depolarizing_error(p_1q, 1))

    nm.add_all_qubit_quantum_error(error, ['id'])
    return nm


def build_circuit(
    msg:   np.ndarray,
    theta: np.ndarray,
    n:     int,
    mode:  str,
) -> QuantumCircuit:
    """
    SNA-QDL circuit for n qudits (3 qubits each convention).
    """

    total_q = 3 * n
    qc      = QuantumCircuit(total_q, total_q)
    qft3    = QFT(3, do_swaps=True).to_gate()
    iqft3   = qft3.inverse()

    for j in range(n):
        q  = 3 * j
        th = float(theta[j])

        for i, bit in enumerate(reversed(format(int(msg[j]), '03b'))):
            if bit == '1':
                qc.x(q + i)

        # Encryption
        qc.append(qft3, [q, q+1, q+2])
        qc.p(th * np.pi / 4, q)
        qc.p(th * np.pi / 2, q + 1)
        qc.p(th * np.pi,     q + 2)

        # Noise
        qc.id(q); qc.id(q+1); qc.id(q+2)

        # Decryption
        qc.p(-th * np.pi / 4, q)
        qc.p(-th * np.pi / 2, q + 1)
        qc.p(-th * np.pi,     q + 2)
        qc.append(iqft3, [q, q+1, q+2])

    qc.measure(range(total_q), range(total_q))
    return qc


def decode_counts(counts: dict, n: int, d: int) -> np.ndarray:
    """
    Majority-vote decode: for each qudit, argmax over all shots in counts.
    Bitstring ordering: Bear in mind that Qiskit stores qubit 0 at rightmost
    position, so we reverse the string so that index i corresponds to qubit i.
    """
    votes = [np.zeros(d, dtype=int) for _ in range(n)]
    for bs_raw, cnt in counts.items():
        bs = bs_raw.replace(' ', '')[::-1]
        for j in range(n):
            sym = int(bs[3*j : 3*j+3][::-1], 2)
            votes[j][sym] += cnt
    return np.array([int(np.argmax(v)) for v in votes])


def pauli_delta_and_bound(p_err: float, d: int, n: int, T: int):
    """Delta and Hoeffding bound for the symmetric Pauli channel."""
    q0    = (1.0 - p_err) + p_err / (d + 1)
    qr    = p_err / (d + 1)
    Delta = q0 - qr                          # = 1 - p_err (exact)
    bound = n * (d - 1) * np.exp(-T * Delta ** 2 / 2.0)
    return Delta, bound


def run_benchmark():
    gammas = _sample_gammas(d, P_ERR, GAMMA_STD)
    fidelity = _damping_fidelity(gammas, d)

    noise_model = build_noise_model(NOISE_MODE, P_ERR, gammas)
    simulator   = AerSimulator(
        method='matrix_product_state',
        noise_model=noise_model,
    )

    total   = len(N_VALUES) * len(T_VALUES)
    results = []

    mode_str = {
        'pauli':   'Pauli channel only',
        'damping': 'Amplitude damping only',
        'both':    'Damping ∘ Pauli  (serial composition)',
    }[NOISE_MODE]

    print(f"\nSNA-QDL Noise Benchmark  |  mode={NOISE_MODE}  p_err={P_ERR}"
          f"  d={d}  shots={SHOTS}")
    print(f"  {mode_str}")
    if NOISE_MODE in ('damping', 'both'):
        gstr = ', '.join(f'{g:.4f}' for g in gammas)
        print(f"  gamma_j (d levels) = [{gstr}]")
        print(f"  Damping fidelity F = {fidelity:.6f}")
    print("=" * 72)

    with tqdm(total=total, desc="Simulating", unit="scenario",
              bar_format="{l_bar}{bar:35}{r_bar}") as pbar:

        for n in N_VALUES:
            kappa = max(1, int(2 * (np.log(n) / np.log(d))))
            msg   = np.random.randint(0, d, size=n)
            key   = np.random.randint(0, d, size=kappa)
            F     = np.random.randint(0, d, size=(n, kappa), dtype=np.int32)
            theta = np.dot(F, key) % d

            qc  = build_circuit(msg, theta, n, NOISE_MODE)
            cqc = transpile(
                qc,
                basis_gates=['u', 'cx', 'p', 'x', 'id'],
                optimization_level=0,
            )

            for T in T_VALUES:
                pbar.set_postfix(n=n, T=T, refresh=True)

                # Theoretical Pauli quantities (interesting to compute for reference)
                Delta, bound = pauli_delta_and_bound(P_ERR, d, n, T)

                t0     = time.time()
                result = simulator.run(cqc, shots=SHOTS).result()
                t_sim  = time.time() - t0
                counts = result.get_counts()

                # Per-shot error rate (independent decoding of each shot)
                per_shot = []
                for bs_raw, cnt in counts.items():
                    bs  = bs_raw.replace(' ', '')[::-1]
                    dec = np.array([int(bs[3*j:3*j+3][::-1], 2) for j in range(n)])
                    per_shot.extend([float(np.mean(dec != msg))] * cnt)

                # Majority-vote decode across all shots
                decoded = decode_counts(counts, n, d)
                mv_rate = float(np.mean(decoded != msg))

                results.append({
                    'n':         n,
                    'T':         T,
                    'mode':      NOISE_MODE,
                    'Delta':     Delta,
                    'bound':     bound,
                    'fidelity':  fidelity,
                    'gammas':    gammas.tolist(),
                    'mv_rate':   mv_rate,
                    'shot_mean': float(np.mean(per_shot)),
                    'shot_std':  float(np.std(per_shot)),
                    't_sim':     t_sim,
                })
                pbar.update(1)

    return results, gammas, fidelity


def print_table(results, fidelity):
    mode = results[0]['mode']
    print()

    # Column layout depends on mode
    if mode == 'pauli':
        print(f"{'N':>6}  {'T':>4}  {'Δ':>8}  {'Bound':>11}  "
              f"{'MV err%':>8}  {'Shot err%±std':>16}  {'Sim(s)':>7}")
        print("-" * 73)
        for r in results:
            bs = f"{r['bound']:.3e}" if r['bound'] < 1e6 else ">1e6"
            print(f"{r['n']:>6,}  {r['T']:>4}  {r['Delta']:>8.5f}  {bs:>11}  "
                  f"{r['mv_rate']*100:>7.2f}%  "
                  f"{r['shot_mean']*100:>7.2f}%±{r['shot_std']*100:.2f}%  "
                  f"{r['t_sim']:>6.1f}s")

    elif mode == 'damping':
        print(f"  Damping fidelity F = {fidelity:.6f}  "
              f"(state-preservation probability per qudit per shot)")
        print(f"{'N':>6}  {'T':>4}  {'F':>8}  "
              f"{'MV err%':>8}  {'Shot err%±std':>16}  {'Sim(s)':>7}")
        print("-" * 65)
        for r in results:
            print(f"{r['n']:>6,}  {r['T']:>4}  {r['fidelity']:>8.5f}  "
                  f"{r['mv_rate']*100:>7.2f}%  "
                  f"{r['shot_mean']*100:>7.2f}%±{r['shot_std']*100:.2f}%  "
                  f"{r['t_sim']:>6.1f}s")

    else:  # both
        print(f"  Damping fidelity F = {fidelity:.6f}  |  "
              f"Pauli Δ = {results[0]['Delta']:.5f}")
        print(f"{'N':>6}  {'T':>4}  {'Δ':>8}  {'Bound':>11}  {'F':>8}  "
              f"{'MV err%':>8}  {'Shot err%±std':>16}  {'Sim(s)':>7}")
        print("-" * 84)
        for r in results:
            bs = f"{r['bound']:.3e}" if r['bound'] < 1e6 else ">1e6"
            print(f"{r['n']:>6,}  {r['T']:>4}  {r['Delta']:>8.5f}  {bs:>11}  "
                  f"{r['fidelity']:>8.5f}  "
                  f"{r['mv_rate']*100:>7.2f}%  "
                  f"{r['shot_mean']*100:>7.2f}%±{r['shot_std']*100:.2f}%  "
                  f"{r['t_sim']:>6.1f}s")


def make_plot(results, gammas, fidelity):
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'sans-serif'

    rng    = np.random.default_rng(0)
    n_vals = sorted(set(r['n'] for r in results))
    t_vals = sorted(set(r['T'] for r in results))
    mode   = results[0]['mode']
    n_max  = max(n_vals)

    # Colors
    BG        = 'white'
    PANEL_BG  = '#F7F9FC'
    GRID_COL  = '#D8E2EE'
    SPINE_COL = '#707070'
    TEXT_COL  = '#1A3A5C'
    SUB_COL   = '#2E6DA4'

    base_colors = [
        '#1A3A5C',
        '#2E6DA4',
        '#5B9BD5',
        '#92BAD9',
        '#9CA3AF',
    ]
    palette  = {n: base_colors[i % len(base_colors)] for i, n in enumerate(n_vals)}
    mkr_list = ['o', 's', '^', 'D', 'x']
    markers  = {t: mkr_list[i % len(mkr_list)] for i, t in enumerate(t_vals)}

    mode_title = {
        'pauli':   'Generalized Pauli Channel',
        'damping': 'Amplitude Damping Channel',
        'both':    'Damping \u2218 Pauli  (serial composition)',
    }[mode]


    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.34, left=0.07, right=0.97, top=0.87, bottom=0.13)

    def _style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=SPINE_COL, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)
            spine.set_linewidth(0.7)
        ax.grid(True, color=GRID_COL, linewidth=0.8, linestyle='--', zorder=0)

    for ax in axes:
        _style_ax(ax)

    ax0 = axes[0]

    for r in results:
        n, T = r['n'], r['T']
        mean = r['shot_mean'] * 100
        std  = max(r['shot_std'] * 100, 0.01)
        xs   = rng.uniform(0, SHOTS, size=SHOTS)
        ys   = np.clip(rng.normal(mean, std, size=SHOTS), 0, 100)
        ax0.scatter(xs, ys, s=9, alpha=0.22,
                    color=palette[n], marker=markers[T],
                    linewidths=0, zorder=3)

    for n_val in n_vals:
        sub  = [r for r in results if r['n'] == n_val]
        mean = np.mean([r['shot_mean'] for r in sub]) * 100
        ax0.axhline(mean, color=palette[n_val], linewidth=1.4,
                    alpha=0.80, linestyle='--', zorder=2)

    ax0.set_xlabel('Shot index', fontsize=10, labelpad=6)
    ax0.set_ylabel('Per-shot symbol error rate  (%)', fontsize=10, labelpad=6)
    ax0.set_title(
    r'Per-shot error scatter  '
    r'$p_{\mathrm{err}}$' + f' = {P_ERR},'
    r'$\ \gamma_{\mathrm{std}}$' + f' = {GAMMA_STD},'
    r'$\ d$' + f' = {d}',
    fontsize=10
    )
    ax0.set_xlim(0, SHOTS)
    ax0.set_ylim(bottom=0)
    ax0.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    leg_n = [Line2D([0],[0], marker='o', color='w',
                    markerfacecolor=palette[n], markeredgewidth=0,
                    markersize=7, label=f'N = {n:,}')
             for n in n_vals]
    leg_t = [Line2D([0],[0], marker=markers[t], color=SPINE_COL,
                    markersize=6, linestyle='none', label=f'T = {t}')
             for t in t_vals]
    leg = ax0.legend(
        handles=leg_n + leg_t, ncol=2,
        framealpha=0.90, labelcolor=TEXT_COL, fontsize=8,
        facecolor='white', edgecolor=GRID_COL,
        loc='upper right',
    )

    ax1 = axes[1]
    t_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(t_vals)))
    all_errors_combined = []

    for idx, T in enumerate(t_vals):
        matches = [r for r in results if r['n'] == n_max and r['T'] == T]
        if not matches:
            continue
        r = matches[0]
        mean_e = r['shot_mean'] * 100
        std_e  = max(r['shot_std'] * 100, 0.01)

        # Reconstruct synthetic shot distribution (same seed logic as scatter)
        rng2   = np.random.default_rng(idx + 42)
        shots  = np.clip(rng2.normal(mean_e, std_e, size=SHOTS), 0, 100)
        all_errors_combined.extend(shots.tolist())

        ax1.hist(
            shots,
            bins=30,
            density=True,
            color=t_colors[idx],
            alpha=0.55,
            edgecolor='white',
            linewidth=0.4,
            label=f'T = {T}',
            zorder=3 + idx,
        )

    if all_errors_combined:
        grand_mean = float(np.mean(all_errors_combined))
        ax1.axvline(grand_mean, color=SUB_COL, linewidth=1.6,
                    linestyle='--', zorder=10,
                    label=f'Mean = {grand_mean:.1f}%')

    ax1.set_xlabel('Per-shot symbol error rate  (%)', fontsize=10, labelpad=6)
    ax1.set_ylabel('Density', fontsize=10, labelpad=6)
    ax1.set_title(
        "Histogram of Error Distribution (" + r"$N =$" + f"{n_max:},  all T values)\n"
        r"$p_{\mathrm{err}}$" + f" = {P_ERR}, " + r"$d$" + f" = {d}, shots = {SHOTS}",
        fontsize=10, pad=8, color=TEXT_COL,
    )
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    leg1 = ax1.legend(
        framealpha=0.90, labelcolor=TEXT_COL, fontsize=8,
        facecolor='white', edgecolor=GRID_COL,
    )

    fig.suptitle(
        f'SNA-QDL  \u00b7  {mode_title}  \u00b7  MPS Simulation',
        color=TEXT_COL, fontsize=12, fontweight='bold', y=0.98,
    )

    plt.show()


if __name__ == '__main__':
    results, gammas, fidelity = run_benchmark()
    print_table(results, fidelity)
    make_plot(results, gammas, fidelity)