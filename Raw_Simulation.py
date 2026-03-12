"""
Benchmark-Raw simulation for SNA-QDL protocol.

This script evaluates the scalability and performance of the SNA-QDL protocol
implementation using Qiskit. The protocol encodes classical symbols (qudits
with dimension d=8) into groups of three qubits, applies a Quantum Fourier
Transform (QFT), introduces phase shifts derived from a secret key, and
subsequently applies the inverse operations to recover the original message.

The purpose of this program is to measure the computational cost of building
and simulating the corresponding quantum circuits for increasing message sizes.
The benchmark records:

- Circuit construction time
- Simulation time
- Correctness of the recovered message

The simulation is executed using the Matrix Product State (MPS) method
provided by Qiskit Aer, which allows the simulation of relatively large
quantum circuits compared to standard statevector approaches. Results are 
printed to the console and stored in a text file ("benchmark_results.txt")
for further analysis.

Dependencies: may be installed using via requirements.txt
Authors: Yauset Cabrera-Aparicio, Jorge García-Díaz, Pino Caballero-Gil (2026)
"""


import numpy as np
import time
import sys
import os
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def run_benchmark():
    # We make use of MPS method to handle larger circuits, but be aware that it may still run out of memory for very large n.
    simulator = AerSimulator(method='matrix_product_state')
    
    powers = [10**1, 10**2, 10**3, 10**4, 2*10**4, 5*10**4]
    
    output_file = "benchmark_results.txt"
    
    header = f"{'N (Qudits)':<15} | {'QUBITS':<15} | {'Encrypt (sec)':<15} | {'Simulation (sec)':<15} | {'Success'}"
    separator = "=" * 80
    
    print("\n" + separator)
    print(header)
    print(separator)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(header + "\n" + separator + "\n")

    for n in powers:
        try:
            d = 8
            kappa = max(1, int(2 * (np.log(n) / np.log(d)))) 
            total_qubits = 3 * n
            
            # Alice prepares message and key
            msg = np.random.randint(0, d, size=n)
            key = np.random.randint(0, d, size=kappa)
            F = np.random.randint(0, d, size=(n, kappa), dtype=np.int32) 
            theta = np.dot(F, key) % d

            # Circuit construction
            t0 = time.time()
            qc = QuantumCircuit(total_qubits, total_qubits)
            
            # QFT definition
            qft_3 = QFT(3, do_swaps=True).to_gate()
            iqft_3 = qft_3.inverse()

            for j in range(n):
                q_idx = 3 * j
                
                # Codification
                m_bin = format(msg[j], '03b')
                for i, bit in enumerate(reversed(m_bin)):
                    if bit == '1': qc.x(q_idx + i)
                
                # Alice
                qc.append(qft_3, [q_idx, q_idx+1, q_idx+2])
                qc.p(theta[j] * np.pi/4, q_idx)
                qc.p(theta[j] * np.pi/2, q_idx + 1)
                qc.p(theta[j] * np.pi, q_idx + 2)
                
                # Bob
                qc.p(-theta[j] * np.pi/4, q_idx)
                qc.p(-theta[j] * np.pi/2, q_idx + 1)
                qc.p(-theta[j] * np.pi, q_idx + 2)
                qc.append(iqft_3, [q_idx, q_idx+1, q_idx+2])

            qc.measure(range(total_qubits), range(total_qubits))
            t_build = time.time() - t0

            # Simulation
            t1 = time.time()
            # Optimization_level=0 mandatory to prevent qiskit from trying to optimize the circuit and running out of memory
            compiled_circuit = transpile(qc, basis_gates=['u', 'cx', 'p', 'x', 'id'], optimization_level=0)
            
            result = simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            t_sim = time.time() - t1

            # Validation
            measured_bits = list(counts.keys())[0]
            expected_bits = ''.join(format(x, '03b') for x in reversed(msg))
            success = "YES" if measured_bits == expected_bits else "NO"

            row = f"{n:<15,} | {total_qubits:<15,} | {t_build:<15.4f} | {t_sim:<15.4f}  | {success}"
            print(row)
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(row + "\n")
            
            # Memory cleanup
            del qc
            del compiled_circuit
            del msg
            del theta

        except MemoryError:
            msg_err = f"10^{int(np.log10(n))} | ERROR: Insufficient memory."
            print(msg_err)
            break
        except Exception as e:
            msg_err = f"10^{int(np.log10(n))} | Error: {str(e)}"
            print(msg_err)
            break

    print(separator)
    print(f"Results saved in: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    run_benchmark()