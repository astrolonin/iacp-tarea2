#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = [1, 2, 4, 8, 16]

# Size 32x32 (n=1024)
totals_32 = [1.641, 2.578, 1.829, 2.135, 2.593]
speedup_32 = [totals_32[0] / t for t in totals_32]

# Size 64x64 (n=4096)
totals_64 = [27.683, 28.193, 29.334, 34.761, 34.853]
speedup_64 = [totals_64[0] / t for t in totals_64]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Speedup plot
ax1.plot(S, speedup_32, 'o-', label='n=1024 (32x32)', linewidth=2, markersize=6)
ax1.plot(S, speedup_64, 's-', label='n=4096 (64x64)', linewidth=2, markersize=6)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Linea base')
ax1.set_xlabel('Numero de streams S')
ax1.set_ylabel('Speedup vs. S=1')
ax1.set_title('Speedup en funcion de S')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(S)

# Total time plot
ax2.plot(S, totals_32, 'o-', label='n=1024 (32x32)', linewidth=2, markersize=6)
ax2.plot(S, totals_64, 's-', label='n=4096 (64x64)', linewidth=2, markersize=6)
ax2.set_xlabel('Numero de streams S')
ax2.set_ylabel('Tiempo total (ms)')
ax2.set_title('Tiempo total en funcion de S')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(S)

plt.tight_layout()
plt.savefig('/home/leninvaleria/2026-1/IaCP/iacp-tarea2/informe/speedup.pdf', dpi=150)
print('Generated informe/speedup.pdf')
