#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5))

# ---- Exp1: Sequential (no overlap) n=1024 ----
colors = {'H2D': '#3498db', 'Kernel': '#e74c3c', 'D2H': '#2ecc71'}

# Timings from experiment (us)
h2d_us = 47
kernels_us = 1194
d2h_us = 349

cursor = 0
y = 0
ax1.broken_barh([(0, h2d_us)], (y-0.35, 0.7), facecolors=colors['H2D'], label='H->D')
ax1.broken_barh([(h2d_us, kernels_us)], (y-0.35, 0.7), facecolors=colors['Kernel'], label='Kernels')
ax1.broken_barh([(h2d_us + kernels_us, d2h_us)], (y-0.35, 0.7), facecolors=colors['D2H'], label='D->H')

ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 1700)
ax1.set_yticks([])
ax1.set_xlabel('Tiempo (us)')
ax1.set_title('Experimento 1: Stream 0 (Sin solapamiento), Total = 1.590 ms')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, axis='x', alpha=0.3)

# ---- Exp2: S=4 streams with overlap ----
S = 4
B = 8
batch_size = 13
m = 100

# Realistic estimates based on nsys data:
# H2D per batch: ~5.6 us, kernel per batch: ~176 us
# But actual overlap means total time ~1.829 ms

# Build a simplified but conceptually accurate Gantt chart
ax2.set_xlim(0, 2000)
ax2.set_ylim(-1.5, S + 0.5)

stream_colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

# For each stream, show H2D overlapped with kernel
# The key insight: kernel for batch N-1 overlaps with H2D for batch N
for s in range(S):
    y_s = s
    batches_for_stream = [(b, s) for b in range(B) if b % S == s]

    prev_kernel_end = 0
    for batch_idx, (b, _) in enumerate(batches_for_stream):
        h2d_start = b * 6  # each H2D starts ~6us apart (pipelined)
        h2d_dur = 6
        kernel_start = max(h2d_start + h2d_dur, prev_kernel_end)
        kernel_dur = 180

        ax2.broken_barh([(h2d_start, h2d_dur)], (y_s-0.3, 0.6),
                         facecolors=stream_colors[s], alpha=0.4, edgecolor=stream_colors[s], linewidth=0.5)
        ax2.broken_barh([(kernel_start, kernel_dur)], (y_s-0.3, 0.6),
                         facecolors=stream_colors[s], alpha=0.8, edgecolor='darkred', linewidth=0.5)
        prev_kernel_end = kernel_start + kernel_dur

# Post-processing phase
ax2.broken_barh([(1600, 200)], (-0.8, 0.6), facecolors='#7f8c8d', alpha=0.5, label='Post-proc + D->H')

ax2.set_yticks(range(S))
ax2.set_yticklabels([f'Stream {i}' for i in range(S)])
ax2.set_xlabel('Tiempo (us)')
ax2.set_title('Experimento 2: S=4 streams con double buffering, Total = 1.829 ms')
ax2.grid(True, axis='x', alpha=0.3)

# Legend
legend_patches = [mpatches.Patch(color=c, alpha=0.8, label=f'S{s}')
                  for s, c in enumerate(stream_colors)]
legend_patches.append(mpatches.Patch(color='#7f8c8d', alpha=0.5, label='Post-proc'))
legend_patches.append(mpatches.Patch(color='gray', alpha=0.3, label='H->D'))
legend_patches.append(mpatches.Patch(color=stream_colors[0], alpha=0.8, label='Kernel'))
ax2.legend(handles=legend_patches, loc='upper right', fontsize=7, ncol=2)

# Add overlap indicator
ax2.annotate('', xy=(30, 1), xytext=(30, 2.5),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(35, 1.75, 'Overlap\nH2D||Kernel', fontsize=8, color='black',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

plt.tight_layout()
plt.savefig('/home/leninvaleria/2026-1/IaCP/iacp-tarea2/informe/timeline.pdf', dpi=150)
print('Generated informe/timeline.pdf')
