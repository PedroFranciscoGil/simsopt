import numpy as np
import matplotlib.pyplot as plt

orders = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

qa_data=[[4.98e-03, 1.33e-03, 1.15e-03, 2.62e-03],
         [1.16e-03, 1.86e-03, 9.86e-04, 9.68e-04],
         [1.13e-03, 9.93e-04, 1.19e-03, 9.76e-04],
         [1.82e-03, 9.01e-04, 1.41e-03, 2.48e-03],
         [1.02e-03, 1.25e-03, 2.13e-03, 1.10e-03],
         [1.63e-03, 1.40e-03, 1.55e-03, 1.63e-03],
         [1.08e-03, 1.54e-03, 1.43e-03, 1.56e-03],
         [3.89e-03, 1.39e-03, 9.62e-04, 2.01e-03],
         [1.01e-03, 9.91e-04, 1.15e-03, 2.07e-03],
         [2.42e-03, 1.31e-03, 1.09e-03, 1.31e-03],
         [1.24e-03, 1.36e-03, 5.95e-03, 1.30e-03],
         [2.27e-03, 1.01e-03, 2.67e-03, 9.16e-04]
]

qa_means = np.mean(qa_data, axis=1)
qa_stds = np.std(qa_data, axis=1)

plt.figure(figsize=(12, 8))

colors = {
    'QA': '#1f77b4'
}

plt.errorbar(orders, qa_means, yerr=qa_stds, 
            marker='o', markersize=8, capsize=5, capthick=2,
            color=colors['QA'], label='QA',
            linewidth=2, linestyle='-')

plt.xlabel('Order', fontsize=14, fontweight='bold')
plt.ylabel('|B⋅N|/|B|', fontsize=14, fontweight='bold')
plt.title('|B⋅N|/|B| vs Order (Semilog Plot)', fontsize=16, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3, linestyle='--', which='both')
plt.legend(loc='upper right', fontsize=12, frameon=True, 
           fancybox=True, shadow=True, framealpha=0.9)
plt.xticks(orders)
plt.tight_layout()
plt.show()
plt.savefig('stellarator_semilog_bdotn_analysis.png', dpi=300, bbox_inches='tight')