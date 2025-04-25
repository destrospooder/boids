import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("random_search_results_cafeteria.csv")

# Plot average coverage vs each gain parameter
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# k_coh vs average
axs[0].scatter(df['k_coh'], df['average'], color='tab:blue')
axs[0].set_title('Avg Coverage vs Cohesion Gain (k_coh)')
axs[0].set_xlabel('Cohesion Gain (k_coh)')
axs[0].set_ylabel('Average Coverage (%)')

# k_ali vs average
axs[1].scatter(df['k_ali'], df['average'], color='tab:orange')
axs[1].set_title('Avg Coverage vs Alignment Gain (k_ali)')
axs[1].set_xlabel('Alignment Gain (k_ali)')
axs[1].set_ylabel('Average Coverage (%)')

# k_col vs average
axs[2].scatter(df['k_col'], df['average'], color='tab:green')
axs[2].set_title('Avg Coverage vs Separation Gain (k_col)')
axs[2].set_xlabel('Separation Gain (k_col)')
axs[2].set_ylabel('Average Coverage (%)')

plt.tight_layout()
plt.show()
