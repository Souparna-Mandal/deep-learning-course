import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

data = {
    'Baseline MLP': [49.16, 47.78],
    'one_conv_pool (K=3) & ClassificationHead (32,64)': [60.33, 58.97],
    '3 X one_conv_pool (K=3) & ClassificationHead (32,64)': [59.1, 57.77],
    '3 X two_conv_pool (K=3) & ClassificationHead (32,64)': [64.43, 62.53],
    '3 X two_conv_pool (K=3) & ClassificationHead (32,64,128,256)': [59.87, 58.41],
    '3 X two_conv_pool (K=2) & ClassificationHead (32,64)': [55.69, 56.32],
    '3 X two_conv_pool (K=4) & ClassificationHead (32,64)': [67.42, 65.75],
    '3 X two_conv_pool (K=5) & ClassificationHead (32,64)': [69.12, 65.24],
    'M7 with stride=2 downsampling instead of MaxPooling': [65.11, 61.68],
}

models = list(data.keys())
train_acc = np.array([v[0] for v in data.values()])
val_acc   = np.array([v[1] for v in data.values()])

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))

bars1 = ax.bar(x - width/2, train_acc, width, color='#1f77b4')
bars2 = ax.bar(x + width/2, val_acc,   width, color='#ff7f0e')

# Value labels
for i in range(len(models)):
    ax.text(x[i] - width/2, train_acc[i] + 0.5, f'{train_acc[i]:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(x[i] + width/2, val_acc[i] + 0.5, f'{val_acc[i]:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=8)
ax.set_title('Task 1: CIFAR-10 Classification Accuracy by Model', fontsize=10)

short_labels = [f'M{i+1}' for i in range(len(models))]
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=0, fontsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_ylim(40, 74)

# Train/Val legend (top-left)
train_patch = Patch(color='#1f77b4', label='Train')
val_patch   = Patch(color='#ff7f0e', label='Validation')
legend1 = ax.legend([train_patch, val_patch], ['Train', 'Validation'], 
                    fontsize=7, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Model M1-M9 legend (below Train/Val, still left)
model_desc = [f'M{i+1}: {models[i]}' for i in range(len(models))]
model_patches = [Patch(color='white', label=desc) for desc in model_desc]
legend2 = ax.legend(model_patches, model_desc, 
                    fontsize=7, loc='upper left', bbox_to_anchor=(0.02, 0.9))

# Add both legends
ax.add_artist(legend1)
ax.add_artist(legend2)

plt.tight_layout()
plt.savefig("cifar10_models_accuracy_final.png", dpi=300, bbox_inches='tight')
plt.close()
