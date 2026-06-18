import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data points (Algorithm, HOTA, IDF1, MOTA)
data = [
    # ["SUSHI", 66.5, 83.1, 81.1],
    ["AdapTrack", 65.7, 82.3, 79.9],
    ["PIA", 66.0, 81.1, 82.2],
    ["ImprAsso", 66.4, 82.1, 82.2],

    ["Hybrid-SORT-ReID", 64.0, 78.7, 79.9],
    ["FineTrack", 64.3, 79.5, 80.0],
    ["StrongSORT++", 64.4, 79.5, 79.6], 
    ["Deep OC-SORT", 64.9, 80.6, 79.4],
    # ["SparseTrack", 65.1, 80.1, 81.0],
    ["UCMCTrack+", 65.7, 81.0, 80.6],
    ["BoostTrack++", 66.6, 82.2, 80.7],
    ["StableTrack", 67.1, 83.0, 82.0],
    ["TrackTrack", 67.1, 83.1, 81.8],
    ["PrismTrack (Ours)", 67.9, 83.5, 82.4]
]

# data = [
#     ["NewTrack (Ours)", 67.9, 83.5, 82.4],
#     ["TrackTrack", 67.1, 83.1, 81.8],
#     ["StableTrack", 67.1, 83.0, 82.0],
#     ["BoostTrack++", 66.6, 82.2, 80.7],
#     ["UCMCTrack+", 65.7, 81.0, 80.6],
#     ["SparseTrack", 65.1, 80.1, 81.0],
#     ["Deep OC-SORT", 64.9, 80.6, 79.4],
#     ["StrongSORT++", 64.4, 79.5, 79.6], 
#     ["FineTrack", 64.3, 79.5, 80.0],
#     ["Hybrid-SORT-ReID", 64.0, 78.7, 79.9]
# ]

df = pd.DataFrame(data, columns=['Tracker', 'HOTA', 'IDF1', 'MOTA'])

# 2. Set up graph style
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Bubble size
bubble_scale = 200
df['bubble_size'] = np.power(df['HOTA'], 2) * bubble_scale

# 3. Draw Bubble Chart
cmap = sns.color_palette("viridis", len(df))
scatter = sns.scatterplot(data=df, x='MOTA', y='IDF1', size='bubble_size',
                        hue='Tracker', palette=cmap, sizes=(1400, 6000), 
                        alpha=0.6, linewidth=0, legend=False)

# 4. Draw the center of the circle
plt.scatter(df['MOTA'], df['IDF1'], color='black', s=1, zorder=3)

# 5. Algorithm name
for i, txt in enumerate(df['Tracker']):
    x_val = df['MOTA'][i]
    y_val = df['IDF1'][i]
    
    font_weight = 'bold' if "(Ours)" in txt else 'normal'
    y_off = 0.3 
    
    # Manual fine-tuning
    if "Hybrid-SORT-ReID" in txt: y_off = +0.4
    if "FineTrack" in txt: y_off = +0.4
    if "Deep OC-SORT" in txt: y_off = +0.45
    if "SparseTrack" in txt: y_off = +0.5
    if "UCMCTrack" in txt: y_off = +0.5
    if "BoostTrack++" in txt: y_off = +0.55
    if "StrongSORT++" in txt: y_off = +0.4
    if "TrackTrack" in txt: y_off = -0.55
    if "StableTrack" in txt: y_off = +0.55
    if "PrismTrack (Ours)" in txt: y_off = +0.6
    if "AdapTrack" in txt: y_off = +0.5
    if "PIA" in txt: y_off = +0.5
    if "ImprAsso" in txt: y_off = +0.5
    if "SUSHI" in txt: y_off = +0.55

    plt.text(x_val, y_val + y_off, txt, fontsize=11, 
             weight=font_weight, ha='center', va='top')

# 6. Axis formatting by position
plt.xlabel('MOTA', fontsize=14, weight='bold')
plt.ylabel('IDF1', fontsize=14, weight='bold')

plt.xticks(np.arange(79, 83.5, 0.5))
plt.yticks(np.arange(78.5, 85.0, 0.5))
plt.xlim(79, 83)
plt.ylim(78.5, 84.5)

plt.text(0.03, 0.97, 'MOT17', transform=ax.transAxes, fontsize=24, 
         weight='bold', va='top', ha='left')

# 7. Save and display
save_dir = "assets"
os.makedirs(save_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "visualize_result_mot17.png"),
    dpi=300,
    bbox_inches="tight"
)
# plt.show()