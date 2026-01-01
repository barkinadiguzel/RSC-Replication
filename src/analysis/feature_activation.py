import matplotlib.pyplot as plt

def plot_feature_map(feature, save_path="feature.png"):
    # feature: [C, H, W] or [B, C, H, W]
    if feature.dim() == 4:
        feature = feature[0]
    mean_map = feature.mean(0).detach().cpu()
    plt.imshow(mean_map, cmap="viridis")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
