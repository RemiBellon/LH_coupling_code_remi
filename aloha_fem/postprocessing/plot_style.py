import matplotlib.pyplot as plt
import os 

def apply_style():
    """
    Applies a publication-ready aesthetic to all Matplotlib figures.
    Ensures consistency across all diagnostic plots.
    """
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.top": True,
        "ytick.right": True, 
        'figure.dpi': 300,
        "savefig.bbox": "tight",
    })

def save_dual_format(fig, save_dir: str, base_filename: str):
    """
    Saves a Matplotlib figure in both PDF (for reports) and SVG (for slides).
    """
    if not save_dir:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, base_filename)
    
    # Save both formats consecutively
    for ext in [".pdf", ".svg"]:
        fig.savefig(f"{base_path}{ext}")
        print(f"Saved: {base_path}{ext}")