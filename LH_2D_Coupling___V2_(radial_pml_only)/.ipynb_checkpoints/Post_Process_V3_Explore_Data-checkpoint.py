import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

class PMLDataExplorer:
    def __init__(self, h5_filepath):
        """
        Initializes the Data Lake Explorer. Converts HDF5 flat arrays into a Pandas DataFrame.
        """
        print(f"--- Loading Data Lake from {h5_filepath} ---")
        self.filepath = h5_filepath
        self.df = self._load_h5_to_dataframe()
        self._clean_data()
        
        # Set professional Seaborn aesthetics
        sns.set_theme(style="ticks", context="talk")
        
    def _load_h5_to_dataframe(self):
        """Reads flat 1D arrays from HDF5 and aligns them into a Pandas DataFrame."""
        data_dict = {}
        with h5py.File(self.filepath, 'r') as h5f:
            for key in h5f.keys():
                data_dict[key] = h5f[key][:]
        return pd.DataFrame(data_dict)
        
    def _clean_data(self):
        """Removes failed simulations (e.g., where matrix inversion crashed)."""
        initial_count = len(self.df)
        
        # In our extraction script, failed runs were penalized with Gamma = 1.0 and DoFs = 0
        self.df = self.df[self.df['Gamma_S'] < 1.0]
        self.df = self.df[self.df['DoFs'] > 0]
        
        final_count = len(self.df)
        print(f"[INFO] Data cleaned. Removed {initial_count - final_count} failed simulations. "
              f"Valid runs available: {final_count}")

    def plot_universal_scatter(self, x_var, y_var, color_var=None, 
                               log_x=False, log_y=False, log_c=False, 
                               filter_query=None, save_dir=None):
        """
        The Ultimate Plotter. Maps any variable against any other, colored by a third.
        
        Parameters:
        - x_var (str): Column name for X-axis (e.g., 'L_pml_ratio')
        - y_var (str): Column name for Y-axis (e.g., 'Gamma_S')
        - color_var (str): Column name for point colors (e.g., 'S_imag')
        - filter_query (str): Pandas query string to slice data (e.g., 'n_para == 2.0')
        """
        # 1. Apply Filters (Data Slicing)
        # This allows you to look at specific physics regimes instantly
        plot_df = self.df.copy()
        if filter_query is not None:
            try:
                plot_df = plot_df.query(filter_query)
                print(f"[INFO] Applied filter '{filter_query}'. Points remaining: {len(plot_df)}")
            except Exception as e:
                print(f"[ERROR] Invalid filter query: {e}")
                return

        if len(plot_df) == 0:
            print("[WARNING] Filter resulted in empty dataset. Nothing to plot.")
            return

        # 2. Setup the Figure
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 3. Dynamic Color Normalization
        norm = None
        cmap_name = 'magma_r' if log_y or y_var.startswith('Gamma') else 'viridis'

        if color_var is not None:
            col_min = plot_df[color_var].min()
            col_max = plot_df[color_var].max()

            if log_c and col_min > 0:
                norm = colors.LogNorm(vmin=col_min, vmax=col_max)
            else:
                norm = colors.Normalize(vmin=col_min, vmax=col_max)

        # 4. Generate the Scatter Plot using Seaborn
        # Seaborn handles mapping variables to aesthetics flawlessly
        scatter = sns.scatterplot(
            data=plot_df, x=x_var, y=y_var, hue=color_var,
            palette=cmap_name, hue_norm=norm, edgecolor='k', alpha=0.8, s=80, ax=ax
        )

        # 5. Apply Logarithmic Scales if requested
        if log_x: ax.set_xscale('log')
        if log_y: ax.set_yscale('log')

        # 6. Professional Aesthetics & Legends
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.tick_params(direction='in', length=6, width=1.5, bottom=True, top=True, left=True, right=True)
        
        # Fix the legend (Seaborn scatter legend can be clunky with continuous data)
        if color_var is not None:
            if ax.get_legend() is not None:
                ax.get_legend().remove() # Remove default legend
            
            sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(color_var, fontsize=14)

        title = f"{y_var} vs {x_var}"
        if filter_query: title += f"\nFilter: {filter_query}"
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel(x_var, fontsize=14)
        ax.set_ylabel(y_var, fontsize=14)

        plt.tight_layout()
        
        if save_dir is not None:
            filename = f"Scatter_{y_var}_vs_{x_var}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            print(f"[SUCCESS] Plot saved to {filename}")
            
        plt.show()

    def plot_correlation_heatmap(self, save_dir=None):
        """
        Calculates the Spearman Rank Correlation to mathematically prove
        which parameters drive the physics, ignoring non-linear shapes.
        """
        # Select only the numerical physics and output columns
        cols_to_correlate = [
            'S_imag', 'L_pml_ratio', 'S_real', 'p_degree', 'n_para', 'n_e', 
            'Lx_pml_meters', 'DoFs', 'CPU_Time', 'Gamma_S'
        ]
        
        # Filter dataframe to valid columns
        df_corr = self.df[cols_to_correlate]
        
        # Calculate Spearman correlation
        corr_matrix = df_corr.corr(method='spearman')
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Mask the upper triangle (it's redundant)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Draw the heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
                    cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title("Spearman Rank Correlation (Non-Linear Physics Trends)", fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, "Correlation_Heatmap.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Heatmap saved to {save_path}")
            
        plt.show()

    def plot_hypercube_pairplot(self, variables, hue_var, save_dir=None):
        """
        Generates a Scatter Matrix (Pairplot). 
        Plots every 2D combination of the requested variables simultaneously.
        """
        print(f"[INFO] Generating Pairplot for {variables}. This may take a few seconds...")
        
        # We take the log of Gamma_S to make the trends visible
        plot_df = self.df.copy()
        if 'Gamma_S' in variables:
            plot_df['Log_Gamma_S'] = np.log10(plot_df['Gamma_S'])
            variables[variables.index('Gamma_S')] = 'Log_Gamma_S'
            
        # Ensure the hue variable is in the dataframe subset
        cols = variables + [hue_var]
        subset_df = plot_df[cols]
        
        # Generate the Pairplot
        g = sns.pairplot(subset_df, hue=hue_var, palette='viridis', 
                         corner=True, diag_kind='kde', plot_kws={'alpha': 0.7, 's': 30})
        
        g.fig.suptitle(f"7D Hypercube Trend Matrix | Colored by {hue_var}", y=1.02, fontsize=18)
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, "Trend_Matrix_Pairplot.png")
            g.savefig(save_path, dpi=300)
            print(f"[SUCCESS] Pairplot saved to {save_path}")
            
        plt.show()