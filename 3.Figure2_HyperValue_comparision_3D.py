import os
import pandas as pd
import matplotlib.pyplot as plt

heart_dataset = 1
if heart_dataset:
    # 1) heart dataset:
    CSV_PATH    = "../Co_work_Samsad_WISE2025-extension_report/20260330_034509-FormalTest-B6-Frac0.6-heart_Nm20_G50_Pc0.9_Pm0.1_Scaler10_HL5_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv"   # path to the merged front‐points CSV
    OUTPUT_DIR  = "../Co_work_Samsad_WISE2025-extension_report/figures/"               # where to save the grid of plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = 'heart_data_hv_vs_epsilon.pdf'
else:
    # 2) diabetics dataset:
    CSV_PATH    ="../Co_work_Samsad_WISE2025-extension_report/20260324_121738-FormalTest-B6-Frac0.6-Diabetes_Nm20_G50_Pc0.9_Pm0.1_Scaler10_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv"   # path to the merged front‐points CSV
    OUTPUT_DIR  = "../Co_work_Samsad_WISE2025-extension_report/figures/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = 'diabetes_data_hv_vs_epsilon.pdf'


def plot_hv_vs_epsilon(epsilon_totals, hv_nsgaii, hv_moma, hv_amoma, figure_path):
    fonts = 14
    plt.figure(figsize=(8, 6))

    plt.plot(epsilon_totals, hv_nsgaii, '-^', label='NSGA-II-DPK',
             markersize=6, color='#ff7f0e')
    plt.plot(epsilon_totals, hv_moma, '-o', label='MOMA-DPK',
             markersize=6, color='#9467bd')
    plt.plot(epsilon_totals, hv_amoma, '-*', label='AMOMA-DPK',
             markersize=8, color='#d62728')

    plt.xlabel(r'$\epsilon_{\mathrm{total}}$', fontsize=fonts + 2)
    plt.ylabel('Avg Hypervolume', fontsize=fonts + 2)

    plt.xticks(
        epsilon_totals,
        [f"${n}\\ast\\epsilon_m$" for n in epsilon_totals],
        fontsize=fonts
    )
    plt.yticks(fontsize=fonts)

    plt.legend(fontsize=fonts)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=600)
    plt.show()


def plot_hv_from_csv(hv_csv_path, figure_path):
    """
    Reads a CSV with average rows and columns including:
      ['eps_total_idx', 'hv_NSGAII', 'hv_MOMA', 'hv_AMOMA']

    Then draws HV vs epsilon_total_idx.
    """
    df = pd.read_csv(hv_csv_path)
    df['eps_total_idx'] = df['eps_total_idx'].round().astype(int)

    df_avg = df[df['trial'].astype(str) == 'average'].copy()

    eps_list   = df_avg['eps_total_idx'].tolist()
    hv_nsgaii  = df_avg['hv_NSGAII'].tolist()
    hv_moma    = df_avg['hv_MOMA'].tolist()
    hv_amoma   = df_avg['hv_AMOMA'].tolist()

    plot_hv_vs_epsilon(
        eps_list,
        hv_nsgaii,
        hv_moma,
        hv_amoma,
        figure_path
    )

if __name__ == '__main__':

    out_path = os.path.join(OUTPUT_DIR, file_name)
    plot_hv_from_csv(CSV_PATH, out_path)
