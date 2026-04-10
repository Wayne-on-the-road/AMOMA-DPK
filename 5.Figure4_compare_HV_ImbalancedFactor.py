import os
import pandas as pd
import matplotlib.pyplot as plt


heart_dataset = 1

if heart_dataset:
    # heart data
    SCALER_CSV = {
        1: "../Co_work_Samsad_WISE2025-extension_report/20260330_123341-FormalTest-B6-Frac0.6-heart_Nm20_G50_Pc0.9_Pm0.1_Scaler1_HL5_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv",
        5: "../Co_work_Samsad_WISE2025-extension_report/20260330_142920-FormalTest-B6-Frac0.6-heart_Nm20_G50_Pc0.9_Pm0.1_Scaler5_HL5_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv",
        10: "../Co_work_Samsad_WISE2025-extension_report/20260330_162342-FormalTest-B6-Frac0.6-heart_Nm20_G50_Pc0.9_Pm0.1_Scaler10_HL5_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv",
    }

    OUTPUT_DIR = "../Co_work_Samsad_WISE2025-extension_report/figures/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUT_FILE = os.path.join(OUTPUT_DIR, "heart_data_hv_vs_epsilon_scalers.pdf")

else: # diabetes
    SCALER_CSV = {
        1: "../Co_work_Samsad_WISE2025-extension_report/20260324_140340-FormalTest-B6-Frac0.6-Diabetes_Nm20_G50_Pc0.9_Pm0.1_Scaler1_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv",
        5: "../Co_work_Samsad_WISE2025-extension_report/20260324_122405-FormalTest-B6-Frac0.6-Diabetes_Nm20_G50_Pc0.9_Pm0.1_Scaler5_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv",
        10: "../Co_work_Samsad_WISE2025-extension_report/20260324_121738-FormalTest-B6-Frac0.6-Diabetes_Nm20_G50_Pc0.9_Pm0.1_Scaler10_extension_v1.6/spec_nicv_eps_bstab/final_hv_summary.csv",
    }

    OUTPUT_DIR = "../Co_work_Samsad_WISE2025-extension_report/figures/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUT_FILE = os.path.join(OUTPUT_DIR, "diabetes_data_hv_vs_epsilon_scalers.pdf")


def plot_hv_vs_epsilon_multi(scaler_csv_map, output_path):
    """
    scaler_csv_map: dict[int, str]
        mapping each scaler value -> path to its final_hv_summary.csv
    output_path: str
        where to write the combined figure
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    fonts = 14
    plt.figure(figsize=(8, 6))

    # one color per scaler
    colors = plt.cm.tab10.colors

    # one style per method
    method_styles = {
        # "MOMA":  {"col": "hv_MOMA",  "linestyle": "--", "marker": "o"},
        "AMOMA": {"col": "hv_AMOMA", "linestyle": "-.",  "marker": "s"},
    }

    eps_list = []

    # sort scalers so factor=1 is definitely included first and legend is ordered
    scaler_items = sorted(scaler_csv_map.items(), key=lambda x: x[0])

    for i, (scaler, csv_path) in enumerate(scaler_items):
        df = pd.read_csv(csv_path)

        # keep only average rows
        df_avg = df[df["trial"] == "average"].copy()
        if df_avg.empty:
            continue

        df_avg["eps_total_idx"] = df_avg["eps_total_idx"].round().astype(int)
        df_avg = df_avg.sort_values("eps_total_idx")

        eps_list = df_avg["eps_total_idx"].tolist()
        color = colors[i % len(colors)]

        for method_name, style in method_styles.items():
            col = style["col"]
            if col not in df_avg.columns:
                continue

            y = df_avg[col].tolist()

            plt.plot(
                eps_list,
                y,
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=color,
                linewidth=2,
                markersize=6,
                markerfacecolor="none",   # helps reveal overlapping curves
                markeredgewidth=1.5,
                alpha=0.9,
                label=f"{method_name} (factor={scaler})"
            )

    plt.xlabel(r'$\epsilon_{\mathrm{total}}$', fontsize=fonts + 2)
    plt.ylabel('Avg Hypervolume', fontsize=fonts + 2)

    if eps_list:
        plt.xticks(
            eps_list,
            [f"${n}\\ast\\epsilon_m$" for n in eps_list],
            fontsize=fonts
        )

    plt.yticks(fontsize=fonts)
    plt.legend(fontsize=fonts - 2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


if __name__ == "__main__":
    plot_hv_vs_epsilon_multi(SCALER_CSV, OUT_FILE)