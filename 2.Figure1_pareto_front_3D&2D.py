import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_2d_projections_from_final_csv_3x3(
    final_front_csv_path,
    out_dir,
    eps_idx_map,   # list of 3 dicts: [{"eps_total_idx": 4, "trial": 26}, ...]
    methods=("KMeans", "DP-KMeans", "GA-DP", "NSGA-II", "MOMA", "AMOMA"),
    projections=((1, 2), (3, 2), (1, 3)),   # same meaning as current code: (y_dim, x_dim)
    auto_bounds=True,
    use_true_2d_envelope=True,
    figsize=(15, 12),
    filename="pareto_front_3x3.pdf",
):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(final_front_csv_path):
        raise FileNotFoundError(f"CSV file not found: {final_front_csv_path}")

    df = pd.read_csv(final_front_csv_path)

    needed = {"trial", "method", "eps_total"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {final_front_csv_path}: {missing}")

    # def obj_label_for(d, subdf):
    #     col = f"obj_{d}"
    #     if col in subdf.columns and subdf[col].astype(str).str.len().gt(0).any():
    #         raw = subdf[col].dropna().astype(str).mode().iloc[0].strip()
    #
    #         if ":" in raw:
    #             key, direction = raw.split(":", 1)
    #             key = key.strip()
    #             direction = direction.strip().lower()
    #
    #             if direction == "max":
    #                 return f"-{key} (min)"
    #             elif direction == "min":
    #                 return f"{key} (min)"
    #         return raw
    #
    #     return f"f_{d}"

    def obj_label_for(d, subdf):
        col = f"obj_{d}"

        def pretty_name(key):
            key = key.strip().lower()
            if key == "eps":
                return r"$\epsilon$"
            elif key == "bstab":
                return "InStab"
            elif key == "nicv":
                return "NICV"
            return key.upper()

        if col in subdf.columns and subdf[col].astype(str).str.len().gt(0).any():
            raw = subdf[col].dropna().astype(str).mode().iloc[0].strip()

            if ":" in raw:
                key, direction = raw.split(":", 1)
                key = key.strip()
                direction = direction.strip().lower()

                # remove "min" everywhere; only keep sign flip for max objectives
                if direction == "max":
                    return pretty_name(key)
                elif direction == "min":
                    return pretty_name(key)

            return pretty_name(raw)

        # fallback names if obj_d column is missing
        fallback = {
            1: "NICV",
            2: r"$\epsilon$",
            3: "InStab",
        }
        return fallback.get(d, f"f_{d}")

    def normalize_vals(v, vmin, vmax):
        v = np.asarray(v, dtype=float)
        if np.isclose(vmax, vmin):
            return np.zeros_like(v, dtype=float)
        return (v - vmin) / (vmax - vmin)

    def compute_bounds(block_df, fy, fx):
        y_all = block_df[fy].to_numpy(dtype=float)
        x_all = block_df[fx].to_numpy(dtype=float)

        y_all = y_all[np.isfinite(y_all)]
        x_all = x_all[np.isfinite(x_all)]

        if len(y_all) == 0 or len(x_all) == 0:
            return None

        return {
            "ymin": float(np.min(y_all)),
            "ymax": float(np.max(y_all)),
            "xmin": float(np.min(x_all)),
            "xmax": float(np.max(x_all)),
        }

    def pareto_filter_2d_points(arr2d):
        if len(arr2d) <= 1:
            return arr2d.copy()

        keep = np.ones(len(arr2d), dtype=bool)
        for i in range(len(arr2d)):
            if not keep[i]:
                continue
            for j in range(len(arr2d)):
                if i == j or not keep[j]:
                    continue
                if np.all(arr2d[j] <= arr2d[i]) and np.any(arr2d[j] < arr2d[i]):
                    keep[i] = False
                    break
        return arr2d[keep]

    def sort_envelope_for_plot(arr2d):
        if len(arr2d) <= 1:
            return arr2d
        order = np.argsort(arr2d[:, 1])  # sort by x because arr is [y, x]
        return arr2d[order]

    markers = ['x', 's', 'D', '^', 'o', '*']
    label_map = {
        "KMeans": "KMeans",
        "DP-KMeans": "Naive-DPK",
        "GA-DP": "GA-DPK",
        "NSGA-II": "NSGA-II-DPK",
        "MOMA": "MOMA-DPK",
        "AMOMA": "AMOMA-DPK",
    }

    #
    color_map = {
        "KMeans": "#7f7f7f",  # gray
        "DP-KMeans": "#1f77b4",  # blue
        "GA-DP": "#2ca02c",  # green
        "NSGA-II": "#ff7f0e",  # orange
        "MOMA": "#9467bd",  # purple
        "AMOMA": "#d62728",  # red
    }

    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    col_titles = [r"NICV vs $\epsilon$", r"InStab vs $\epsilon$", "NICV vs InStab"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=16)

    for row_idx in range(n_rows):
        if row_idx >= len(eps_idx_map):
            for ax in axes[row_idx]:
                ax.axis("off")
            continue

        cfg = eps_idx_map[row_idx]
        trial = cfg["trial"]
        eps_total_idx_target = cfg["eps_total_idx"]

        block = df[
            (df["trial"].astype(str) == str(trial)) &
            (pd.to_numeric(df["eps_total_idx"], errors="coerce").fillna(-999).astype(int) == int(eps_total_idx_target))
        ].copy()

        if block.empty:
            for ax in axes[row_idx]:
                ax.set_title(f"eps_total_idx={eps_total_idx_target}\n<no data>")
                ax.axis("off")
            continue

        eps_total_val = block["eps_total"].dropna().iloc[0] if block["eps_total"].notna().any() else None

        # put one row label on the far left
        if eps_total_val is not None:
            row_label = rf"$\epsilon_{{total}}={eps_total_idx_target}*\epsilon_m$"
        else:
            row_label = rf"$\epsilon_{{total}}$ idx={eps_total_idx_target}"

        axes[row_idx, 0].text(
            -0.2, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=16
        )

        for col_idx, (y_dim, x_dim) in enumerate(projections):
            ax = axes[row_idx, col_idx]

            fy = f"f_{y_dim}"
            fx = f"f_{x_dim}"

            if fy not in block.columns or fx not in block.columns:
                ax.set_title(f"Missing {fy}/{fx}")
                ax.axis("off")
                continue

            bounds = compute_bounds(block, fy, fx)
            if bounds is None:
                ax.set_title("No finite data")
                ax.axis("off")
                continue

            plotted_any = False

            for i, method in enumerate(methods):
                gm = block[block["method"] == method].copy()
                if gm.empty:
                    continue

                arr = gm[[fy, fx]].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr).all(axis=1)]
                if len(arr) == 0:
                    continue

                if use_true_2d_envelope:
                    arr = pareto_filter_2d_points(arr)

                if len(arr) == 0:
                    continue

                arr = sort_envelope_for_plot(arr)

                y_norm = normalize_vals(arr[:, 0], bounds["ymin"], bounds["ymax"])
                x_norm = normalize_vals(arr[:, 1], bounds["xmin"], bounds["xmax"])

                plot_kwargs = dict(
                    linestyle='-',
                    marker=markers[i % len(markers)],
                    markersize=5,
                    label=label_map.get(method, method)
                )

                if method in color_map:
                    plot_kwargs["color"] = color_map[method]

                ax.plot(x_norm, y_norm, **plot_kwargs)
                plotted_any = True

            ylab = obj_label_for(y_dim, block)
            xlab = obj_label_for(x_dim, block)

            # no seed in title anymore
            ax.set_xlabel(f"Normalized {xlab}")
            ax.set_ylabel(f"Normalized {ylab}")
            ax.grid(True)

            if not plotted_any:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    handles, labels = None, None
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(labels),
            frameon=True,
            fontsize=16,
        )

    # leave room on the left for row labels
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()
    print(f"Saved combined figure to {out_path}")

def plot_3d_fronts_static_from_final_csv_1x3(
    final_front_csv_path,
    out_dir,
    eps_idx_map,
    methods=("KMeans", "DP-KMeans", "GA-DP", "NSGA-II", "MOMA", "AMOMA"),
    dims=(1, 2, 3),
    normalized=True,
    norm_scope="trial_eps",
    max_points_per_method=None,
    dpi=300,
    figsize=(17, 5.4),
    filename="pareto_front_3d_1x3.png",
    elev=22,
    azim=-55,
):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(final_front_csv_path):
        raise FileNotFoundError(f"CSV file not found: {final_front_csv_path}")

    df = pd.read_csv(final_front_csv_path)

    needed = {"trial", "method", "eps_total"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {final_front_csv_path}: {missing}")

    fcols = [f"f_{d}" for d in dims]
    missing_f = [c for c in fcols if c not in df.columns]
    if missing_f:
        raise ValueError(f"Missing objective columns: {missing_f}")

    df = df[df["method"].isin(methods)].copy()
    df = df.dropna(subset=["eps_total"] + fcols)

    def obj_label_for(d, subdf):
        col = f"obj_{d}"

        def pretty_name(key):
            key = str(key).strip().lower()
            if key == "eps":
                return r"$\epsilon$"
            elif key == "bstab":
                return "InStab"
            elif key == "nicv":
                return "NICV"
            return key.upper()

        if col in subdf.columns and subdf[col].astype(str).str.len().gt(0).any():
            raw = subdf[col].dropna().astype(str).mode().iloc[0].strip()
            if ":" in raw:
                key, _direction = raw.split(":", 1)
                return pretty_name(key)
            return pretty_name(raw)

        fallback = {
            1: "NICV",
            2: r"$\epsilon$",
            3: "InStab",
        }
        return fallback.get(d, f"f_{d}")

    markers = ['x', 's', 'D', '^', 'o', '*']

    color_map = {
        "KMeans": "#7f7f7f",
        "DP-KMeans": "#1f77b4",
        "GA-DP": "#2ca02c",
        "NSGA-II": "#ff7f0e",
        "MOMA": "#9467bd",
        "AMOMA": "#d62728",
    }

    label_map = {
        "KMeans": "KMeans",
        "DP-KMeans": "Naive-DPK",
        "GA-DP": "GA-DPK",
        "NSGA-II": "NSGA-II-DPK",
        "MOMA": "MOMA-DPK",
        "AMOMA": "AMOMA-DPK",
    }

    global_norm_cache = {}
    if normalized and norm_scope == "trial_global":
        trial_values = sorted(set(str(x) for x in df["trial"].astype(str).unique()))
        for tr in trial_values:
            dft = df[df["trial"].astype(str) == tr].copy()
            if dft.empty:
                continue
            F_all = dft[[f"f_{dims[0]}", f"f_{dims[1]}", f"f_{dims[2]}"]].to_numpy(dtype=float)
            ideal = np.nanmin(F_all, axis=0)
            nadir = np.nanmax(F_all, axis=0)
            den = np.where((nadir - ideal) == 0, 1.0, (nadir - ideal))
            global_norm_cache[tr] = (ideal, den)

    def normalize_xyz(XYZ, ideal, den):
        XYZn = (XYZ - ideal) / den
        return np.clip(XYZn, 0.0, 1.0)

    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    for col_idx in range(3):
        ax = axes[col_idx]

        if col_idx >= len(eps_idx_map):
            ax.set_axis_off()
            continue

        cfg = eps_idx_map[col_idx]
        trial = cfg["trial"]
        eps_total_idx_target = cfg["eps_total_idx"]

        block = df[
            (df["trial"].astype(str) == str(trial)) &
            (pd.to_numeric(df["eps_total_idx"], errors="coerce").fillna(-999).astype(int) == int(eps_total_idx_target))
        ].copy()

        if block.empty:
            ax.set_title(rf"$\epsilon_{{total}}={eps_total_idx_target}\epsilon_m$", fontsize=12, pad=2)
            continue

        xlab = obj_label_for(dims[0], block)
        ylab = obj_label_for(dims[1], block)
        zlab = obj_label_for(dims[2], block)

        if normalized:
            if norm_scope == "trial_global":
                if str(trial) not in global_norm_cache:
                    ax.set_title(rf"$\epsilon_{{total}}={eps_total_idx_target}\epsilon_m$", fontsize=12, pad=2)
                    continue
                ideal, den = global_norm_cache[str(trial)]
            else:
                F_all = block[[f"f_{dims[0]}", f"f_{dims[1]}", f"f_{dims[2]}"]].to_numpy(dtype=float)
                ideal = np.nanmin(F_all, axis=0)
                nadir = np.nanmax(F_all, axis=0)
                den = np.where((nadir - ideal) == 0, 1.0, (nadir - ideal))

        plotted_any = False

        for mi, method in enumerate(methods):
            gm = block[block["method"] == method].copy()
            if gm.empty:
                continue

            if max_points_per_method is not None and len(gm) > max_points_per_method:
                gm = gm.sample(n=max_points_per_method, random_state=0)

            XYZ = gm[[f"f_{dims[0]}", f"f_{dims[1]}", f"f_{dims[2]}"]].to_numpy(dtype=float)

            if normalized:
                XYZ = normalize_xyz(XYZ, ideal, den)

            X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

            scatter_kwargs = dict(
                marker=markers[mi % len(markers)],
                label=label_map.get(method, method),
                color=color_map.get(method, None),
                s=30,
                depthshade=False,
                alpha=0.95,
                edgecolors='none',
            )

            if method == "AMOMA":
                scatter_kwargs["s"] = 48

            ax.scatter(X, Y, Z, **scatter_kwargs)
            plotted_any = True

        if normalized:
            ax.set_xlabel(f"Normalized {xlab}", labelpad=6, fontsize=11)
            ax.set_ylabel(f"Normalized {ylab}", labelpad=6, fontsize=11)
            ax.set_zlabel(f"Normalized {zlab}", labelpad=6, fontsize=11)
        else:
            ax.set_xlabel(xlab, labelpad=6, fontsize=11)
            ax.set_ylabel(ylab, labelpad=6, fontsize=11)
            ax.set_zlabel(zlab, labelpad=6, fontsize=11)

        # column label closer to plot
        ax.set_title(rf"$\epsilon_{{total}}={eps_total_idx_target}\epsilon_m$", fontsize=16)

        # better 3D viewing angle
        ax.view_init(elev=elev, azim=azim)

        # publication-style box shape
        ax.set_box_aspect((1, 1, 0.9))

        # cleaner panes
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))

        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 1.0))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 1.0))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 1.0))

        # lighter grid
        ax.grid(True)
        try:
            ax.xaxis._axinfo["grid"]['linewidth'] = 0.6
            ax.yaxis._axinfo["grid"]['linewidth'] = 0.6
            ax.zaxis._axinfo["grid"]['linewidth'] = 0.6

            ax.xaxis._axinfo["grid"]['linestyle'] = '-'
            ax.yaxis._axinfo["grid"]['linestyle'] = '-'
            ax.zaxis._axinfo["grid"]['linestyle'] = '-'

            ax.xaxis._axinfo["grid"]['color'] = (0.85, 0.85, 0.85, 1.0)
            ax.yaxis._axinfo["grid"]['color'] = (0.85, 0.85, 0.85, 1.0)
            ax.zaxis._axinfo["grid"]['color'] = (0.85, 0.85, 0.85, 1.0)
        except Exception:
            pass

        # smaller ticks for cleaner paper look
        ax.tick_params(axis='both', which='major', labelsize=9, pad=1)
        ax.tick_params(axis='z', which='major', labelsize=9, pad=1)

        if normalized:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_zticks([0.0, 0.5, 1.0])

        if not plotted_any:
            ax.text2D(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    handles, labels = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(labels),
            frameon=True,
            fontsize=16,
        )

    # pull subplots upward and reduce blank space above titles
    # fig.subplots_adjust(
    #     left=0.0,
    #     right=0.95,
    #     bottom=0.0,
    #     top=1,
    #     wspace=0.00
    # )

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Saved combined 3D figure to {out_path}")

def main():
    heart_dataset = 0

    if heart_dataset:
        csv_path = "../Co_work_Samsad_WISE2025-extension_report/20260330_034509-FormalTest-B6-Frac0.6-heart_Nm20_G50_Pc0.9_Pm0.1_Scaler10_HL5_extension_v1.6/spec_nicv_eps_bstab/final_front_points.csv"
        out_dir = "../Co_work_Samsad_WISE2025-extension_report/figures/"
        eps_idx_map = [
            {"eps_total_idx": 3, "trial": 12},
            {"eps_total_idx": 6, "trial": 4},
            {"eps_total_idx": 9, "trial": 1},
        ]
        file_name_2d = "heart_data_pareto_fronts_3x3.pdf"
        file_name_3d = "heart_data_3Dpareto_fronts_1x3.pdf"


    else:
        csv_path = "../Co_work_Samsad_WISE2025-extension_report/20260324_121738-FormalTest-B6-Frac0.6-Diabetes_Nm20_G50_Pc0.9_Pm0.1_Scaler10_extension_v1.6/spec_nicv_eps_bstab/final_front_points.csv"
        out_dir = "../Co_work_Samsad_WISE2025-extension_report/figures/"
        eps_idx_map = [
            {"eps_total_idx": 3, "trial": 16},
            {"eps_total_idx": 6, "trial": 16},
            {"eps_total_idx": 9, "trial": 3},
        ]
        file_name_2d = "diabetes_data_pareto_fronts_3x3.pdf"
        file_name_3d = "diabetes_data_3Dpareto_fronts_1x3.pdf"

    plot_2d_projections_from_final_csv_3x3(
        final_front_csv_path=csv_path,
        out_dir=out_dir,
        eps_idx_map=eps_idx_map,
        methods=("KMeans", "DP-KMeans", "GA-DP", "NSGA-II", "MOMA","AMOMA"),
        projections=((1, 3), (2, 3), (1, 2)),
        auto_bounds=True,
        use_true_2d_envelope=True,
        figsize=(15, 12),
        filename=file_name_2d,
    )

    plot_3d_fronts_static_from_final_csv_1x3(
        final_front_csv_path=csv_path,
        out_dir=out_dir,
        eps_idx_map=eps_idx_map,
        methods=("KMeans", "DP-KMeans", "GA-DP", "NSGA-II", "MOMA", "AMOMA"),
        dims=(1, 2, 3),
        normalized=True,
        norm_scope="trial_eps",
        max_points_per_method=None,
        dpi=600,
        figsize=(18, 6),
        filename=file_name_3d,
        elev=22,
        azim=-55,
    )




if __name__ == "__main__":
    main()