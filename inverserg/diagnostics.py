import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .measurements import measurement_samples


@dataclass
class DistributionDiagnostic:
    measurement: str
    blocked_mean: float
    coarse_mean: float
    blocked_std: float
    coarse_std: float
    ks_statistic: float
    ks_critical_value: float
    consistent: bool


def _ks_statistic(x: torch.Tensor, y: torch.Tensor) -> float:
    x_sorted = torch.sort(x.flatten()).values
    y_sorted = torch.sort(y.flatten()).values
    merged = torch.sort(torch.cat([x_sorted, y_sorted])).values
    x_cdf = torch.searchsorted(x_sorted, merged, right=True).float() / max(len(x_sorted), 1)
    y_cdf = torch.searchsorted(y_sorted, merged, right=True).float() / max(len(y_sorted), 1)
    return float(torch.max(torch.abs(x_cdf - y_cdf)))


def _ks_critical_value(n_x: int, n_y: int, alpha: float) -> float:
    if alpha != 0.05:
        raise ValueError("Only alpha=0.05 is currently supported for KS diagnostics.")
    return 1.36 * ((n_x + n_y) / max(n_x * n_y, 1)) ** 0.5


def analyze_distribution_consistency(
    blocked_field: torch.Tensor,
    coarse_field: torch.Tensor,
    measurement_names: tuple[str, ...],
    ks_alpha: float = 0.05,
) -> tuple[list[DistributionDiagnostic], dict[str, torch.Tensor]]:
    blocked_samples = measurement_samples(blocked_field, measurement_names)
    coarse_samples = measurement_samples(coarse_field, measurement_names)
    diagnostics = []
    for measurement_name in measurement_names:
        blocked = blocked_samples[measurement_name]
        coarse = coarse_samples[measurement_name]
        ks = _ks_statistic(blocked, coarse)
        critical_value = _ks_critical_value(len(blocked), len(coarse), alpha=ks_alpha)
        diagnostics.append(
            DistributionDiagnostic(
                measurement=measurement_name,
                blocked_mean=float(blocked.mean()),
                coarse_mean=float(coarse.mean()),
                blocked_std=float(blocked.std(unbiased=False)),
                coarse_std=float(coarse.std(unbiased=False)),
                ks_statistic=ks,
                ks_critical_value=critical_value,
                consistent=ks <= critical_value,
            )
        )
    merged_samples = {f"blocked_{key}": value for key, value in blocked_samples.items()}
    merged_samples.update({f"coarse_{key}": value for key, value in coarse_samples.items()})
    return diagnostics, merged_samples


def save_distribution_diagnostics(
    blocked_field: torch.Tensor,
    coarse_field: torch.Tensor,
    output_dir: str | Path,
    measurement_names: tuple[str, ...],
    ks_alpha: float = 0.05,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics, measurement_data = analyze_distribution_consistency(
        blocked_field,
        coarse_field,
        measurement_names=measurement_names,
        ks_alpha=ks_alpha,
    )

    samples_path = output_dir / "measurement_samples.csv"
    fieldnames = []
    rows = max(len(values) for values in measurement_data.values())
    for key in measurement_data:
        fieldnames.append(key)
    with samples_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(rows):
            row = {}
            for key, values in measurement_data.items():
                row[key] = (
                    float(values[row_idx]) if row_idx < len(values) else ""
                )
            writer.writerow(row)

    figure_path = output_dir / "distribution_diagnostics.png"
    fig, axes = plt.subplots(len(diagnostics), 2, figsize=(10, 3 * len(diagnostics)))
    if len(diagnostics) == 1:
        axes = [axes]
    for axis_row, diagnostic in zip(axes, diagnostics):
        blocked = measurement_data[f"blocked_{diagnostic.measurement}"].numpy()
        coarse = measurement_data[f"coarse_{diagnostic.measurement}"].numpy()
        hist_ax, cdf_ax = axis_row
        hist_ax.hist(blocked, bins=12, alpha=0.6, density=True, label="blocked-fine")
        hist_ax.hist(coarse, bins=12, alpha=0.6, density=True, label="coarse HMC")
        hist_ax.set_title(f"{diagnostic.measurement} histogram")
        hist_ax.legend()

        blocked_sorted = torch.sort(torch.tensor(blocked)).values
        coarse_sorted = torch.sort(torch.tensor(coarse)).values
        blocked_cdf = torch.arange(1, len(blocked_sorted) + 1) / len(blocked_sorted)
        coarse_cdf = torch.arange(1, len(coarse_sorted) + 1) / len(coarse_sorted)
        cdf_ax.plot(blocked_sorted.numpy(), blocked_cdf.numpy(), label="blocked-fine")
        cdf_ax.plot(coarse_sorted.numpy(), coarse_cdf.numpy(), label="coarse HMC")
        cdf_ax.set_title(f"{diagnostic.measurement} CDF")
        cdf_ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    report_path = output_dir / "distribution_report.md"
    lines = [
        "# Distribution Diagnostics",
        "",
        f"Two-sample KS alpha: `{ks_alpha}`",
        "",
        "| Measurement | Blocked Mean | Coarse Mean | Blocked Std | Coarse Std | KS | KS Critical | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for diagnostic in diagnostics:
        verdict = "consistent" if diagnostic.consistent else "mismatch"
        lines.append(
            f"| {diagnostic.measurement} | {diagnostic.blocked_mean:.6f} | {diagnostic.coarse_mean:.6f} | "
            f"{diagnostic.blocked_std:.6f} | {diagnostic.coarse_std:.6f} | {diagnostic.ks_statistic:.6f} | "
            f"{diagnostic.ks_critical_value:.6f} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- Samples: `{samples_path.name}`",
            f"- Figure: `{figure_path.name}`",
        ]
    )
    report_path.write_text("\n".join(lines))
    return samples_path, figure_path, report_path
