from pathlib import Path

from inverserg.actions import LocalWilsonLoopAction
from inverserg.baselines import tree_level_coarse_beta
from inverserg.blocking import FixedGaugeCovariantBlocker
from inverserg.diagnostics import save_distribution_diagnostics
from inverserg.hmc import HMCU1Sampler
from inverserg.training import RGTrainingConfig, generate_fine_ensemble


def main() -> None:
    config = RGTrainingConfig(
        fine_lattice_size=8,
        fine_beta=4.0,
        n_fine_samples=32,
        n_model_samples=32,
        sampler_burn_in=24,
        sampler_thin=2,
    )
    measurement_names = ("plaquette", "rectangle_x", "rectangle_y", "wilson_2x2", "topological_charge")
    fine_configs = generate_fine_ensemble(config)
    blocked_configs = FixedGaugeCovariantBlocker()(fine_configs)

    coarse_beta = tree_level_coarse_beta(config.fine_beta)
    coarse_action = LocalWilsonLoopAction.wilson(coarse_beta)
    coarse_sampler = HMCU1Sampler(
        lattice_size=config.fine_lattice_size // 2,
        action=coarse_action,
        n_steps=config.hmc_steps,
        step_size=config.hmc_step_size,
        device=config.device,
    )
    coarse_configs, acceptance_rate, _ = coarse_sampler.sample(
        n_samples=config.n_model_samples,
        burn_in=config.sampler_burn_in,
        thin=config.sampler_thin,
    )

    output_dir = Path("artifacts") / "distribution_diagnostics"
    samples_path, figure_path, report_path = save_distribution_diagnostics(
        blocked_configs,
        coarse_configs,
        output_dir=output_dir,
        measurement_names=measurement_names,
    )

    print("coarse acceptance rate:", acceptance_rate)
    print("samples:", samples_path)
    print("figure:", figure_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
