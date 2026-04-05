from inverserg.baselines import tree_level_coarse_beta
from inverserg.actions import LocalWilsonLoopAction
from inverserg.blocking import FixedGaugeCovariantBlocker
from inverserg.hmc import HMCU1Sampler
from inverserg.measurements import summarize_observables
from inverserg.training import RGTrainingConfig, generate_fine_ensemble


def squared_mismatch(blocked_summary: dict[str, float], coarse_summary: dict[str, float]) -> dict[str, float]:
    report = {}
    for key, blocked_value in blocked_summary.items():
        if key not in coarse_summary:
            continue
        report[key] = (blocked_value - coarse_summary[key]) ** 2
    return report


def main() -> None:
    config = RGTrainingConfig(fine_lattice_size=8, fine_beta=4.0, n_fine_samples=8, sampler_burn_in=24, sampler_thin=2)
    fine_configs = generate_fine_ensemble(config)
    coarse_beta = tree_level_coarse_beta(config.fine_beta)
    coarse_configs = FixedGaugeCovariantBlocker()(fine_configs)
    coarse_action = LocalWilsonLoopAction.wilson(coarse_beta)
    coarse_sampler = HMCU1Sampler(
        lattice_size=config.fine_lattice_size // 2,
        action=coarse_action,
        n_steps=config.hmc_steps,
        step_size=config.hmc_step_size,
        device=config.device,
    )
    coarse_ensemble, acceptance_rate, _ = coarse_sampler.sample(
        n_samples=config.n_fine_samples,
        burn_in=config.sampler_burn_in,
        thin=config.sampler_thin,
    )
    blocked_summary = summarize_observables(coarse_configs)
    coarse_summary = summarize_observables(coarse_ensemble)
    mismatch_report = squared_mismatch(blocked_summary, coarse_summary)

    print("Baseline hypothesis beta_c ~= beta_f / 4")
    print("fine_beta:", config.fine_beta)
    print("coarse_beta_init:", coarse_beta)
    print("blocked-fine observables:", blocked_summary)
    print("coarse Wilson observables:", coarse_summary)
    print("observable mismatch report:", mismatch_report)
    print("coarse acceptance rate:", acceptance_rate)


if __name__ == "__main__":
    main()
