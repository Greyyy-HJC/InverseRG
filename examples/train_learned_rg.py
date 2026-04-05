from inverserg.baselines import tree_level_coarse_beta
from inverserg.training import RGTrainingConfig, train_learned_rg


def main() -> None:
    config = RGTrainingConfig(
        fine_lattice_size=8,
        fine_beta=3.0,
        coarse_beta_init=tree_level_coarse_beta(3.0),
        n_fine_samples=16,
        n_model_samples=16,
        sampler_burn_in=24,
        sampler_thin=2,
        epochs=12,
        learning_rate=3e-2,
    )
    result = train_learned_rg(config=config)
    print("Baseline mismatch:", f"{result.baseline_mismatch:.6f}")
    print("Final mismatch:", f"{result.final_mismatch:.6f}")
    print("Learned blocking weights:", result.learned_path_weights)
    print("Learned coefficients:", result.learned_action_coefficients)


if __name__ == "__main__":
    main()
