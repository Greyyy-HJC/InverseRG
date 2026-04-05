from inverserg.baselines import tree_level_coarse_beta
from inverserg.training import RGTrainingConfig, train_learned_rg


def main() -> None:
    config = RGTrainingConfig(
        fine_lattice_size=8,
        fine_beta=3.0,
        coarse_beta_init=tree_level_coarse_beta(3.0),
        n_fine_samples=16,
        n_model_samples=16,
        n_test_samples=8,
        sampler_burn_in=24,
        sampler_thin=2,
        epochs=12,
        learning_rate=3e-2,
        blocker_type="spatial",
        spatial_hidden_dim=16,
        spatial_kernel_size=3,
    )
    result = train_learned_rg(config=config)
    print("Baseline mismatch:", f"{result.baseline_mismatch:.6f}")
    print("Final mismatch:", f"{result.final_mismatch:.6f}")
    print("Optimized measurement set:", result.measurement_names)
    print("Evaluation measurement set:", result.evaluation_measurement_names)
    print("Blocker summary:", result.blocker_summary)
    print("Learned coefficients:", result.learned_action_coefficients)
    if result.test_distribution_metrics:
        print("Test-set distribution metrics:")
        for name, metrics in result.test_distribution_metrics.items():
            print(f"  {name}: MMD={metrics['mmd']:.6f}, "
                  f"blocked_mean={metrics['blocked_mean']:.4f}, "
                  f"model_mean={metrics['model_mean']:.4f}")


if __name__ == "__main__":
    main()
