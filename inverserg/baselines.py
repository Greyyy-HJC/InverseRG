def tree_level_coarse_beta(fine_beta: float, blocking_factor: int = 2, dimensions: int = 2) -> float:
    if dimensions != 2:
        raise ValueError("The current baseline helper is only calibrated for 2D U(1).")
    return fine_beta / float(blocking_factor**2)
