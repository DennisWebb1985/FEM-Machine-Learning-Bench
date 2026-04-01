from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0) -> None:
        self.patience = patience
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> tuple[bool, bool]:
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True, False

        self.counter += 1
        should_stop = self.counter >= self.patience
        return False, should_stop
