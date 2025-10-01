import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.attack.attack import AttackBase

LinearRegressionLike = Ridge | LinearRegression | Lasso

class LinearRegressionAttack(AttackBase):
    def __init__(self, model: LinearRegressionLike) -> None:
        super().__init__()
        self.model = model

    def execute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        self.model.fit(X, Y)
        return self.model.coef_.T
    
    def __str__(self) -> str:
        return str(self.model)