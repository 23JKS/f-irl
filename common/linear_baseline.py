"""
Linear Feature Baseline for TRPO
Simple linear regression baseline to reduce variance
"""
import numpy as np
import torch


class LinearFeatureBaseline:
    """
    Linear baseline using polynomial features
    Simpler and more stable than neural network baseline
    """
    def __init__(self, obs_dim, reg_coeff=1e-5):
        self.obs_dim = obs_dim
        self.reg_coeff = reg_coeff
        self.coeffs = None
        
    def _features(self, obs):
        """
        Compute polynomial features: [1, s, s^2]
        """
        obs = np.array(obs)
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        
        # Features: constant + linear + quadratic
        ones = np.ones((obs.shape[0], 1))
        features = np.concatenate([ones, obs, obs ** 2], axis=1)
        return features
    
    def fit(self, observations, returns):
        """
        Fit linear regression: V(s) = features(s) @ coeffs
        """
        observations = np.array(observations)
        returns = np.array(returns)
        
        # Compute features
        features = self._features(observations)
        
        # Ridge regression: (X^T X + λI)^{-1} X^T y
        featmat = features
        reg_coeff = self.reg_coeff
        
        # Add regularization
        XT_y = featmat.T @ returns
        XT_X = featmat.T @ featmat
        
        # Solve with regularization
        for _ in range(5):
            try:
                self.coeffs = np.linalg.solve(
                    XT_X + reg_coeff * np.eye(featmat.shape[1]),
                    XT_y
                )
                break
            except np.linalg.LinAlgError:
                reg_coeff *= 10
        else:
            # If still fails, use pseudo-inverse
            self.coeffs = np.linalg.lstsq(featmat, returns, rcond=None)[0]
    
    def predict(self, observations):
        """
        Predict values for observations
        """
        if self.coeffs is None:
            return np.zeros(len(observations))
        
        features = self._features(observations)
        return features @ self.coeffs
    
    def __call__(self, observations):
        """
        Make it callable like a function
        """
        return self.predict(observations)
