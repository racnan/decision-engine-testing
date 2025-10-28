"""
Bayesian Optimization engine for hyperparameter tuning.

This module implements Bayesian Optimization using Gaussian Processes with
Matérn kernel and Expected Improvement acquisition function for efficient
hyperparameter search.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from config import HyperparameterConfig


class BayesianOptimizer:
    """
    Bayesian Optimization engine using Gaussian Processes.
    
    Features:
    - Gaussian Process with Matérn kernel (ν=2.5)
    - Expected Improvement acquisition function
    - Parameter normalization to [0,1] range
    - Multi-objective optimization support
    """
    
    def __init__(self, config: HyperparameterConfig):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            config: Hyperparameter configuration object
        """
        self.config = config
        self.param_names = config.get_parameter_names()
        self.param_bounds = config.get_parameter_bounds()
        self.param_types = config.get_parameter_types()
        self.log_scale_flags = config.get_log_scale_flags()
        
        # Normalized bounds [0, 1] for all parameters
        self.normalized_bounds = [(0.0, 1.0) for _ in self.param_names]
        self.n_params = len(self.param_names)
        
        # Initialize Gaussian Process
        self._initialize_gp()
        
        # Storage for optimization history
        self.X_normalized = []  # Normalized parameter values
        self.y = []  # Objective values
        self.best_params = None
        self.best_score = -np.inf
        self.best_iteration = 0
        
    def _initialize_gp(self):
        """Initialize the Gaussian Process with Matérn kernel."""
        # Matérn kernel with ν=2.5 (smoothness parameter)
        matern_kernel = Matern(length_scale=np.ones(self.n_params), 
                              length_scale_bounds=(1e-3, 1e3),
                              nu=2.5)
        
        # Constant kernel for signal variance
        constant_kernel = ConstantKernel(constant_value=1.0,
                                       constant_value_bounds=(1e-3, 1e3))
        
        # White kernel for noise
        white_kernel = WhiteKernel(noise_level=1e-6,
                                 noise_level_bounds=(1e-10, 1e-1))
        
        # Combined kernel
        kernel = constant_kernel * matern_kernel + white_kernel
        
        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
    
    def _normalize_parameters(self, params: np.ndarray) -> np.ndarray:
        """
        Normalize parameters to [0,1] range.
        
        Args:
            params: Parameter values in original scale
            
        Returns:
            Normalized parameter values
        """
        normalized = np.zeros_like(params)
        
        for i, (param, (low, high)) in enumerate(zip(params, self.param_bounds)):
            if self.log_scale_flags[i]:
                # Log scale normalization
                log_param = np.log(param)
                log_low = np.log(low)
                log_high = np.log(high)
                normalized[i] = (log_param - log_low) / (log_high - log_low)
            else:
                # Linear scale normalization
                normalized[i] = (param - low) / (high - low)
        
        return np.clip(normalized, 0.0, 1.0)
    
    def _denormalize_parameters(self, normalized_params: np.ndarray) -> np.ndarray:
        """
        Denormalize parameters from [0,1] range to original scale.
        
        Args:
            normalized_params: Normalized parameter values
            
        Returns:
            Parameter values in original scale
        """
        params = np.zeros_like(normalized_params)
        
        for i, (norm_param, (low, high)) in enumerate(zip(normalized_params, self.param_bounds)):
            if self.log_scale_flags[i]:
                # Log scale denormalization
                log_low = np.log(low)
                log_high = np.log(high)
                log_param = log_low + norm_param * (log_high - log_low)
                params[i] = np.exp(log_param)
            else:
                # Linear scale denormalization
                params[i] = low + norm_param * (high - low)
        
        return params
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.
        
        Args:
            X: Normalized parameter values to evaluate
            xi: Exploration parameter
            
        Returns:
            Expected improvement values
        """
        if len(self.X_normalized) == 0:
            # No data yet, return zeros
            return np.zeros(len(X))
        
        # Reshape for GP prediction
        X = X.reshape(-1, self.n_params)
        
        # Predict mean and standard deviation
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Current best score
        best_score = np.max(self.y)
        
        # Expected improvement
        with np.errstate(divide='warn', invalid='warn'):
            imp = mu - best_score - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _acquisition_function(self, X: np.ndarray) -> float:
        """
        Acquisition function for optimization (negative EI for minimization).
        
        Args:
            X: Normalized parameter values
            
        Returns:
            Negative expected improvement (for minimization)
        """
        ei = self._expected_improvement(X.reshape(1, -1))
        return -ei[0]  # Negative for minimization
    
    def suggest_next_parameters(self, n_random_starts: int = 25) -> np.ndarray:
        """
        Suggest next parameters to evaluate using acquisition function.
        
        Args:
            n_random_starts: Number of random starting points for optimization
            
        Returns:
            Suggested parameters in original scale
        """
        if len(self.X_normalized) < 2:
            # Not enough data for GP, return random point
            return self._suggest_random_parameters()
        
        # Fit GP to current data
        X_array = np.array(self.X_normalized)
        y_array = np.array(self.y)
        
        try:
            self.gp.fit(X_array, y_array)
        except Exception as e:
            print(f"Warning: GP fitting failed, using random suggestion: {e}")
            return self._suggest_random_parameters()
        
        # Optimize acquisition function
        best_x = None
        best_ei = -np.inf
        
        # Multiple random restarts
        for _ in range(n_random_starts):
            # Random starting point
            x0 = np.random.uniform(0, 1, self.n_params)
            
            # Optimize
            result = minimize(
                self._acquisition_function,
                x0=x0,
                bounds=self.normalized_bounds,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if result.success:
                x = result.x
                ei = -result.fun  # Convert back to positive
                
                if ei > best_ei:
                    best_ei = ei
                    best_x = x
        
        if best_x is None:
            # Optimization failed, return random point
            return self._suggest_random_parameters()
        
        # Denormalize and return
        return self._denormalize_parameters(best_x)
    
    def _suggest_random_parameters(self) -> np.ndarray:
        """
        Suggest random parameters within bounds.
        
        Returns:
            Random parameters in original scale
        """
        params = np.zeros(self.n_params)
        
        for i, (low, high) in enumerate(self.param_bounds):
            if self.log_scale_flags[i]:
                # Log scale sampling
                log_low = np.log(low)
                log_high = np.log(high)
                log_param = np.random.uniform(log_low, log_high)
                params[i] = np.exp(log_param)
            else:
                # Linear scale sampling
                params[i] = np.random.uniform(low, high)
        
        return params
    
    def observe(self, params: np.ndarray, score: float):
        """
        Observe the result of evaluating parameters.
        
        Args:
            params: Parameters that were evaluated
            score: Objective score achieved
        """
        # Normalize parameters
        normalized_params = self._normalize_parameters(params)
        
        # Add to history
        self.X_normalized.append(normalized_params)
        self.y.append(score)
        
        # Update best if improved
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.best_iteration = len(self.y) - 1
    
    def get_best_parameters(self) -> tuple:
        """
        Get the best parameters found so far.
        
        Returns:
            Tuple of (best_params, best_score, best_iteration)
        """
        return self.best_params, self.best_score, self.best_iteration
    
    def get_optimization_history(self) -> dict:
        """
        Get the complete optimization history.
        
        Returns:
            Dictionary with optimization history
        """
        return {
            'parameters': [self._denormalize_parameters(x) for x in self.X_normalized],
            'scores': self.y.copy(),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'best_iteration': self.best_iteration,
            'n_evaluations': len(self.y)
        }
    
    def predict_performance(self, params: np.ndarray) -> tuple:
        """
        Predict performance for given parameters using the GP model.
        
        Args:
            params: Parameters to predict performance for
            
        Returns:
            Tuple of (predicted_mean, predicted_std)
        """
        if len(self.X_normalized) < 2:
            # Not enough data for prediction
            return 0.0, 1.0
        
        # Normalize parameters
        normalized_params = self._normalize_parameters(params).reshape(1, -1)
        
        # Fit GP if not already fitted
        if not hasattr(self.gp, 'X_train_'):
            X_array = np.array(self.X_normalized)
            y_array = np.array(self.y)
            self.gp.fit(X_array, y_array)
        
        # Predict
        mean, std = self.gp.predict(normalized_params, return_std=True)
        return mean[0], std[0]
    
    def calculate_improvement_potential(self, params: np.ndarray) -> float:
        """
        Calculate the potential improvement over current best for given parameters.
        
        Args:
            params: Parameters to evaluate
            
        Returns:
            Expected improvement over current best
        """
        if len(self.X_normalized) < 2:
            return 0.0
        
        # Normalize parameters
        normalized_params = self._normalize_parameters(params)
        
        # Calculate expected improvement
        ei = self._expected_improvement(normalized_params.reshape(1, -1))
        return ei[0]
    
    def __str__(self) -> str:
        """String representation of the optimizer."""
        return f"BayesianOptimizer(params={self.n_params}, " \
               f"evaluations={len(self.y)}, best_score={self.best_score:.4f})"
