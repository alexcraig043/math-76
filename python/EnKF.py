import numpy as np


class EnKF:
    def __init__(
        self,
        ensemble_size,  # Number of ensemble members
        state_dimension,  # Dimension of the state vector
        observation_dimension,  # Dimension of the observation vector
        forward_model,  # A function that propagates the state forward (e.g., f(x) = x_next)
        observation_operator,  # A matrix that maps the true state space to the observed space
        ensemble_variance,
        observation_variance,
        initial_ensemble=None,
    ):
        self.ensemble_size = ensemble_size
        self.state_dimension = state_dimension
        self.observation_dimension = observation_dimension
        self.forward_model = forward_model
        self.observation_operator = np.array(
            observation_operator
        )  # Ensure this is a NumPy array
        self.ensemble_variance = ensemble_variance
        self.observation_variance = observation_variance

        if initial_ensemble is not None:
            if initial_ensemble.shape != (state_dimension, ensemble_size):
                raise ValueError(
                    "initial_ensemble must have shape (state_dimension, ensemble_size)"
                )
            self.ensemble = initial_ensemble
        else:
            self.ensemble = np.random.randn(
                state_dimension, ensemble_size
            )  # Random initial ensemble

    def propagation_step(self):
        ensemble_noise = np.random.normal(
            0,
            np.sqrt(self.ensemble_variance),
            size=(self.state_dimension, self.ensemble_size),
        )  # Generate noise for each ensemble member
        for i in range(self.ensemble_size):
            self.ensemble[:, i] = (
                self.forward_model(self.ensemble[:, i]) + ensemble_noise[:, i]
            )  # Update each ensemble member

    def assimilation_step(self, observation):
        prior_covariance = np.cov(self.ensemble)
        kalman_gain = self.kalman_update(prior_covariance)
        perturbed_observation = observation + np.random.normal(
            0, np.sqrt(self.observation_variance), self.observation_dimension
        )
        self.ensemble = self.kalman_filter(kalman_gain, perturbed_observation)

    def kalman_update(self, prior_covariance):
        H = self.observation_operator
        R = self.observation_variance * np.eye(self.observation_dimension)
        S = H @ prior_covariance @ H.T + R
        kalman_gain = prior_covariance @ H.T @ np.linalg.inv(S)
        return kalman_gain

    def kalman_filter(self, kalman_gain, perturbed_observation):

        new_posterior_ensemble = np.zeros((self.state_dimension, self.ensemble_size))
        H = self.observation_operator
        for i in range(self.ensemble_size):
            projected_ensemble_observation = H @ self.ensemble[:, i]
            # Ensure difference is shaped correctly for matrix multiplication (column vector)
            difference = perturbed_observation.reshape(
                -1, 1
            ) - projected_ensemble_observation.reshape(-1, 1)
            correction = kalman_gain @ difference
            new_posterior_ensemble[:, i] = self.ensemble[:, i] + correction.flatten()
        return new_posterior_ensemble
