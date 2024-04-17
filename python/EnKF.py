import numpy as np


class EnKF:
    def __init__(
        self,
        F,  # A function that propagates the state forward (e.g., f(x) = x_next)
        H,  # A matrix that maps the true state space to the observed space
        Qsqrt,  # The square root of the state noise covariance matrix
        Rsqrt,  # The square root of the observation noise covariance matrix
        seed,  # Seed for random number generator
        n_members=10,  # Number of ensemble members
        initialization=None,  # Initial ensemble members
    ):
        # Bind attributes
        self.F = F
        self.H = H
        self.Qsqrt = Qsqrt  # State noise (also Sigma in notation)
        self.Rsqrt = Rsqrt  # Measurement noise (also Gamma in notation)
        self.n = self.H.shape[1]  # Infer state space dimension
        self.m = self.H.shape[0]  # Measurement space dimension
        self.n_members = n_members

        if seed is not None:
            np.random.seed(seed)

        # Are we adding state space noise?
        if self.Qsqrt is None:
            self.with_state_noise = False
        else:
            self.with_state_noise = True

        # Handle ensemble initialization
        if (initialization is not None) and (n_members is not None):
            assert initialization.shape == (
                self.n_members,
                self.n,
            ), "invalid shape for initialization."
            self.members = initialization
        elif (initialization is None) and (n_members is not None):
            # Give random intialization
            self.members = np.random.normal(size=(self.n_members, self.n))
        else:
            raise ValueError("Must give n_members!")

    def advance_ensemble(self, v):
        """Update the entire model based on the new observation y."""

        # Execute propagation step
        self.propagation_step()

        # Execute analysis step
        self.analysis_step(v)

        return None

    def propagation_step(self):
        """Carries out the propagation step."""

        # Generate the blank forecast ensemble
        self.members_pred = np.zeros_like(self.members)

        # For each ensemble member
        for j in range(self.n_members):
            # Apply the forward model to the ensemble member
            if self.with_state_noise:
                # With state noise
                self.members_pred[j, :] = self.F(self.members[j, :]) + (
                    self.Qsqrt @ np.random.normal(size=self.n)
                )
            else:
                # Without state noise
                self.members_pred[j, :] = self.F(self.members[j, :])

        # Get the mean prediction of the forecasted ensemble
        # Note: taking the average of each state entry across all members
        self.mean_pred = np.mean(self.members_pred, axis=0)

        # Get the covariance of the forecasted ensemble (each row is a state variable, so we need to transpose the matrix)
        self.cov_pred = np.cov(self.members_pred.T)

        return None

    def analysis_step(self, v):
        """Carries out the analysis step."""

        # Generate the blank analysis ensemble
        v_perturbed = np.zeros((self.n_members, self.m))

        # For each ensemble member, generate a perturbed observation
        for j in range(self.n_members):
            v_perturbed[j, :] = v + (self.Rsqrt @ np.random.normal(size=self.m))

        # Define the matrix that needs to be inverted
        # Note: self.Rsqrt @ self.Rsqrt.T generates the observation noise covariance matrix
        B = self.H @ self.cov_pred @ self.H.T + self.Rsqrt @ self.Rsqrt.T

        # Invert the matrix
        B_inv = np.linalg.inv(B)

        # For each ensemble member
        for j in range(self.n_members):
            # Compute the Kalman gain
            K = self.cov_pred @ self.H.T @ B_inv

            # Compute the analysis state (u_new = u + K(y - H u))
            new_state = self.members_pred[j, :] + K @ (
                v_perturbed[j, :] - self.H @ self.members_pred[j, :]
            )

            # Update the ensemble member
            self.members[j, :] = new_state

        # Update the mean and covariance of the analysis ensemble
        self.mean = np.mean(self.members_pred, axis=0)
        self.cov = np.cov(self.members_pred.T)

        return None
