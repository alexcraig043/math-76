import numpy as np
import matplotlib.pyplot as plt
from EnKF import EnKF


# Define a simple linear forward model
def linear_forward_model(state, A):
    return A @ state


# Define the true state evolution for comparison (simple linear model)
def true_state_evolution(initial_state, A, steps):
    states = [initial_state]
    for _ in range(steps - 1):
        states.append(A @ states[-1])
    return np.array(states)


# Initialize EnKF
state_dimension = 2
observation_dimension = 2
ensemble_size = 100

# Generate a random linear forward model A
A = np.array(np.random.randn(state_dimension, observation_dimension))
# Multiply transpose of A with A to ensure it is positive definite
A = A.T @ A
# Perform QR decomposition to get an orthogonal matrix
A, R = np.linalg.qr(A)

### Rotational matrix example
# theta = 0.3
# A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

H = np.eye(state_dimension)  # Observe all state variables

enkf = EnKF(
    ensemble_size=ensemble_size,
    state_dimension=state_dimension,
    observation_dimension=observation_dimension,
    forward_model=lambda x: linear_forward_model(x, A=A),
    observation_operator=H,
    ensemble_variance=0.01,
    observation_variance=0.01,
    initial_ensemble=np.random.randn(state_dimension, ensemble_size),
)

# Define time steps
times = np.arange(100)

# True initial state and evolution
true_initial_state = np.random.randn(state_dimension)
true_states = true_state_evolution(true_initial_state, A=A, steps=len(times))

# Run EnKF
states = [np.mean(enkf.ensemble, axis=1)]
for time_step in range(1, len(times)):
    enkf.propagation_step()
    # Generate observation from true state with noise
    true_state = true_states[time_step]
    observation = true_state + np.random.normal(
        0, np.sqrt(enkf.observation_variance), size=observation_dimension
    )
    enkf.assimilation_step(observation)
    states.append(np.mean(enkf.ensemble, axis=1))

# Convert states to a numpy array for easier handling
states_array = np.array(states)

# Plotting
plt.figure(figsize=(10, 6))
# Adjusted plotting for multi-dimensional data
plt.plot(times, true_states[:, 0], "r-", label="True State")
plt.plot(
    times, states_array[:, 0], "b--", label="EnKF Mean State of the first component"
)
plt.scatter(
    times,
    states_array[:, 0],
    color="blue",
    s=10,
    label="EnKF Mean State at Each Step",
)
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.title("True State vs. EnKF Predicted State")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("../plots/EnFK.png", dpi=600.0)

plt.show()
