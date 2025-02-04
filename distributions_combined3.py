import numpy as np
import matplotlib.pyplot as plt

# Define the solar availability function
def solar_availability(t):
    """
    Computes solar availability based on a sinusoidal function.
    :param t: Time in hours
    :return: Solar availability (0 to 1)
    """
    return 0.5 * (1 + np.sin((2 * np.pi * t / 24) - (np.pi / 2)))

# Set parameters for Weibull-distributed wind availability
shape, scale = 2.3, 9.5  # Weibull shape and scale parameters
max_possible_speed = scale * 1.5  # Maximum wind speed used for normalization

# Set time parameters for a 24-hour period
single_day_hours = 24
t_values_24h = np.linspace(0, single_day_hours, 1000)  # Smooth time values for 24 hours

# Generate wind speeds using Weibull distribution
np.random.seed(42)  # For reproducibility
wind_availability_24h = np.random.weibull(shape, single_day_hours) * scale

# Normalize and clip wind availability to be between 0.4 and 1.0
wind_availability_24h = np.clip(wind_availability_24h / max_possible_speed, 0.4, 1.0)

# Extend wind availability values for smooth plotting
t_hours_24h = np.arange(single_day_hours)  # Discrete hourly values
wind_availability_24h_extended = np.interp(t_values_24h, t_hours_24h, wind_availability_24h)  # Interpolation

# Compute solar availability over 24 hours
A_values_24h = solar_availability(t_values_24h)

# Compute the combined availability (average of solar and wind)
combined_availability = (A_values_24h + wind_availability_24h_extended) / 2

# Plot solar and wind availability over 24 hours
plt.figure(figsize=(12, 6))
plt.plot(t_values_24h, A_values_24h, label="Solar Availability (Sine Wave)", color="orange", linewidth=2)
plt.plot(t_values_24h, wind_availability_24h_extended, label="Wind Availability (Weibull-based)", color="blue", linestyle="dashed", linewidth=2)


plt.xlabel("Time (hours)", fontsize=30)
plt.ylabel("Normalised Availability", fontsize=30)
plt.title("Solar and Wind Availability Over 24 Hours", fontsize=32, fontweight="bold")
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=24)
plt.grid(True)

# Show the first plot
plt.tight_layout()
plt.show()

# Now plot the combined availability
plt.figure(figsize=(12, 6))
plt.plot(t_values_24h, combined_availability, label="Combined Availability", color="purple", linewidth=2)


plt.xlabel("Time (hours)", fontsize=30)
plt.ylabel("Normalised Availability", fontsize=30)
plt.title("Combined Solar and Wind Availability Over 24 Hours", fontsize=32, fontweight="bold")
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.grid(True)

# Show the combined distribution plot
plt.tight_layout()
plt.show()
