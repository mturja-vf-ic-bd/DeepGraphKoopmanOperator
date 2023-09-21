import numpy as np
import matplotlib.pyplot as plt

# Create a time array
t = np.linspace(0, 2 * np.pi, 1000)  # Time values from 0 to 2*pi

# Define frequencies
frequencies = [1.8]  # You can add more frequencies as needed

# Plot sine waves for each frequency
plt.figure(figsize=(30, 6))
for freq in frequencies:
    # Generate sine wave
    y = np.sin(2 * np.pi * freq * t)

    # Plot the sine wave
    plt.plot(t, y)

plt.xticks([])
plt.yticks([])
plt.show()
