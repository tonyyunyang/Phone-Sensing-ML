import matplotlib.pyplot as plt

# Define data
labels = [f'C{i+1}' for i in range(16)]
data1 = [0.4, 1, 1, 0.2, 0.4, 0.6, 0.2, 1, 0.4, 1, 0.4, 0.8, 0.4, 0.4, 1, 0.4]
data2 = [0.4, 1, 1, 0.8, 0.8, 0.8, 0.2, 1, 0.4, 1, 0.4, 0.8, 0.4, 0.4, 1, 0.4]
data3 = [1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0, 0.6, 0.8, 1, 1, 1, 0.8]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(labels, data1, marker='o', label='Single 16 cell CNN')
plt.plot(labels, data2, marker='o', label='Single 16 cell CNN + RSS')
plt.plot(labels, data3, marker='o', label='Duo CNN + RSS')

# Set y ticks to percentage
plt.yticks([i/10 for i in range(11)], [f'{i*10}%' for i in range(11)])

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Three Options of Acoustic Tracking')

# Show plot
plt.show()
