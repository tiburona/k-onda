# test_debugging.py
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create an array with NumPy and perform a calculation
    data = np.array([1, 2, 3, 4, 5])
    squared = np.square(data)

    # Plot the result using Matplotlib
    plt.figure()
    plt.plot(data, squared)
    plt.title("Plot of Squared Values")
    plt.xlabel("Original Values")
    plt.ylabel("Squared Values")
    plt.show()

if __name__ == "__main__":
    main()
