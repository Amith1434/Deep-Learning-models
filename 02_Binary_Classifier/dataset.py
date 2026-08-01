from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

X, y = make_circles(n_samples = 100, noise = 0.03, random_state = 42)

if __name__ == "__main__":
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

