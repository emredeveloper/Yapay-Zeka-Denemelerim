import numpy as np
import matplotlib.pyplot as plt

# Quadratic loss: f(x, y) = 0.5 * (a*x^2 + b*y^2)
# Choose a << b to create a narrow, steep valley in y (causes oscillations)
a, b = 1.0, 50.0

def loss(w):
    x, y = w
    return 0.5 * (a * x**2 + b * y**2)

def grad(w):
    x, y = w
    return np.array([a * x, b * y])

def gradient_descent(w0, lr=0.08, steps=60):
    w = w0.copy()
    traj = [w.copy()]
    for _ in range(steps):
        g = grad(w)
        w = w - lr * g
        traj.append(w.copy())
    return np.array(traj)

def momentum(w0, lr=0.08, beta=0.9, steps=60):
    w = w0.copy()
    v = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        g = grad(w)
        v = beta * v + (1 - beta) * g           # moving average of gradients
        w = w - lr * v
        traj.append(w.copy())
    return np.array(traj)

# Run experiments
w0 = np.array([2.5, 2.5])
traj_gd = gradient_descent(w0, lr=0.08, steps=60)
traj_mom_low = momentum(w0, lr=0.08, beta=0.5, steps=60)  # small momentum
traj_mom_high = momentum(w0, lr=0.08, beta=0.9, steps=60) # large momentum

# Plot contours and trajectories
xlin = np.linspace(-3, 3, 400)
ylin = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(xlin, ylin)
Z = 0.5 * (a * X**2 + b * Y**2)

plt.figure(figsize=(10, 8))
cs = plt.contour(X, Y, Z, levels=20, cmap='gray')
plt.clabel(cs, inline=1, fontsize=8)

def plot_traj(traj, label, color):
    plt.plot(traj[:,0], traj[:,1], '-o', ms=3, lw=1.5, label=label, color=color)
    plt.scatter(traj[0,0], traj[0,1], color=color, edgecolor='k', zorder=3)

plot_traj(traj_gd, 'GD (no momentum)', '#1f77b4')
plot_traj(traj_mom_low, 'Momentum β=0.5', '#2ca02c')
plot_traj(traj_mom_high, 'Momentum β=0.9', '#d62728')

plt.scatter(0, 0, c='k', s=60, marker='*', label='Optimum')
plt.title('Gradient Descent vs Momentum on Anisotropic Quadratic')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid(alpha=0.2)
plt.show()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def make_data(n=200):
    mean0 = np.array([-2.0, 0.0])
    mean1 = np.array([2.0, 0.5])
    cov0 = np.array([[1.0, 0.4],[0.4, 0.7]])
    cov1 = np.array([[1.0, -0.3],[-0.3, 0.5]])
    X0 = np.random.multivariate_normal(mean0, cov0, n)
    X1 = np.random.multivariate_normal(mean1, cov1, n)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return X, y, Xb

def bce_loss(W, Xb, y):
    p = sigmoid(Xb @ W)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def acc(W, Xb, y):
    p = sigmoid(Xb @ W)
    yhat = (p >= 0.5).astype(np.float64)
    return np.mean(yhat == y)

def train_gd(Xb, y, lr=0.1, epochs=200):
    n, d = Xb.shape
    W = np.zeros(d)
    losses, accs = [], []
    for _ in range(epochs):
        p = sigmoid(Xb @ W)
        g = (Xb.T @ (p - y)) / n
        W = W - lr * g
        losses.append(bce_loss(W, Xb, y))
        accs.append(acc(W, Xb, y))
    return W, np.array(losses), np.array(accs)

def train_momentum(Xb, y, lr=0.1, beta=0.9, epochs=200):
    n, d = Xb.shape
    W = np.zeros(d)
    v = np.zeros_like(W)
    losses, accs = [], []
    for _ in range(epochs):
        p = sigmoid(Xb @ W)
        g = (Xb.T @ (p - y)) / n
        v = beta * v + (1 - beta) * g
        W = W - lr * v
        losses.append(bce_loss(W, Xb, y))
        accs.append(acc(W, Xb, y))
    return W, np.array(losses), np.array(accs)

X, y, Xb = make_data(n=250)
W_gd, loss_gd, acc_gd = train_gd(Xb, y, lr=0.15, epochs=250)
W_m, loss_m, acc_m = train_momentum(Xb, y, lr=0.15, beta=0.9, epochs=250)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_gd, label='GD')
plt.plot(loss_m, label='Momentum')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Logistic Regression Loss')
plt.legend()
plt.grid(alpha=0.2)

plt.subplot(1,2,2)
plt.plot(acc_gd, label='GD')
plt.plot(acc_m, label='Momentum')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Accuracy')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

def plot_decision_boundary(ax, W, color):
    x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
    if abs(W[2]) > 1e-8:
        y_vals = -(W[0] + W[1]*x_vals)/W[2]
        ax.plot(x_vals, y_vals, color=color, lw=2)
    else:
        x0 = -W[0]/(W[1]+1e-12)
        ax.axvline(x0, color=color, lw=2)

plt.figure(figsize=(8,6))
idx0 = (y==0)
idx1 = (y==1)
plt.scatter(X[idx0,0], X[idx0,1], s=20, alpha=0.7, label='Class 0')
plt.scatter(X[idx1,0], X[idx1,1], s=20, alpha=0.7, label='Class 1')
ax = plt.gca()
plot_decision_boundary(ax, W_gd, '#1f77b4')
plot_decision_boundary(ax, W_m, '#d62728')
plt.legend(['GD boundary','Momentum boundary','Class 0','Class 1'])
plt.title('Decision Boundaries: GD vs Momentum (Logistic Regression)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(alpha=0.2)
plt.show()