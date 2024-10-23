from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

def create_data():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    return X, y

def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(10,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return model

def train_model_with_history(model, optimizer, X, y, batch_size, epochs, optimizer_name):
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = []
    
    for epoch in range(epochs):
        hist = model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0)
        loss = hist.history['loss'][0]
        history.append(loss)
        print(f"Epoch {epoch + 1}/{epochs} - {optimizer_name} Loss: {loss:.4f}")
    return history

# Create data
X, y = create_data()

# Create models
model_sgd = create_model()
model_adam = create_model()
model_rmsprop = create_model()
model_adagrad = create_model()

# Create optimizers
optimizer_sgd = optimizers.SGD(learning_rate=0.01)
optimizer_adam = optimizers.Adam(learning_rate=0.001)
optimizer_rmsprop = optimizers.RMSprop(learning_rate=0.001)
optimizer_adagrad = optimizers.Adagrad(learning_rate=0.01)

# Set training parameters
epochs = 50
batch_size = 32

# Train models with different optimizers
print("\nTraining with SGD Optimizer:")
sgd_loss = train_model_with_history(model_sgd, optimizer_sgd, X, y, batch_size, epochs, 'SGD')

print("\nTraining with Adam Optimizer:")
adam_loss = train_model_with_history(model_adam, optimizer_adam, X, y, batch_size, epochs, 'Adam')

print("\nTraining with RMSprop Optimizer:")
rmsprop_loss = train_model_with_history(model_rmsprop, optimizer_rmsprop, X, y, batch_size, epochs, 'RMSprop')

print("\nTraining with Adagrad Optimizer:")
adagrad_loss = train_model_with_history(model_adagrad, optimizer_adagrad, X, y, batch_size, epochs, 'Adagrad')

# Plot loss comparison
plt.plot(range(1, epochs + 1), sgd_loss, label='SGD', color='blue')
plt.plot(range(1, epochs + 1), adam_loss, label='Adam', color='orange')
plt.plot(range(1, epochs + 1), rmsprop_loss, label='RMSprop', color='green')
plt.plot(range(1, epochs + 1), adagrad_loss, label='Adagrad', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Optimizer Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()
