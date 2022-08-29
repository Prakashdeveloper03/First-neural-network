import tensorflow as tf

# MNIST dataset
mnist = tf.keras.datasets.mnist

# Data loader
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocess
x_train, x_test = x_train / 255.0, x_test / 255.0

# Fully connected neural network with one hidden layer
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

# Loss and optimizer
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)

# Saves the model
model.save("model/mnist.h5")
