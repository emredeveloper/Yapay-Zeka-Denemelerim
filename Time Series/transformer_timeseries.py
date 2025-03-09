import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Fetch Apple stock data (2010-2023)
df = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
df = df[['Close']]  # Use closing prices

# Normalize data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Create sequences (input: 60 days, output: 1 day)
SEQ_LENGTH = 60
BATCH_SIZE = 32

generator = TimeseriesGenerator(
    scaled, scaled,
    length=SEQ_LENGTH,
    batch_size=BATCH_SIZE
)

# Split into train/test (80/20)
split = int(len(scaled) * 0.8)
train_data = scaled[:split]
test_data = scaled[split:]

train_gen = TimeseriesGenerator(
    train_data, train_data,
    length=SEQ_LENGTH,
    batch_size=BATCH_SIZE
)
test_gen = TimeseriesGenerator(
    test_data, test_data,
    length=SEQ_LENGTH,
    batch_size=BATCH_SIZE
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout, TimeDistributed

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

class ReduceMeanLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # LSTM layer to capture temporal patterns
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)  # Adjust dropout rate
    
    # Transformer layer to model long-range dependencies
    transformer_out = TransformerBlock(embed_dim=128, num_heads=8, ff_dim=256)(lstm_out)
    transformer_out = Dropout(0.3)(transformer_out)  # Adjust dropout rate
    
    # Attention mechanism
    attention = MultiHeadAttention(num_heads=4, key_dim=128)(transformer_out, transformer_out)
    attention = ReduceMeanLayer()(attention)  # Use custom layer to reduce mean
    
    # Final prediction
    outputs = Dense(1)(attention)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')  # Adjust learning rate
    return model

model = build_hybrid_model(input_shape=(SEQ_LENGTH, 1))
model.summary()

# Early stopping to prevent overfitting
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,  # Increase patience
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=5,  # Increase number of epochs
    callbacks=[callback]
)

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Predict on test data
test_predictions = model.predict(test_gen)
test_true = np.concatenate([y for x, y in test_gen], axis=0)

# Inverse transform to original scale
test_predictions = scaler.inverse_transform(test_predictions)
test_true = scaler.inverse_transform(test_true)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(test_true):], test_true, label='Actual Prices')
plt.plot(df.index[-len(test_predictions):], test_predictions, label='Predicted Prices')
plt.title('Apple Stock Price Prediction')
plt.legend()
plt.show()

# Calculate RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
print(f'Root Mean Squared Error: {rmse:.2f}')

# Get attention weights from the model
attention_layer = model.layers[4]  # Index of MultiHeadAttention layer
attention_model = Model(
    inputs=model.inputs,
    outputs=attention_layer.output
)

# Get attention scores for a sample
sample_input = test_gen[0][0][0:1]  # First batch, first sample
attention_scores = attention_model.predict(sample_input)

print(attention_scores)