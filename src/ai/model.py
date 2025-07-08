from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(input_shape: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)  # Regression output: next percentage change
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
