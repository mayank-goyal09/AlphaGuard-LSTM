import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

class VolatilityPredictor:
    def __init__(self, ticker, sequence_length=30, model_type='LSTM'):
        self.ticker = ticker.upper()
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model_path(self):
        ext = 'keras' if self.model_type == 'LSTM' else 'pkl'
        return os.path.join(self.model_dir, f"{self.ticker}_{self.model_type.lower()}_model.{ext}")

    def _get_scaler_path(self):
        return os.path.join(self.model_dir, f"{self.ticker}_scaler.pkl")

    def prepare_data(self, df, features, target_col='Vol_Range', train_split=0.8, is_training=True):
        """Scales features and creates time-series sequences of length `sequence_length`."""
        df_clean = df[features + [target_col]].dropna()
        
        feature_data = df_clean[features].values
        target_data = df_clean[target_col].values.reshape(-1, 1)

        if is_training:
            scaled_features = self.scaler.fit_transform(feature_data)
            scaled_target = self.target_scaler.fit_transform(target_data)
            
            # Save scalers
            with open(self._get_scaler_path(), 'wb') as f:
                pickle.dump((self.scaler, self.target_scaler), f)
        else:
            # Load scalers
            scaler_path = self._get_scaler_path()
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler, self.target_scaler = pickle.load(f)
                scaled_features = self.scaler.transform(feature_data)
                scaled_target = self.target_scaler.transform(target_data)
            else:
                # Fallback if no saved scaler
                scaled_features = self.scaler.fit_transform(feature_data)
                scaled_target = self.target_scaler.fit_transform(target_data)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i - self.sequence_length:i])
            y.append(scaled_target[i, 0])

        X = np.array(X)
        y = np.array(y)

        if is_training:
            split_idx = int(len(X) * train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            return X_train, y_train, X_test, y_test
        else:
            return X, y

    def build_lstm_model(self, input_shape):
        """Constructs and compiles the LSTM Neural Network."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self, df, features, target_col='Vol_Range', epochs=10, batch_size=32):
        """Trains the selected model type (LSTM or Random Forest) and saves it."""
        if self.model_type == 'LSTM':
            X_train, y_train, X_test, y_test = self.prepare_data(df, features, target_col, is_training=True)
            
            if len(X_train) == 0:
                raise ValueError("Insufficient data to build sequences for training.")
                
            self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            print(f"Training LSTM model for {self.ticker}...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            self.model.save(self._get_model_path())
            return history.history['loss'][-1], history.history['val_loss'][-1]
            
        elif self.model_type == 'RandomForest':
            # Preprocessing for RF (doesn't need 3D sequential data, but we use the flattened sequence inputs)
            X_train, y_train, X_test, y_test = self.prepare_data(df, features, target_col, is_training=True)
            
            # Flatten the 3D sequences to 2D for RF
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            print(f"Training Random Forest model for {self.ticker}...")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_flat, y_train)
            
            # Save RF model
            with open(self._get_model_path(), 'wb') as f:
                pickle.dump(self.model, f)
                
            train_loss = np.mean((self.model.predict(X_train_flat) - y_train) ** 2)
            val_loss = np.mean((self.model.predict(X_test_flat) - y_test) ** 2)
            return train_loss, val_loss

    def load_saved_model(self):
        """Loads a previously trained model and its scalers from disk."""
        model_path = self._get_model_path()
        scaler_path = self._get_scaler_path()
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False
            
        with open(scaler_path, 'rb') as f:
            self.scaler, self.target_scaler = pickle.load(f)
            
        if self.model_type == 'LSTM':
            self.model = load_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        return True

    def predict(self, df, features, target_col='Vol_Range'):
        """Predicts volatility for the last sequence available in the dataset."""
        # Ensure the model is loaded
        if self.model is None:
            if not self.load_saved_model():
                raise FileNotFoundError(f"Model for {self.ticker} not trained and no saved model found.")
                
        # Prepare latest sequence
        df_clean = df[features + [target_col]].dropna()
        if len(df_clean) < self.sequence_length:
            # Not enough data for prediction, use a fallback of recent Vol_Range
            return df_clean[target_col].iloc[-1]
            
        feature_data = df_clean[features].values
        scaled_features = self.scaler.transform(feature_data)
        
        # Take the last sequence
        latest_seq = scaled_features[-self.sequence_length:]
        
        if self.model_type == 'LSTM':
            latest_seq = np.expand_dims(latest_seq, axis=0) # (1, sequence_length, num_features)
            pred_scaled = self.model.predict(latest_seq, verbose=0)[0, 0]
        else:
            latest_seq_flat = latest_seq.reshape(1, -1)
            pred_scaled = self.model.predict(latest_seq_flat)[0]
            
        # Inverse transform to original volatility scale
        pred_actual = self.target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        return pred_actual
