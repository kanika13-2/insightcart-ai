from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_ann(df):
    # Features
    X = df[['age', 'purchase_amount']]

    # Target: High spender (above average)
    y = (df['purchase_amount'] > df['purchase_amount'].mean()).astype(int)

    # Model
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X, y, epochs=5, verbose=0)

    return model