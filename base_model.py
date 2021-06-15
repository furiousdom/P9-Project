from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from performance_meter import measure_and_print_performance

def get_model(model_name, input_shape):
    model = Sequential(name=model_name)

    model.add(Input(shape=(input_shape, )))
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))

    metrics=['accuracy', 'mean_squared_error']
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    print(model.summary())
    return model

def train(model_name, dataset, batch_size, epochs, callbacks=None):
    model = get_model(model_name)
    model.fit(dataset['x_train'], dataset['y_train'], batch_size, epochs, callbacks=callbacks)
    predictions = model.predict(dataset['x_test'])
    measure_and_print_performance(model_name, dataset['name'], dataset['y_test'], predictions.flatten())

def test(model_name, datasets, checkpoint_path):
    model = get_model(model_name)
    model.load_weights(checkpoint_path)
    for dataset in datasets:
        predictions = model.predict(dataset['x_test'])
        measure_and_print_performance(model_name, dataset['name'], dataset['y_test'], predictions.flatten())
