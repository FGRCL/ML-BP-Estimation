from keras.metrics import MeanAbsoluteError
from tensorflow import keras
from wandb import init
from wandb.integration.keras import WandbCallback

from mlbpestimation.configuration import configuration
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.heartbeat import build_heartbeat_cnn_model
from mlbpestimation.vitaldb.casesplit import load_vitaldb_dataset


def main():
    epochs = 10
    init(project=configuration['wandb.project_name'], entity=configuration['wandb.entity'],
         config=configuration['wandb.config'], mode=configuration['wandb.mode'])

    train, val, _ = load_vitaldb_dataset()
    (train, val), model = build_heartbeat_cnn_model([train, val])
    train, val = train.take(100), val.take(10)

    model.summary()
    model.compile(optimizer='Adam', loss=keras.losses.MeanSquaredError(),
                  metrics=[
                      MeanAbsoluteError(),
                      StandardDeviation(AbsoluteError())
                  ]
                  )

    model.fit(train, epochs=epochs, callbacks=[WandbCallback()], validation_data=val)

    model.save('models')


if __name__ == '__main__':
    main()
