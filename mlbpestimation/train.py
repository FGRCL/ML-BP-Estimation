from argparse import ArgumentParser
from sys import argv

from keras.metrics import MeanAbsoluteError
from tensorflow import keras
from wandb import Settings, init
from wandb.integration.keras import WandbCallback

from mlbpestimation.configuration import configuration
from mlbpestimation.data.mimic4.dataset import load_mimic_dataset
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.baseline import build_baseline_model


def main(run_name):
    print(run_name)
    epochs = 10
    init(project=configuration['wandb.project_name'], entity=configuration['wandb.entity'],
         config=configuration['wandb.config'], mode=configuration['wandb.mode'], settings=Settings(start_method='fork'),
         name=run_name)

    train, val, _ = load_mimic_dataset()
    (train, val), model = build_baseline_model([train, val], frequency=63)

    model.summary()
    model.compile(optimizer='Adam', loss=keras.losses.MeanSquaredError(),
                  metrics=[
                      MeanAbsoluteError(),
                      StandardDeviation(AbsoluteError())
                  ]
                  )

    model.fit(train, epochs=epochs, callbacks=[WandbCallback()], validation_data=val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_name')
    arguments = parser.parse_args(argv[1:])
    main(arguments.run_name)
