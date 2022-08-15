from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError
from wandb import Settings, init
from wandb.integration.keras import WandbCallback

from mlbpestimation.configuration import configuration
from mlbpestimation.data.mimic4.dataset import load_mimic_dataset
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.baseline import build_baseline_model


def main():
    init(project=configuration['wandb.project_name'], entity=configuration['wandb.entity'],
         config=configuration['wandb.config'], mode=configuration['wandb.mode'], settings=Settings(start_method='fork'),
         name=configuration['wandb.run_name'])

    train, val, _ = load_mimic_dataset()
    (train, val), model = build_baseline_model([train, val], frequency=63)

    model.summary()
    model.compile(optimizer='Adam', loss=MeanSquaredError(),
                  metrics=[
                      MeanAbsoluteError(),
                      StandardDeviation(AbsoluteError())
                  ]
                  )

    model.fit(train, epochs=100, callbacks=[WandbCallback()], validation_data=val, steps_per_epoch=10000)


if __name__ == '__main__':
    main()
