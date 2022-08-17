from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import MeanAbsoluteError
from wandb import Settings, init
from wandb.integration.keras import WandbCallback

from mlbpestimation.configuration import configuration
from mlbpestimation.metrics.standardeviation import AbsoluteError, StandardDeviation
from mlbpestimation.models.baseline import build_baseline_model
from mlbpestimation.vitaldb.casesplit import load_vitaldb_dataset


def main():
    init(project=configuration['wandb.project_name'], entity=configuration['wandb.entity'],
         config=configuration['wandb.config'], mode=configuration['wandb.mode'], settings=Settings(start_method='fork'),
         name=configuration['wandb.run_name'])

    datasets = load_vitaldb_dataset()
    datasets, model = build_baseline_model(datasets, frequency=63)

    model.summary()
    model.compile(optimizer='Adam', loss=MeanSquaredError(),
                  metrics=[
                      MeanAbsoluteError(),
                      StandardDeviation(AbsoluteError())
                  ]
                  )

    model.fit(datasets.train, epochs=100, callbacks=[WandbCallback()], validation_data=datasets.validation)


if __name__ == '__main__':
    main()
