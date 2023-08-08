from unittest import TestCase

from hydra import compose, initialize

from mlbpestimation.train import main


class TrainIntegrationTest(TestCase):
    def test_train_mimic_window_mlp(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=window", "hypothesis/model=mlp"])

    def test_train_mimic_window_resnet(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=window", "hypothesis/model=resnet"])

    def test_train_mimic_window_rnnmlp(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=window", "hypothesis/model=rnnmlp"])

    def test_train_mimic_window_slapnicar(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=window", "hypothesis/model=slapnicar"])

    def test_train_mimic_window_tazarv(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=window", "hypothesis/model=tazarv"])

    def test_train_mimic_window_transofmerencoder(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=window", "hypothesis/model=transformerencoder"])

    def test_train_mimic_heartbeat_mlp(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=heartbeat", "hypothesis/model=mlp"])

    def test_train_mimic_heartbeat_resnet(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=heartbeat", "hypothesis/model=resnet"])

    def test_train_mimic_heartbeat_rnnmlp(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=heartbeat", "hypothesis/model=rnnmlp"])

    def test_train_mimic_heartbeat_slapnicar(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=heartbeat", "hypothesis/model=slapnicar"])

    def test_train_mimic_heartbeat_tazarv(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=heartbeat", "hypothesis/model=tazarv"])

    def test_train_mimic_heartbeat_transofmerencoder(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=heartbeat", "hypothesis/model=transformerencoder"])

    def test_train_mimic_beatsequence_mlp(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=beatsequence", "hypothesis/model=mlp"])

    def test_train_mimic_beatsequence_resnet(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=beatsequence", "hypothesis/model=resnet_seq"])

    def test_train_mimic_beatsequence_rnnmlp(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=beatsequence", "hypothesis/model=rnnmlp"])

    def test_train_mimic_beatsequence_transofmerencoder(self):
        self._train(["hypothesis/dataset/source=mimic", "hypothesis/dataset/decorators=beatsequence", "hypothesis/model=transformerencoder"])

    def _train(self, test_overrides):
        with initialize(version_base=None, config_path='../mlbpestimation/configuration'):
            fast_overrides = ["hypothesis.optimization.batch_size=32", "hypothesis.optimization.epoch=1", "hypothesis.optimization.n_batches=10"]
            overrides = test_overrides + fast_overrides
            config = compose(config_name='train', overrides=overrides)

            main(config)
