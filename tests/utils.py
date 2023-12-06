from tensorflow.python.data import Dataset


def get_dataset_output_shapes(dataset: Dataset):
    return [spec.shape for spec in dataset.element_spec]
