import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset


def convert_to_overlapping_windows(
    window_size: int, data: np.ndarray, labels: np.ndarray | None = None, step_size: int = 1
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert time series data into overlapping windows.

    :param data: Time series data as a 2D numpy array (shape: [n_samples, n_features]).
    :param labels: Corresponding labels for the time series data.
    :param window_size: Size of each window.
    :param step_size: Step size for moving the window.
    :return: Tuple of windows and corresponding labels.
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // step_size + 1

    windows = np.zeros((n_windows, window_size, n_features))
    window_labels = np.zeros(n_windows) if labels is not None else None

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]
        if labels is not None:
            window_labels[i] = labels[start + window_size - 1]  # Label of the last element in the window

    return windows, window_labels


def convert_dataset_to_overlapping_windows(
    window_size: int, dataset: ConceptsDataset, step_size: int = 1
) -> ConceptsDataset:
    """
    Convert ConceptsDataset to the version with overlapping windows.
    """
    transformed_train_concepts = []
    transformed_test_concepts = []
    for concept in dataset.train_concepts():
        windows, labels = convert_to_overlapping_windows(window_size, concept.data, concept.labels, step_size)
        transformed_train_concepts.append(Concept(name=concept.name, data=windows, labels=labels))

    for concept in dataset.test_concepts():
        windows, labels = convert_to_overlapping_windows(window_size, concept.data, concept.labels, step_size)
        transformed_test_concepts.append(Concept(name=concept.name, data=windows, labels=labels))

    return ConceptsDataset(
        name=dataset.name(), train_concepts=transformed_train_concepts, test_concepts=transformed_test_concepts
    )
