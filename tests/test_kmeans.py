import pytest
import numpy as np
from src.kmeans import MPIKMeans
from src.config import Config

@pytest.fixture
def sample_config():
    return Config(
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        random_state=42,
        data_path="data/sample_data.csv",
        log_level="INFO"
    )

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(100, 2)

def test_kmeans_initialization(sample_config):
    model = MPIKMeans(sample_config)
    assert model.n_clusters == 3
    assert model.max_iter == 100
    assert model.tol == 1e-4

def test_kmeans_fit(sample_config, sample_data):
    model = MPIKMeans(sample_config)
    model.fit(sample_data)
    assert hasattr(model, 'centroids_')
    assert model.centroids_.shape == (3, 2)

def test_kmeans_predict(sample_config, sample_data):
    model = MPIKMeans(sample_config)
    model.fit(sample_data)
    labels = model.predict(sample_data)
    assert labels.shape == (100,)
    assert np.all(labels >= 0) and np.all(labels < 3)