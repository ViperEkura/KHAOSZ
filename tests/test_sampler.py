from khaosz.core import *
from khaosz.trainer import *
from khaosz.trainer.data_util import *

def test_random_sampler_consistency(random_dataset):
    """Test RandomSampler produces consistent results with same seed"""
    dataset = random_dataset
    
    # Create two samplers with same seed
    sampler1 = RandomSampler(dataset, seed=42)
    sampler2 = RandomSampler(dataset, seed=42)
    
    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))
    
    assert indices1 == indices2

def test_random_sampler_different_seeds(random_dataset):
    """Test RandomSampler produces different results with different seeds"""
    dataset = random_dataset
    
    # Create two samplers with different seeds
    sampler1 = RandomSampler(dataset, seed=42)
    sampler2 = RandomSampler(dataset, seed=123)
    
    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))
    
    # Very high probability they should be different
    assert indices1 != indices2

def test_sampler_state_persistence(random_dataset):
    """Test that sampler state is correctly saved and loaded"""
    dataset = random_dataset
    n = len(dataset)
    
    # Create sampler and get some indices
    sampler = RandomSampler(dataset, seed=42)
    iter1 = iter(sampler)
    indices1 = [next(iter1) for _ in range(min(10, n))]
    
    # Save state
    state_dict = sampler.state_dict()
    
    # Get more indices
    indices2 = [next(iter1) for _ in range(min(10, n - len(indices1)))]
    
    # Create new sampler and load state
    sampler2 = RandomSampler(dataset, seed=42)
    sampler2.load_state_dict(state_dict)
    
    # Check that new sampler produces same sequence from saved point
    iter2 = iter(sampler2)
    indices3 = [next(iter2) for _ in range(min(10, n - len(indices1)))]
    
    assert indices2 == indices3

def test_sampler_across_epochs(random_dataset):
    """Test sampler behavior across multiple epochs"""
    dataset = random_dataset
    n = len(dataset)
    
    sampler = RandomSampler(dataset, seed=42)
    
    # Get indices for first epoch
    epoch1_indices = list(iter(sampler))
    assert len(epoch1_indices) == n
    
    # Get indices for second epoch
    epoch2_indices = list(iter(sampler))
    assert len(epoch2_indices) == n
    
    # Check that epochs have different order (should be random)
    assert epoch1_indices != epoch2_indices
    
    # Check that all indices are present in each epoch
    assert set(epoch1_indices) == set(range(n))
    assert set(epoch2_indices) == set(range(n))