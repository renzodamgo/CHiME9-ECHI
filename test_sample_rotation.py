#!/usr/bin/env python3
"""
Test script to verify the rotating sample selection works correctly.
"""
import random


def test_update_epoch_samples():
    """Test the sample rotation logic"""
    
    # Mock dataset class
    class MockDataset:
        def __init__(self, size):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {"id": f"sample_{idx:04d}"}
    
    # Mock DataLoader class
    class MockDataLoader:
        def __init__(self, dataset):
            self.dataset = dataset
    
    def update_epoch_samples(dataset, split: str, epoch: int, debug: bool = False):
        """
        Update sample selection for the current epoch to provide training diversity.
        Returns new sample IDs to save for this epoch.
        """
        
        data_len = len(dataset.dataset) if hasattr(dataset, 'dataset') else len(dataset)
        
        # Create a consistent seed based on epoch for reproducibility
        random.seed(42 + epoch)
        
        # Enhanced sample selection based on dataset size
        if debug:
            # In debug mode, use fewer samples
            pool_size = min(8, data_len // 20)
            samples_per_epoch = min(3, pool_size)
        else:
            # Full training: more diverse sample selection
            pool_size = min(20, data_len // 50)  # Pool of 20 samples, roughly 1% of dataset
            samples_per_epoch = min(6, pool_size)  # Save 6 samples per epoch for diversity
        
        # Generate random sample indices for this epoch
        sample_indices = random.sample(range(data_len), pool_size)
        
        # Select samples for this epoch, rotating through the pool
        epoch_start_idx = (epoch * samples_per_epoch) % pool_size
        epoch_sample_indices = []
        for i in range(samples_per_epoch):
            idx = sample_indices[(epoch_start_idx + i) % pool_size]
            epoch_sample_indices.append(idx)
        
        # Get the actual dataset object to fetch sample IDs
        actual_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        samples = [actual_dataset.__getitem__(i)["id"] for i in epoch_sample_indices]
        
        print(f"=== EPOCH {epoch} SAMPLE SELECTION ({split.upper()}) ===")
        print(f"Selected {len(samples)} samples from pool of {pool_size}")
        print(f"Sample indices: {epoch_sample_indices}")
        print(f"Sample IDs: {samples}")
        
        return samples

    # Test with dataset similar to actual size (4944 samples)
    mock_dataset = MockDataset(4944)
    mock_loader = MockDataLoader(mock_dataset)
    
    print("Testing sample rotation across epochs:")
    print("=" * 60)
    
    # Test first 5 epochs
    all_samples = set()
    for epoch in range(5):
        samples = update_epoch_samples(mock_loader, "train", epoch, debug=False)
        all_samples.update(samples)
        print()
    
    print(f"Total unique samples across 5 epochs: {len(all_samples)}")
    print(f"Expected: Different samples each epoch, demonstrating rotation")
    print("=" * 60)
    
    # Test debug mode
    print("\nTesting debug mode (smaller dataset):")
    debug_samples = update_epoch_samples(mock_loader, "train", 0, debug=True)
    print(f"Debug mode samples: {len(debug_samples)} (should be <= 3)")


if __name__ == "__main__":
    test_update_epoch_samples()