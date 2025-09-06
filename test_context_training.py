#!/usr/bin/env python3
"""
Test script for Context-Aware Multi-Speaker Training setup.

This script validates that all components work together before running full training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import logging
from omegaconf import OmegaConf

# Test imports
def test_imports():
    """Test that all required modules can be imported."""
    logging.info("🔍 Testing imports...")
    
    try:
        from shared.MultiSpeakerContextGridNet import MultiSpeakerContextGridNet
        logging.info("   ✅ MultiSpeakerContextGridNet")
        
        from train.multi_context_loss import contrastive_multi_speaker_loss, log_separation_metrics
        logging.info("   ✅ multi_context_loss")
        
        from shared.core_utils import get_device
        logging.info("   ✅ core_utils")
        
        from shared.signal_utils import STFTWrapper
        logging.info("   ✅ signal_utils")
        
        return True
    except ImportError as e:
        logging.error(f"   ❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality."""
    logging.info("🔍 Testing model creation...")
    
    try:
        from shared.MultiSpeakerContextGridNet import MultiSpeakerContextGridNet
        
        model = MultiSpeakerContextGridNet(
            n_imics=4,
            n_layers=3,
            emb_dim=24,
            context_layers=2,
            context_heads=2
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"   ✅ Model created with {total_params:,} parameters")
        
        # Test forward pass
        batch_size, mics, freq, time = 2, 4, 17, 25
        spec = torch.randn(batch_size, mics, freq, time, 2)
        
        # Single speaker
        spk_single = torch.randn(batch_size, freq, time, 2)
        spk_lens_single = torch.tensor([time, time])
        output_single = model(spec, spk_single, spk_lens_single)
        logging.info(f"   ✅ Single speaker forward: {output_single.shape}")
        
        # Multi-speaker
        num_speakers = 3
        spk_multi = torch.randn(batch_size, num_speakers, freq, time, 2)
        spk_lens_multi = torch.tensor([[time, time, time], [time, time, time]])
        output_multi = model(spec, spk_multi, spk_lens_multi)
        logging.info(f"   ✅ Multi-speaker forward: {output_multi.shape}")
        
        return True
    except Exception as e:
        logging.error(f"   ❌ Model test failed: {e}")
        return False

def test_loss_function():
    """Test the contrastive multi-speaker loss."""
    logging.info("🔍 Testing loss function...")
    
    try:
        from train.multi_context_loss import contrastive_multi_speaker_loss, log_separation_metrics
        
        batch_size, num_speakers, time = 2, 3, 1000
        
        # Create test data
        predicted = torch.randn(batch_size, num_speakers, time)
        target = torch.randn(batch_size, num_speakers, time)
        
        context_info = {
            'num_speakers': num_speakers,
            'separation_difficulty': torch.rand(batch_size),
            'speaker_similarities': torch.rand(batch_size, num_speakers, num_speakers),
        }
        
        # Test loss computation
        loss, components = contrastive_multi_speaker_loss(predicted, target, context_info)
        
        logging.info(f"   ✅ Loss computed: {loss.item():.4f}")
        logging.info(f"   ✅ Components: {list(components.keys())}")
        
        # Test logging
        log_separation_metrics(components, epoch=1, batch_idx=0, split="test")
        logging.info("   ✅ Metrics logging works")
        
        return True
    except Exception as e:
        logging.error(f"   ❌ Loss test failed: {e}")
        return False

def test_configuration():
    """Test that configuration files are valid."""
    logging.info("🔍 Testing configuration...")
    
    try:
        # Test training config
        train_config = OmegaConf.load("config/train/multi_context_ha.yaml")
        logging.info(f"   ✅ Training config: {train_config.shared.exp_name}")
        
        # Test model config
        model_config = OmegaConf.load("config/model/multi_context_gridnet.yaml")
        logging.info(f"   ✅ Model config: {model_config.name}")
        
        # Validate key parameters
        assert train_config.model.name == "MultiSpeakerContextGridNet"
        assert "loss_weights" in train_config.train
        assert model_config.context_layers > 0
        
        logging.info("   ✅ Configuration validation passed")
        return True
    except Exception as e:
        logging.error(f"   ❌ Configuration test failed: {e}")
        return False

def test_data_compatibility():
    """Test compatibility with existing data loading."""
    logging.info("🔍 Testing data compatibility...")
    
    try:
        # Test that our training script can import required functions
        from train.echi import collate_fn_joint
        logging.info("   ✅ collate_fn_joint import successful")
        
        from train.enhanced_echi_joint import EnhancedECHIJoint
        logging.info("   ✅ EnhancedECHIJoint import successful") 
        
        # Test basic data structures (without actual collating which needs complete data)
        logging.info("   ✅ Data loading components available")
        logging.info("   ℹ️  Full data loading test requires actual dataset")
        
        return True
    except Exception as e:
        logging.error(f"   ❌ Data compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s'
    )
    
    logging.info("🚀 CONTEXT-AWARE MULTI-SPEAKER TRAINING - COMPATIBILITY TEST")
    logging.info("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation), 
        ("Loss Function", test_loss_function),
        ("Configuration", test_configuration),
        ("Data Compatibility", test_data_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logging.info("")
        try:
            if test_func():
                passed += 1
                logging.info(f"✅ {test_name}: PASSED")
            else:
                logging.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logging.error(f"❌ {test_name}: FAILED with exception: {e}")
    
    logging.info("")
    logging.info("="*70)
    logging.info(f"🎯 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logging.info("🎉 ALL TESTS PASSED - Ready for training!")
        logging.info("")
        logging.info("💡 To start training:")
        logging.info("   python run_train_context.py")
        logging.info("   python run_train_context.py shared.exp_name=my-context-test")
        return 0
    else:
        logging.error("❌ Some tests failed - please fix before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())