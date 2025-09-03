import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
import logging
from .echi import ECHIJoint
from .enhanced_speaker_detection import detect_active_speakers_energy, validate_speaker_activity_batch


class EnhancedECHIJoint(ECHIJoint):
    """
    Enhanced ECHIJoint with energy-based active speaker validation.
    
    Improvements over base ECHIJoint:
    1. Energy-based validation of file-based activity detection
    2. Confidence scores for speaker activity  
    3. Runtime validation during training
    4. Detailed logging of activity decisions
    """
    
    def __init__(self, subset, audio_device, noisy_signal, ref_signal, rainbow_signal,
                 sessions_file, segments_file, debug, 
                 energy_threshold_db=-40, min_speech_duration=0.1, validate_energy=True):
        """
        Args:
            validate_energy: Whether to use energy validation on top of file-based detection
            energy_threshold_db: dB threshold for considering speech active
            min_speech_duration: Minimum seconds of speech required for active classification
        """
        self.validate_energy = validate_energy
        self.energy_threshold_db = energy_threshold_db  
        self.min_speech_duration = min_speech_duration
        
        # Call parent constructor
        super().__init__(subset, audio_device, noisy_signal, ref_signal, rainbow_signal,
                        sessions_file, segments_file, debug)
        
        # Statistics
        self.activity_stats = {
            "total_samples": 0,
            "file_based_active": 0, 
            "energy_validated_active": 0,
            "corrections_made": 0
        }

    def __getitem__(self, index):
        # Get base implementation
        out = super().__getitem__(index)
        
        if not self.validate_energy:
            return out
            
        # Energy-based validation
        meta = self.manifest[index]
        file_based_mask = out["speaker_active_mask"]  # [K]
        
        # Analyze energy in target signals
        target_all = out["target_all"]  # [K, Tw]
        K, Tw = target_all.shape
        
        energy_validated_mask = []
        confidence_scores = []
        corrections = 0
        
        for k in range(K):
            file_active = file_based_mask[k].item()
            target_audio = target_all[k]  # [Tw]
            
            # Energy analysis
            energy_active, confidence = self._analyze_target_energy(target_audio)
            
            # Decision logic: combine file-based + energy-based
            if file_active and not energy_active:
                # File says active, energy says silent → trust energy (likely silence padding)
                final_active = False
                corrections += 1
                logging.debug(f"Sample {out['id']} speaker {k}: file_active=True, energy_active=False → Silent")
                
            elif not file_active and energy_active:
                # File says silent, energy says active → trust file (likely crosstalk/noise)
                final_active = False  # Conservative: trust file-based detection
                logging.debug(f"Sample {out['id']} speaker {k}: file_active=False, energy_active=True → Silent (conservative)")
                
            else:
                # File and energy agree
                final_active = file_active
                
            energy_validated_mask.append(final_active)
            confidence_scores.append(confidence)
        
        # Update output
        out["speaker_active_mask"] = torch.tensor(energy_validated_mask, dtype=torch.bool)
        out["speaker_confidence_scores"] = torch.tensor(confidence_scores, dtype=torch.float32)
        
        # Update statistics
        self.activity_stats["total_samples"] += 1
        self.activity_stats["file_based_active"] += file_based_mask.sum().item()
        self.activity_stats["energy_validated_active"] += out["speaker_active_mask"].sum().item()
        self.activity_stats["corrections_made"] += corrections
        
        return out
    
    def _analyze_target_energy(self, audio_tensor):
        """
        Analyze energy in target audio tensor.
        
        Args:
            audio_tensor: [T] audio signal
            
        Returns:
            is_active: bool, whether speaker is considered active
            confidence: float, confidence in the decision [0,1]
        """
        T = len(audio_tensor)
        if T == 0:
            return False, 0.0
            
        # Frame-wise energy analysis
        frame_length = int(0.025 * 16000)  # 25ms at 16kHz
        hop_length = int(0.010 * 16000)    # 10ms hop
        
        if T < frame_length:
            # Very short audio - use global energy
            energy = torch.mean(audio_tensor ** 2).item()
            energy_db = 10 * torch.log10(energy + 1e-10)
            is_active = energy_db > self.energy_threshold_db
            confidence = min(max((energy_db - self.energy_threshold_db) / 20.0, 0.0), 1.0)
            return is_active, confidence
        
        # Frame-based analysis
        frames = audio_tensor.unfold(0, frame_length, hop_length)  # [n_frames, frame_length]
        frame_energy = torch.mean(frames ** 2, dim=1)               # [n_frames]
        frame_energy_db = 10 * torch.log10(frame_energy + 1e-10)   # [n_frames]
        
        # Active frames
        active_frames = frame_energy_db > self.energy_threshold_db
        n_active = active_frames.sum().item()
        n_total = len(active_frames)
        
        # Duration check
        active_duration = n_active * hop_length / 16000
        active_ratio = n_active / max(n_total, 1)
        
        # Decision
        is_active = (active_duration >= self.min_speech_duration and active_ratio >= 0.05)
        
        # Confidence
        if is_active:
            avg_energy = frame_energy_db[active_frames].mean().item() if n_active > 0 else self.energy_threshold_db
            confidence = min(max((avg_energy - self.energy_threshold_db) / 20.0, 0.0), 1.0)
        else:
            confidence = max(active_ratio, 0.1)  # Some confidence even when inactive
            
        return is_active, confidence
    
    def get_activity_statistics(self):
        """Get statistics about speaker activity detection."""
        stats = self.activity_stats.copy()
        if stats["total_samples"] > 0:
            stats["avg_file_active_per_sample"] = stats["file_based_active"] / stats["total_samples"]
            stats["avg_energy_active_per_sample"] = stats["energy_validated_active"] / stats["total_samples"]
            stats["correction_rate"] = stats["corrections_made"] / stats["total_samples"]
        
        return stats
    
    def log_activity_summary(self):
        """Log summary of activity detection performance."""
        stats = self.get_activity_statistics()
        
        logging.info("=== Enhanced Activity Detection Summary ===")
        logging.info(f"Total samples processed: {stats['total_samples']}")
        logging.info(f"File-based active speakers: {stats['file_based_active']}")
        logging.info(f"Energy-validated active speakers: {stats['energy_validated_active']}")
        logging.info(f"Corrections made: {stats['corrections_made']}")
        
        if stats["total_samples"] > 0:
            logging.info(f"Avg active speakers per sample (file): {stats['avg_file_active_per_sample']:.2f}")
            logging.info(f"Avg active speakers per sample (energy): {stats['avg_energy_active_per_sample']:.2f}")
            logging.info(f"Correction rate: {stats['correction_rate']*100:.1f}%")


def collate_fn_joint_enhanced(batch):
    """Enhanced collate function that handles confidence scores."""
    from .echi import collate_fn_joint
    
    # Use base collate function
    out = collate_fn_joint(batch)
    
    # Add confidence scores if present
    if "speaker_confidence_scores" in batch[0]:
        confidence_scores = [x["speaker_confidence_scores"] for x in batch]
        # Pad to same K dimension
        max_K = max(scores.shape[0] for scores in confidence_scores)
        padded_scores = []
        for scores in confidence_scores:
            if scores.shape[0] < max_K:
                padding = torch.zeros(max_K - scores.shape[0])
                scores = torch.cat([scores, padding])
            padded_scores.append(scores)
        
        out["speaker_confidence_scores"] = torch.stack(padded_scores)  # [B, K]
    
    return out


if __name__ == "__main__":
    # Test enhanced dataset
    print("Enhanced ECHIJoint dataset with energy-based validation")
    print("Features:")
    print("- File-based activity detection (existing)")
    print("- Energy-based validation (new)")
    print("- Confidence scores per speaker")
    print("- Detailed activity statistics")
    print("- Runtime validation during training")