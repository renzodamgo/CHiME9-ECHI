import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
import logging


def detect_active_speakers_energy(target_paths, threshold_db=-40, min_duration_sec=0.1, sample_rate=16000):
    """
    Enhanced active speaker detection using energy analysis.
    
    Args:
        target_paths: List of target audio paths (None for missing speakers)
        threshold_db: Energy threshold in dB for considering speech active
        min_duration_sec: Minimum duration of active speech required
        sample_rate: Audio sample rate
    
    Returns:
        active_mask: [K] boolean tensor indicating active speakers
        confidence_scores: [K] confidence scores for each speaker's activity
    """
    K = len(target_paths)
    active_mask = []
    confidence_scores = []
    
    for i, target_path in enumerate(target_paths):
        if target_path is None:
            # File-based detection: speaker is silent
            active_mask.append(False)
            confidence_scores.append(0.0)
            continue
            
        try:
            # Load target audio
            audio, fs = torchaudio.load(str(target_path))
            audio = audio.squeeze(0)  # [T]
            
            if fs != sample_rate:
                # Resample if needed
                resampler = torchaudio.transforms.Resample(fs, sample_rate)
                audio = resampler(audio)
            
            # Energy-based analysis
            is_active, confidence = _analyze_speech_activity(
                audio, threshold_db, min_duration_sec, sample_rate
            )
            
            active_mask.append(is_active)
            confidence_scores.append(confidence)
            
        except Exception as e:
            logging.warning(f"Failed to analyze speaker {i} activity: {e}")
            # Fall back to file-based detection
            active_mask.append(True)  # Conservative: assume active if file exists
            confidence_scores.append(0.5)  # Low confidence
    
    return torch.tensor(active_mask, dtype=torch.bool), torch.tensor(confidence_scores)


def _analyze_speech_activity(audio, threshold_db, min_duration_sec, sample_rate):
    """
    Analyze speech activity in audio signal.
    
    Returns:
        is_active: Boolean indicating if speaker is active
        confidence: Float confidence score [0, 1]
    """
    # Compute frame-wise energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    # Pad audio to ensure we get frames
    if len(audio) < frame_length:
        audio = F.pad(audio, (0, frame_length - len(audio)))
    
    # Compute energy per frame
    frames = audio.unfold(0, frame_length, hop_length)  # [n_frames, frame_length]
    frame_energy = torch.mean(frames ** 2, dim=1)       # [n_frames]
    
    # Convert to dB
    frame_energy_db = 10 * torch.log10(frame_energy + 1e-10)
    
    # Find active frames above threshold
    active_frames = frame_energy_db > threshold_db
    
    # Check if we have enough active speech
    n_active_frames = active_frames.sum().item()
    total_frames = len(active_frames)
    
    active_duration = n_active_frames * hop_length / sample_rate
    active_ratio = n_active_frames / max(total_frames, 1)
    
    # Decision logic
    is_active = active_duration >= min_duration_sec and active_ratio >= 0.1
    
    # Confidence based on active ratio and energy levels
    if is_active:
        # High confidence if good active ratio and strong energy
        avg_active_energy = frame_energy_db[active_frames].mean().item()
        energy_confidence = min((avg_active_energy - threshold_db) / 20.0, 1.0)  # Normalize
        ratio_confidence = min(active_ratio * 2.0, 1.0)  # More weight to higher ratios
        confidence = 0.7 * ratio_confidence + 0.3 * max(energy_confidence, 0.0)
    else:
        # Low confidence, proportional to what activity we did find
        confidence = min(active_ratio * 0.5, 0.3)  # Cap low confidence at 0.3
    
    return is_active, confidence


def validate_speaker_activity_batch(target_batch, active_mask_batch):
    """
    Validate active speaker masks against actual audio energy.
    
    Args:
        target_batch: [B, K, T] target audio batch
        active_mask_batch: [B, K] current active mask
        
    Returns:
        corrected_mask: [B, K] energy-validated active mask
        corrections_made: int number of corrections made
    """
    B, K, T = target_batch.shape
    corrected_mask = active_mask_batch.clone()
    corrections_made = 0
    
    for b in range(B):
        for k in range(K):
            # Compute energy for this speaker
            audio = target_batch[b, k]  # [T]
            energy = torch.mean(audio ** 2).item()
            
            # Convert to dB
            energy_db = 10 * torch.log10(energy + 1e-10)
            
            current_active = active_mask_batch[b, k].item()
            
            # Validation logic
            if current_active and energy_db < -50:  # Marked active but very low energy
                corrected_mask[b, k] = False
                corrections_made += 1
                logging.debug(f"Corrected speaker {k} in batch {b}: active→silent (energy: {energy_db:.1f}dB)")
                
            elif not current_active and energy_db > -20:  # Marked silent but high energy
                corrected_mask[b, k] = True  
                corrections_made += 1
                logging.debug(f"Corrected speaker {k} in batch {b}: silent→active (energy: {energy_db:.1f}dB)")
    
    if corrections_made > 0:
        logging.info(f"Made {corrections_made} activity corrections in batch")
    
    return corrected_mask, corrections_made


if __name__ == "__main__":
    # Test the enhanced detection
    print("Testing enhanced speaker activity detection...")
    
    # Create dummy test data
    test_paths = [
        "/tmp/test_active.wav",  # Would be active speaker
        None,                    # Silent speaker (no file)
        "/tmp/test_quiet.wav"    # Would be quiet speaker
    ]
    
    # Mock the detection (in real use, files would exist)
    print("Enhanced detection would analyze:")
    print("- Energy levels in target audio")
    print("- Duration of active speech")  
    print("- Confidence scores per speaker")
    print("- Validation against file-based detection")