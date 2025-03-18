import os
from datasets import load_dataset
import torchaudio
import torch

# Load the test split of the dataset from Hugging Face
dataset = load_dataset("DynamicSuperb/SourceSeparation_libri2Mix_test", split="test")

# Create an output directory for the downloaded files
output_dir = "libri2mix_test_files"
os.makedirs(output_dir, exist_ok=True)

# Iterate over the dataset and save each audio file
for idx, sample in enumerate(dataset):
    # The audio field is expected to be a dictionary with keys "array" (the waveform as a NumPy array)
    # and "sampling_rate"
    audio_info = sample["audio"]
    waveform_np = audio_info["array"]
    sample_rate = audio_info["sampling_rate"]

    # Convert the NumPy array to a PyTorch tensor
    waveform = torch.tensor(waveform_np)

    # Ensure waveform has a channel dimension (Torchaudio expects [channels, time])
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Convert waveform to 16-bit PCM.
    # The waveform is assumed to be in the range [-1, 1]. Multiply by 32767, clamp to [-32768, 32767],
    # then convert to torch.int16.
    int_waveform = (waveform * 32767).clamp(-32768, 32767).to(torch.int16)

    # Create an output filename
    output_path = os.path.join(output_dir, f"sample_{idx}.wav")

    # Save the waveform as a WAV file using torchaudio
    torchaudio.save(output_path, int_waveform, sample_rate)
    print(f"Saved {output_path}")
