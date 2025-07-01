# Enhancement

The script to enhance audio using a given model can be run using

```bash
python run_enhancement.py
```

This is equivalent to running the following steps:

```bash
python scripts.enhancement.resample
python scripts.enhancement.enhance
```

The resample step simply resamples all audio for the dev/test (defined in
`config.enhancment.main`) and saves it. This stage only need to be run once
for any enhancement systems which use the same sample rate.

The enhance stage loads in the resampled 36 minute wav files from above, and
puts them through a provided enhancement system. By default, this script can
run passthrough, which returns the first channel of audio, and the baseline
system processing. We have included the option for writing a plugin for this
script where you can define a custom enhancement function based on your system.

## Enhancement Plugins

An enhancement plugin is a Python class which processes a the full audio
from a session. The core functionality of the class is the function `process_session`,
which takes a full session of audio and the participant speaker id audio in,
and produces a full session of that person's speech out.

To use a custom enhancement plugin, you must first set the environment variable
for where to find the plugins:

```bash
export ECHIPLUGINS="$PWD/enhancement_plugins"
```

This sets the directory of where the enhance script should look for options.
To write a custom plugin, there are two things that need adding to the code:
the processing plugin as a Python class, and the parameters for the plugin as a
yaml config.

An example plugin is included in `enhancement_plugins`; this simply does the
`passthrough` processing, but can be used as a guide for how to implement your
own enhancement plugin:

```python
from typing import Dict
import torch
import soxr

from enhancement.registry import register_enhancement, Enhancement

@register_enhancement("example")
class Example(Enhancement):
    def __init__(self, output_sample_rate: int) -> None:
        self.output_sample_rate = output_sample_rate

    def process_session(
        self,
        device_audio: torch.Tensor,
        device_fs: int,
        spkid_audio: torch.Tensor,
        spkid_fs: int,
        kwargs: Dict | None = None,
    ) -> torch.Tensor:
        output = soxr.resample(
            device_audio[0].detach().cpu().numpy(), device_fs, self.output_sample_rate
        )
        return torch.from_numpy(output)
```

There are a few key points for using the plugin:

- **Registry:** You must import `register_enhancement` and add the decorator to
your class. The name ("example" above) is how you access the plugin, as
specified by `config/enhancement/main.enhancement_name`
- **Init params:** The parameters used to initialise the class should be stored
in `config/enhancement/enhance_args/<your-algorithm-name>.yaml`
- **Process session:** Each class must have a `process_session` function, which
take the audio for a full session and the participant speech audio as input,
and outputs the enhanced audio for the full session. This is expected by the
`Enhancement` protocol.

Once you have defined this, you can then run the enhancement stages using

```bash
python run_enhancement.py enhancement_name=<your-algorithm-name> enhance_args=<your-algorithm-name>
```
