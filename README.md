# speakerbox-analysis

[![Build Status](https://github.com/PugetSoundClinic-PIT/speakerbox-analysis/workflows/CI/badge.svg)](https://github.com/PugetSoundClinic-PIT/speakerbox-analysis/actions)
[![Documentation](https://github.com/PugetSoundClinic-PIT/speakerbox-analysis/workflows/Documentation/badge.svg)](https://PugetSoundClinic-PIT.github.io/speakerbox-analysis)

Data processing and analysis functions for the Speakerbox Demographics and Interruptions paper.

---

## Installation

**Stable Release:** `pip install speakerbox-analysis`<br>
**Development Head:** `pip install git+https://github.com/PugetSoundClinic-PIT/speakerbox-analysis.git`

## Quickstart

### Seattle Speakerbox Model

Pull Seattle data for training, prepare and make splits, and 
finally train and evaluate a new Seattle Speakerbox model.


```bash
speakerbox-analysis all_in_one prepare_dataset_and_train_and_eval_model
```

The above command is shorthand for the following:

```bash
speakerbox-analysis data prepare_for_model_training
speakerbox-analysis model train_and_eval
```

## Documentation

For full package documentation please visit [PugetSoundClinic-PIT.github.io/speakerbox-analysis](https://PugetSoundClinic-PIT.github.io/speakerbox-analysis).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
