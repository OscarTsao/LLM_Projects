# Data Augmentation Guide

## Overview

-This project now supports 19 text augmentation methods from two powerful libraries:
- **10 methods from nlpaug**: Advanced NLP augmentation techniques
- **9 methods from textattack**: Adversarial text attack methods

> **Runtime note:** Several TextAttack strategies download large models
> via TensorFlow Hub. The pipeline now includes `tensorflow`/`tensorflow-hub` at runtime,
> switches DataLoader workers to the `spawn` start method (so GPU-aware augmenters such as
> CLARE can run inside worker processes), and will automatically disable an augmentation
> method for the remainder of a trial the first time it fails (e.g., due to missing network
> permissions). The original sample is returned without augmentation in that case so HPO
> trials can continue without stalling.

## Augmentation Methods

### NLPaug Methods (10 total)

1. `nlp_ContextualWordEmbedding` - BERT-based contextual word replacement
2. `nlp_RandomWord` - Random word operations (insert/substitute/swap/delete)
3. `nlp_Spelling` - Introduce spelling errors
4. `nlp_Keyboard` - Keyboard typo simulation
5. `nlp_Ocr` - OCR error simulation
6. `nlp_Split` - Word splitting augmentation
7. `nlp_RandomChar` - Random character operations
8. `nlp_ContextualSentence` - Contextual sentence-level augmentation
9. `nlp_Lambada` - Lambada-based augmentation (masked language model)
10. `nlp_CharSwap` - Character swap augmentation

### TextAttack Methods (9 total)

1. `ta_TextFoolerJin2019` - Word substitution using word embeddings
2. `ta_PWWSRen2019` - Probability Weighted Word Saliency
3. `ta_BAEGarg2019` - BERT-based adversarial examples
4. `ta_DeepWordBugGao2018` - Character-level transformations
5. `ta_HotFlipEbrahimi2017` - Gradient-based word substitution
6. `ta_IGAWang2019` - Improved Genetic Algorithm
7. `ta_Kuleshov2017` - Adversarial paraphrasing
8. `ta_Alzantot2018` - Faster Genetic Algorithm
9. `ta_BERTAttack` - BERT-MLM based adversarial attack

## HPO Configuration

### Two-Stage Sampling

The hyperparameter optimization uses a two-stage sampling approach:

-**Stage 1: Number of Augmentations**
- Samples an integer from 0 to 19
- Determines how many augmentation methods to apply

**Stage 2: Method Selection**
- For each augmentation slot (0 to num_augmentations-1)
- Samples which specific method to use from all 28 available methods

### HPO Parameters

```yaml
num_augmentations:
  type: int
  range: [0, 28]
  description: Number of augmentation methods to apply

aug_method_0 to aug_method_18:
  type: categorical
  choices: [all 19 methods]
  description: Specific method for each augmentation slot

aug_prob:
  type: float
  range: [0.05, 0.30]
  description: Probability of applying each augmentation

aug_compose_mode:
  type: categorical
  choices: [sequential, random_one]
  description: How to compose multiple augmentations
    - sequential: Apply all selected methods in sequence
    - random_one: Apply only one randomly selected method
```

## Usage Example

```python
from dataaug_multi_both.augment import create_augmenter, AugmentedDataset
import random

# Create an augmenter from HPO parameters
params = {
    "num_augmentations": 3,
    "aug_method_0": "nlp_Synonym",
    "aug_method_1": "ta_TextFoolerJin2019",
    "aug_method_2": "nlp_BackTranslation",
    "aug_prob": 0.15,
    "aug_compose_mode": "sequential",
}

rng = random.Random(42)
augmenter = create_augmenter(params, rng)

# Apply augmentation to text
text = "I feel anxious and have trouble sleeping."
augmented_text = augmenter(text)
print(augmented_text)

# Wrap a dataset with augmentation
augmented_dataset = AugmentedDataset(
    dataset=train_dataset,
    augmenter=augmenter,
    field="sentence_text"
)
```

## Installation

All augmentation-related dependencies are managed through Poetry. After `poetry install` the
environment contains:

- `nlpaug` and `textattack` for the augmentation backends
- `tensorflow` and `tensorflow-hub` to satisfy TextAttack’s contextual strategies (e.g., CLARE)

If you are extending the project outside of Poetry, ensure the following packages are present (and that NLTK corpora such as `wordnet` and `averaged_perceptron_tagger_eng` are downloaded):

```bash
pip install nlpaug textattack tensorflow tensorflow-hub
```

## Architecture

```
src/dataaug_multi_both/augment/
├── __init__.py                  # Main exports
├── nlpaug_factory.py            # NLPaug augmenter factory (17 methods)
├── textattack_methods.py        # TextAttack augmenter factory (11 methods)
├── unified_augmenter.py         # Unified interface for both libraries
└── textattack_factory.py        # Legacy augmenter (backward compatibility)

src/dataaug_multi_both/hpo/
└── space.py                     # HPO search space with two-stage sampling
```

## Legacy Support

The codebase maintains backward compatibility with the previous augmentation system. Legacy code using the old `EvidenceAugmenter` and mask-based selection will continue to work.

## Performance Considerations

- **Lazy Initialization**: Augmenters are initialized on first use to save memory
- **Caching**: Models are loaded once and reused across augmentations
- **Error Handling**: Failed augmentations fall back to original text and that method is skipped for the remainder of the trial, preventing repeated network retries. Required NLTK resources are fetched automatically the first time they are needed.
- **DataLoader Parallelism**: The training loop now defaults to several worker processes per trial so GPU time is not lost waiting on CPU-bound augmentation.
- **Batch Processing**: For production, consider batching augmentations

## References

- [nlpaug Documentation](https://github.com/makcedward/nlpaug)
- [TextAttack Documentation](https://github.com/QData/TextAttack)
