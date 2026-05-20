# Trained PTQ Accuracy and Sensitivity

Dataset: MNIST binary `is_zero`, trained and evaluated on balanced zero/nonzero splits.

## Accuracy

| Model | FP32 balanced | FP32 full-test | INT16 PTQ balanced | INT8 PTQ balanced | INT4 PTQ balanced |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv wide32 | 98.16% | 98.67% | 98.16% | 98.16% | 98.11% |
| MLP h256 | 98.57% | 98.33% | 98.57% | 98.57% | 98.37% |

## Per-Layer Sensitivity

### Conv wide32

| Layer | INT8 one-layer balanced | INT8 drop vs INT16 | INT4 one-layer balanced | INT4 drop vs INT16 |
| --- | ---: | ---: | ---: | ---: |
| conv1 | 98.16% | 0.00% | 98.01% | 0.15% |
| conv2 | 98.16% | 0.00% | 98.21% | -0.05% |
| conv3 | 98.16% | 0.00% | 98.16% | 0.00% |
| conv4 | 98.16% | 0.00% | 98.21% | -0.05% |

### MLP h256

| Layer | INT8 one-layer balanced | INT8 drop vs INT16 | INT4 one-layer balanced | INT4 drop vs INT16 |
| --- | ---: | ---: | ---: | ---: |
| fc1 | 98.57% | 0.00% | 98.42% | 0.15% |
| fc2 | 98.57% | 0.00% | 98.21% | 0.36% |
| fc3 | 98.57% | 0.00% | 98.21% | 0.36% |
| fc4 | 98.57% | 0.00% | 98.37% | 0.20% |

## Raw JSON

```json
{
  "balanced_valid_samples": 1960,
  "epochs": 6,
  "full_test_samples": 10000,
  "models": {
    "Conv wide32": {
      "activation_maxes": {
        "conv1": 1.0,
        "conv2": 1.4138168096542358,
        "conv3": 2.5181658267974854,
        "conv4": 6.227910041809082
      },
      "fp32_balanced": {
        "accuracy": 0.9816326530612245,
        "balanced_accuracy": 0.9816326530612245,
        "loss": 0.05750316308469188,
        "neg_accuracy": 0.986734693877551,
        "pos_accuracy": 0.976530612244898,
        "samples": 1960.0
      },
      "fp32_full": {
        "accuracy": 0.9867,
        "balanced_accuracy": 0.9821677451468392,
        "loss": 0.039343184623867274,
        "neg_accuracy": 0.9878048780487805,
        "pos_accuracy": 0.976530612244898,
        "samples": 10000.0
      },
      "ptq_full_precision": {
        "16": {
          "accuracy": 0.9816326530612245,
          "balanced_accuracy": 0.9816326530612245,
          "loss": 0.05750331975975815,
          "neg_accuracy": 0.986734693877551,
          "pos_accuracy": 0.976530612244898,
          "samples": 1960.0
        },
        "4": {
          "accuracy": 0.9811224489795919,
          "balanced_accuracy": 0.9811224489795918,
          "loss": 0.061173102320456994,
          "neg_accuracy": 0.9846938775510204,
          "pos_accuracy": 0.9775510204081632,
          "samples": 1960.0
        },
        "8": {
          "accuracy": 0.9816326530612245,
          "balanced_accuracy": 0.9816326530612245,
          "loss": 0.05759102033109081,
          "neg_accuracy": 0.986734693877551,
          "pos_accuracy": 0.976530612244898,
          "samples": 1960.0
        }
      },
      "sensitivity": {
        "conv1": {
          "4bit_balanced_accuracy": 0.9801020408163266,
          "4bit_drop_vs_int16": 0.001530612244897922,
          "8bit_balanced_accuracy": 0.9816326530612245,
          "8bit_drop_vs_int16": 0.0
        },
        "conv2": {
          "4bit_balanced_accuracy": 0.9821428571428572,
          "4bit_drop_vs_int16": -0.0005102040816327147,
          "8bit_balanced_accuracy": 0.9816326530612245,
          "8bit_drop_vs_int16": 0.0
        },
        "conv3": {
          "4bit_balanced_accuracy": 0.9816326530612245,
          "4bit_drop_vs_int16": 0.0,
          "8bit_balanced_accuracy": 0.9816326530612245,
          "8bit_drop_vs_int16": 0.0
        },
        "conv4": {
          "4bit_balanced_accuracy": 0.9821428571428571,
          "4bit_drop_vs_int16": -0.0005102040816326037,
          "8bit_balanced_accuracy": 0.9816326530612245,
          "8bit_drop_vs_int16": 0.0
        }
      }
    },
    "MLP h256": {
      "activation_maxes": {
        "fc1": 1.0,
        "fc2": 1.3028104305267334,
        "fc3": 1.6738659143447876,
        "fc4": 3.699176788330078
      },
      "fp32_balanced": {
        "accuracy": 0.9857142857142858,
        "balanced_accuracy": 0.9857142857142858,
        "loss": 0.050871008756209396,
        "neg_accuracy": 0.9806122448979592,
        "pos_accuracy": 0.9908163265306122,
        "samples": 1960.0
      },
      "fp32_full": {
        "accuracy": 0.9833,
        "balanced_accuracy": 0.9866498484094304,
        "loss": 0.052336020845919845,
        "neg_accuracy": 0.9824833702882484,
        "pos_accuracy": 0.9908163265306122,
        "samples": 10000.0
      },
      "ptq_full_precision": {
        "16": {
          "accuracy": 0.9857142857142858,
          "balanced_accuracy": 0.9857142857142858,
          "loss": 0.05086948822955696,
          "neg_accuracy": 0.9806122448979592,
          "pos_accuracy": 0.9908163265306122,
          "samples": 1960.0
        },
        "4": {
          "accuracy": 0.9836734693877551,
          "balanced_accuracy": 0.9836734693877551,
          "loss": 0.05288275091015563,
          "neg_accuracy": 0.9744897959183674,
          "pos_accuracy": 0.9928571428571429,
          "samples": 1960.0
        },
        "8": {
          "accuracy": 0.9857142857142858,
          "balanced_accuracy": 0.9857142857142858,
          "loss": 0.05089023575490835,
          "neg_accuracy": 0.9806122448979592,
          "pos_accuracy": 0.9908163265306122,
          "samples": 1960.0
        }
      },
      "sensitivity": {
        "fc1": {
          "4bit_balanced_accuracy": 0.9841836734693878,
          "4bit_drop_vs_int16": 0.001530612244897922,
          "8bit_balanced_accuracy": 0.9857142857142858,
          "8bit_drop_vs_int16": 0.0
        },
        "fc2": {
          "4bit_balanced_accuracy": 0.9821428571428572,
          "4bit_drop_vs_int16": 0.0035714285714285587,
          "8bit_balanced_accuracy": 0.9857142857142858,
          "8bit_drop_vs_int16": 0.0
        },
        "fc3": {
          "4bit_balanced_accuracy": 0.9821428571428572,
          "4bit_drop_vs_int16": 0.0035714285714285587,
          "8bit_balanced_accuracy": 0.9857142857142858,
          "8bit_drop_vs_int16": 0.0
        },
        "fc4": {
          "4bit_balanced_accuracy": 0.9836734693877551,
          "4bit_drop_vs_int16": 0.0020408163265306367,
          "8bit_balanced_accuracy": 0.9857142857142858,
          "8bit_drop_vs_int16": 0.0
        }
      }
    }
  },
  "seed": 7,
  "train_samples": 10000
}
```
