
# Seq1F1B: Efficient Pipeline Parallelism for Long Sequence Training

Welcome to the Seq1F1B repository! Seq1F1B is an efficient sequence-level 1F1B (one-forward-one-backward) pipeline scheduling method designed to enhance the distributed training of large language models (LLMs) on long sequences. This method builds upon the Megatron-LM framework and introduces novel strategies to optimize memory usage and reduce pipeline bubbles.

## Introduction

In recent years, pipeline parallelism has become essential for the distributed training of LLMs. However, as the length of training sequences reaches 32k and even 128k, existing pipeline parallel methods often encounter high memory footprints and significant bubble sizes, negatively impacting training efficiency. Seq1F1B addresses these challenges by decomposing batch-level schedulable units into finer-grained sequence-level units, thereby improving workload balance and reducing memory usage.

Seq1F1B supports efficient training of a 30B parameter LLM on sequences up to 64k using 64 NVIDIA A100 GPUs without requiring recomputation strategies. This level of performance is unattainable with current pipeline parallel methods.

## Key Features

- **Fine-Grained Schedulable Units**: Decomposes batch-level units into sequence-level units to enhance workload balance.
- **Reduced Memory Footprint**: Optimized memory usage allows for training on longer sequences.
- **Smaller Pipeline Bubbles**: Efficient scheduling reduces idle time during pipeline stages.
- **Scalable Training**: Capable of training LLMs with up to 30B parameters on sequences up to 64k using 64 NVIDIA A100 GPUs.
- **Based on Megatron-LM**: Built upon the robust and widely-used Megatron-LM.



| ![seq1f1b_original.png](./picture/seq1f1b_original.png) |
|:--:|
| *Seq1F1B timeline* |

## Installation

To get started with Seq1F1B, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Seq1F1B.git
   cd Seq1F1B
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Seq1F1B integrates seamlessly with Megatron-LM. Here's an example of how to run training with Seq1F1B:

1. Prepare your dataset and configuration files. Here, we take codeparrot as an example:
   ```python
   from datasets import load_dataset
   train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')
   train_data.to_json("codeparrot_data.json", lines=True)  
   ```
   ```bash
   pip install nltk
      python tools/preprocess_data.py \
       --input codeparrot_data.json \
       --output-prefix codeparrot \
       --vocab vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --json-keys content \
       --workers 32 \
       --chunk-size 25 \
       --append-eod
   ```

2. Run the training script with Seq1F1B pipeline parallelism:
   ```bash
   bash exp.sh 
   ```

For detailed usage instructions and configuration options, please refer to our [documentation](docs/README.md).

## Results

Seq1F1B demonstrates significant improvements over existing methods:

- **Memory Efficiency**: Reduced memory footprint allows training on longer sequences without recomputation.
- **Performance**: Achieves smaller pipeline bubbles, resulting in faster training times.
- **Scalability**: Supports large-scale training on modern GPU clusters.

| ![mem.png](./picture/seq1f1b_memory.png) |
|:--:|
| *Memory usage comparison between Seq1F1B and existing methods.* |

## Contributing

We welcome contributions from the community! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

Seq1F1B is released under original
Megatron's License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Seq1F1B builds upon the Megatron-LM framework. We thank the Megatron-LM development team for their excellent work and support.

## Future Work

We plan to release our code and further improvements to advance the training of LLMs on long sequences. Stay tuned for updates!

<!-- ## Citation 
If you use this codebase, or otherwise found our work valuable, please cite:

```

``` -->
