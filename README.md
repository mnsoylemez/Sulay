# Sulay
# Sulay: Transformer-Based Language Model

This repository contains a PyTorch-based Transformer language model for text generation, along with a text analysis script to inspect datasets. The model uses rotary positional encoding and SentencePiece Byte Pair Encoding (BPE) tokenization, making it suitable for natural language processing tasks like text generation and language modeling.

## Features

- Transformer architecture with rotary positional encoding for improved sequence modeling.
- SentencePiece BPE tokenization for flexible text processing.
- Custom dataset class for text data handling.
- Top-k sampling for controlled text generation.
- Text analysis script to report dataset statistics (character counts, unique characters, frequencies).
- Open-source under the MIT License.

## Prerequisites

### Hardware

- **RAM**: At least 16GB (32GB or higher recommended).
- **GPU**: Optional CUDA-compatible GPU for faster training (CPU supported but for large datasests GPU is a must).
- **Storage**: \~1GB for dataset, model weights, and tokenizer files.

### Software

- **Python**: 3.8 or higher.
- **Operating System**: Linux, macOS, or Windows (Linux recommended).
- **Dependencies**:

  ```bash
  pip install torch>=2.0.0 sentencepiece>=0.1.99 tqdm>=4.66.1 nltk>=3.8.1 numpy>=1.24.0
  ```

  Install dependencies using:

  ```bash
  pip install -r requirements.txt
  ```

### Dataset

- A plain text file (e.g., `dataset.txt`) containing the training data (e.g., books, articles).
- Recommended: At least 1MB of text for meaningful training.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mnsoylemez/Sulay.git
   cd Sulay
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Place your text file in the `data/` directory (e.g., `data/dataset.txt`).
   - Ensure the file is UTF-8 encoded.

## Usage

### Step 1: Analyze the Dataset

Run the text analysis script to inspect your dataset:

```bash
python text_analysis.py
```

- Update the `file_path` in `text_analysis.py` to point to your dataset (e.g., `data/dataset.txt`).
- **Outputs**:
  - Dataset length, first 100 characters, unique characters, vocabulary size.
  - Decomposed and non-standard characters.
  - Character frequencies sorted by Unicode.

### Step 2: Train the Model and Generate Text

Run the training script to train the Transformer model and generate text:

```bash
python transformer_lm.py
```

- Update `file_path`, `data_file`, and `save_path` in `transformer_lm.py` to match your dataset, temporary file, and model save locations (e.g., `data/dataset.txt`, `data/TokDataFile.txt`, `models/transformer_model.pth`).
- **Outputs**:
  - `data/TokDataFile.txt`: Temporary tokenized text.
  - `spm_bpe.model` and `spm_bpe.vocab`: SentencePiece model and vocabulary.
  - `models/transformer_model.pth`: Trained model weights.
  - Generated text (500 tokens) after training.

### Step 3: Generate Custom Text

To generate text with a custom prompt, modify `transformer_lm.py`:

```python
context = torch.tensor([[sp.encode("Your prompt here", out_type=int)[0]]], dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500, temperature=1.0, top_k=10)
generated_text = sp.decode(generated_ids[0].tolist())
print(generated_text)
```

Rerun:

```bash
python transformer_lm.py
```

## Project Structure

```
Sulay/
├── data/
│   ├── dataset.txt             # Input text dataset
│   ├── TokDataFile.txt         # Temporary tokenized text
│   ├── spm_bpe.model           # SentencePiece model
│   └── spm_bpe.vocab           # SentencePiece vocabulary
├── models/
│   └── transformer_model.pth   # Trained model weights
├── transformer_lm.py           # Main script for training/generation
├── text_analysis.py            # Dataset analysis script
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── LICENSE                     # License file
└── .gitignore                  # Git ignore file
```

## Configuration

Key parameters in `transformer_lm.py` (edit `config` dictionary):

- `embed_dim`: Embedding size (default: 256).
- `num_heads`: Number of attention heads (default: 4).
- `num_layers`: Number of Transformer layers (default: 4).
- `block_size`: Maximum sequence length (default: 128).
- `learning_rate`: Initial learning rate (default: 5e-5).
- `batch_size`: Training batch size (default: 32).
- `max_iters`: Maximum training iterations (default: 2000).

## Troubleshooting

- **File Not Found**: Ensure `dataset.txt` exists in `data/` and paths in scripts are correct.
- **CUDA Errors**: Set `device='cpu'` in `transformer_lm.py` to run on CPU.
- **Poor Text Quality**:
  - Increase dataset size (&gt;1MB).
  - Adjust generation parameters (`temperature`, `top_k`).
  - Tune hyperparameters (`learning_rate`, `dropout`).
- **NLTK Errors**: Ensure `punkt` and `punkt_tab` are downloaded (internet required on first run).

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, open a GitHub issue or contact soylemeznurhan@gmail.com.
