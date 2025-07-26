# RAG Comparison Tool: Hybrid Search vs Naive RAG

A comprehensive evaluation framework for comparing Retrieval-Augmented Generation (RAG) approaches using the RAGAS evaluation methodology. This tool implements and benchmarks **Hybrid Search with Reranking** against **Naive RAG** to provide objective performance metrics.

![RAG Comparison Tool](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-0.1+-purple.svg)

## 🚀 Features

### **Hybrid Search Implementation**
- **BM25 + Dense Retrieval**: Combines lexical and semantic search approaches
- **Cohere Reranking**: Advanced reranking using `rerank-english-v3.0` model
- **Ensemble Weighting**: Configurable balance between sparse and dense retrieval
- **Contextual Compression**: Intelligent document filtering and ranking

### **Comprehensive Evaluation**
- **RAGAS Framework**: 6 core evaluation metrics
  - Answer Relevancy
  - Faithfulness  
  - Context Precision
  - Context Recall
  - Answer Similarity
  - Answer Correctness
- **Fallback Evaluation**: Basic similarity metrics when RAGAS unavailable
- **Token Tracking**: Monitor API usage and costs
- **Success Rate Monitoring**: Track processing efficiency

### **User Experience**
- **Web Interface**: Intuitive drag-and-drop file upload
- **Real-time Progress**: Live evaluation updates
- **Interactive Results**: Side-by-side metric comparison
- **Error Handling**: Graceful degradation with helpful messages

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │    │  Ground Truth    │    │   Web Interface │
│                 │    │   JSON File      │    │                 │
└─────────┬───────┘    └─────────┬────────┘    └─────────────────┘
          │                      │                       │
          └──────────┬───────────┘                       │
                     │                                   │
          ┌──────────▼───────────┐                       │
          │  Document Processing │                       │
          │  - Text Extraction   │                       │
          │  - Chunking          │                       │
          │  - Embedding         │                       │
          └──────────┬───────────┘                       │
                     │                                   │
          ┌──────────▼───────────┐    ┌─────────────────┐ │
          │     Naive RAG        │    │  Hybrid RAG +   │ │
          │  - FAISS Vector DB   │    │   Reranking     │ │
          │  - Simple Retrieval  │    │  - BM25 + Dense │ │
          │                      │    │  - Cohere       │ │
          └──────────┬───────────┘    └─────────┬───────┘ │
                     │                          │         │
                     └─────────┬────────────────┘         │
                               │                          │
                    ┌──────────▼───────────┐              │
                    │   RAGAS Evaluation   │              │
                    │  - Multiple Metrics  │◄─────────────┘
                    │  - Comparison Report │
                    └──────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Cohere API Key (optional, for reranking)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-comparison-tool.git
cd rag-comparison-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
```
http://localhost:5000
```

## 📝 Usage

### 1. Prepare Your Data

**PDF Document**: Any PDF containing the text content for your knowledge base.

**Ground Truth JSON**: Q&A pairs in the following format:
```json
[
    {
        "question": "What is RAG in Natural Language Processing?",
        "answer": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation..."
    }
]
```

### 2. Run Evaluation

1. Upload your PDF document and ground truth JSON file
2. Click "Start RAG Comparison"
3. Monitor real-time progress
4. Review detailed comparison results

### 3. Interpret Results

The tool provides:
- **Metric Scores**: Individual performance scores for each approach
- **Improvement Analysis**: Percentage improvements/declines per metric
- **Token Usage**: API cost tracking
- **Success Rate**: Processing efficiency metrics

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Answer Relevancy** | How relevant the generated answer is to the question |
| **Faithfulness** | How faithful the answer is to the retrieved context |
| **Context Precision** | Precision of the retrieved context |
| **Context Recall** | Recall of the retrieved context |
| **Answer Similarity** | Similarity between generated and ground truth answers |
| **Answer Correctness** | Overall correctness of the generated answer |

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...                    # OpenAI API key
COHERE_API_KEY=...                       # Cohere API key (optional)

# Optional
FLASK_DEBUG=True                         # Enable debug mode
FLASK_PORT=5000                          # Application port
```

### Model Configuration
Modify the `Config` class in `app.py`:
```python
class Config:
    EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'    # Embedding model
    LLM_MODEL = "gpt-4"                           # Language model
    CHUNK_SIZE = 512                              # Document chunk size
    RETRIEVAL_K = 5                               # Documents to retrieve
    ENSEMBLE_WEIGHTS = [0.6, 0.4]                 # Dense vs sparse weights
```

## 🏗️ Project Structure

```
rag-comparison-tool/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── sample_ground_truth.json   # Example ground truth format
├── logs/                      # Application logs
├── uploads/                   # Temporary file storage
└── README.md                  # This file
```

## 🔬 Example Results

```
🏆 Evaluation Results

Naive RAG:
├── Answer Relevancy: 72.3%
├── Faithfulness: 68.9%
├── Context Precision: 71.2%
└── Answer Similarity: 69.8%

Hybrid RAG + Reranking:
├── Answer Relevancy: 84.7% (↗ 17.2%)
├── Faithfulness: 79.3% (↗ 15.1%)
├── Context Precision: 82.6% (↗ 16.0%)
└── Answer Similarity: 78.4% (↗ 12.3%)

Overall Winner: Hybrid RAG with Reranking
Token Usage: 2,347 tokens across 15 queries
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RAGAS Team** for the evaluation framework
- **LangChain** for the retrieval components
- **Cohere** for the reranking API
- **OpenAI** for the language models

## 📈 Citation

If you use this tool in your research, please cite:

```bibtex
@software{rag_comparison_tool,
  title={RAG Comparison Tool: Hybrid Search vs Naive RAG},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/rag-comparison-tool}
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/rag-comparison-tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-comparison-tool/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/rag-comparison-tool/wiki)

## 🔮 Roadmap

- [ ] Support for multiple document formats (DOCX, TXT, etc.)
- [ ] Advanced chunking strategies
- [ ] Custom reranking models
- [ ] Batch evaluation capabilities
- [ ] Export results to CSV/PDF
- [ ] Integration with more LLM providers
- [ ] Docker containerization
- [ ] Kubernetes deployment

---

⭐ **Star this repository** if you find it useful!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/rag-comparison-tool.svg?style=social&label=Star)](https://github.com/yourusername/rag-comparison-tool)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/rag-comparison-tool.svg?style=social&label=Fork)](https://github.com/yourusername/rag-comparison-tool/fork)