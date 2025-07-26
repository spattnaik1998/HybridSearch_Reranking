#!/usr/bin/env python3
"""
Setup script for RAG Comparison Application
Run this script to set up the project structure and dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'templates',
        'uploads',
        'static/css',
        'static/js',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path('.env')
    if not env_path.exists():
        env_content = """# OpenAI API Key (required for GPT-4)
OPENAI_API_KEY=your_openai_api_key_here

# Cohere API Key (required for reranking)
COHERE_API_KEY=your_cohere_api_key_here

# HuggingFace API Token (optional)
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("📝 Created .env file template")
        print("⚠️  Please update the .env file with your actual API keys")
    else:
        print("📝 .env file already exists")

def create_index_html():
    """Create the index.html template"""
    templates_dir = Path('templates')
    index_path = templates_dir / 'index.html'
    
    if not index_path.exists():
        # Copy the HTML content from the frontend artifact
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Comparison: Hybrid vs Naive</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8rem;
            font-weight: 500;
        }

        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            background: #fafafa;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            margin: 10px;
        }

        .start-btn {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            display: block;
            margin: 20px auto;
        }

        .start-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .file-input {
            display: none;
        }

        .status-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 15px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            transition: width 0.3s ease;
            border-radius: 4px;
        }

        .results-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }

        .result-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            border: 2px solid #e9ecef;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e9ecef;
        }

        .metric-name {
            font-weight: 500;
            color: #555;
            text-transform: capitalize;
        }

        .metric-value {
            font-weight: 600;
            color: #333;
            background: #e3f2fd;
            padding: 4px 12px;
            border-radius: 15px;
        }

        .file-status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
            font-weight: 500;
        }

        .file-uploaded {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .hidden {
            display: none;
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .results-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Comparison Tool</h1>
            <p>Compare Hybrid Search with Reranking vs Naive RAG using RAGAS Evaluation</p>
        </div>

        <div class="content">
            <div class="section">
                <h2>📄 Upload Documents</h2>
                <div class="upload-area">
                    <p style="font-size: 1.2rem; color: #666; margin-bottom: 20px;">
                        Upload your PDF document and ground truth Q&A pairs
                    </p>
                    
                    <input type="file" id="pdfFile" class="file-input" accept=".pdf">
                    <input type="file" id="groundTruthFile" class="file-input" accept=".json">
                    
                    <button class="upload-btn" onclick="document.getElementById('pdfFile').click()">
                        📑 Choose PDF Document
                    </button>
                    <button class="upload-btn" onclick="document.getElementById('groundTruthFile').click()">
                        📋 Choose Ground Truth (JSON)
                    </button>
                    
                    <div id="fileStatus"></div>
                </div>
            </div>

            <div class="section">
                <h2>⚡ Run Evaluation</h2>
                <button class="start-btn" id="startBtn" onclick="startEvaluation()" disabled>
                    Start RAG Comparison
                </button>
                
                <div class="status-card hidden" id="statusCard">
                    <h3>📊 Evaluation Progress</h3>
                    <div id="statusMessage">Initializing...</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <div class="section hidden" id="resultsSection">
                <h2>📈 Evaluation Results</h2>
                
                <div class="results-container">
                    <div class="result-card">
                        <h3>🔍 Naive RAG</h3>
                        <div id="naiveResults"></div>
                    </div>
                    
                    <div class="result-card">
                        <h3>🚀 Hybrid RAG with Reranking</h3>
                        <div id="hybridResults"></div>
                    </div>
                </div>
                
                <div class="status-card hidden" id="comparisonSummary">
                    <h3>🏆 Comparison Summary</h3>
                    <div id="summaryContent"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let pdfUploaded = false;
        let groundTruthUploaded = false;

        document.getElementById('pdfFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                document.getElementById('fileStatus').innerHTML += 
                    '<div class="file-status file-uploaded">✅ PDF Document uploaded: ' + e.target.files[0].name + '</div>';
                pdfUploaded = true;
                checkFilesUploaded();
            }
        });

        document.getElementById('groundTruthFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                document.getElementById('fileStatus').innerHTML += 
                    '<div class="file-status file-uploaded">✅ Ground Truth uploaded: ' + e.target.files[0].name + '</div>';
                groundTruthUploaded = true;
                checkFilesUploaded();
            }
        });

        function checkFilesUploaded() {
            if (pdfUploaded && groundTruthUploaded) {
                uploadFiles();
            }
        }

        async function uploadFiles() {
            const formData = new FormData();
            formData.append('pdf_file', document.getElementById('pdfFile').files[0]);
            formData.append('ground_truth', document.getElementById('groundTruthFile').files[0]);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('fileStatus').innerHTML += 
                        '<div class="file-status file-uploaded">✅ Files processed successfully!</div>';
                    document.getElementById('startBtn').disabled = false;
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                document.getElementById('fileStatus').innerHTML += 
                    '<div class="error">❌ Error: ' + error.message + '</div>';
            }
        }

        async function startEvaluation() {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('statusCard').classList.remove('hidden');
            
            try {
                const response = await fetch('/api/start_evaluation', {
                    method: 'POST'
                });

                if (response.ok) {
                    pollStatus();
                } else {
                    const error = await response.json();
                    throw new Error(error.error);
                }
            } catch (error) {
                document.getElementById('statusMessage').innerHTML = 
                    '<div class="error">❌ Error: ' + error.message + '</div>';
            }
        }

        async function pollStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                updateStatus(status);

                if (status.status === 'running') {
                    setTimeout(pollStatus, 2000);
                } else if (status.status === 'completed') {
                    loadResults();
                }
            } catch (error) {
                document.getElementById('statusMessage').innerHTML = 
                    '<div class="error">❌ Error: ' + error.message + '</div>';
            }
        }

        function updateStatus(status) {
            const statusMessage = document.getElementById('statusMessage');
            const progressFill = document.getElementById('progressFill');

            switch (status.status) {
                case 'running':
                    statusMessage.innerHTML = '<div class="loading"></div>' + status.message;
                    progressFill.style.width = status.progress + '%';
                    break;
                case 'completed':
                    statusMessage.innerHTML = '✅ ' + status.message;
                    progressFill.style.width = '100%';
                    break;
                case 'error':
                    statusMessage.innerHTML = '<div class="error">❌ Error: ' + status.message + '</div>';
                    break;
            }
        }

        async function loadResults() {
            try {
                const response = await fetch('/api/results');
                const results = await response.json();
                displayResults(results);
                document.getElementById('resultsSection').classList.remove('hidden');
            } catch (error) {
                console.error('Error loading results:', error);
            }
        }

        function displayResults(results) {
            const naiveResults = document.getElementById('naiveResults');
            const hybridResults = document.getElementById('hybridResults');
            
            naiveResults.innerHTML = generateMetricsHTML(results.naive_rag.evaluation);
            hybridResults.innerHTML = generateMetricsHTML(results.hybrid_rerank.evaluation);
            
            displayComparisonSummary(results.comparison_summary);
        }

        function generateMetricsHTML(evaluation) {
            const metrics = ['answer_relevancy', 'faithfulness', 'context_precision', 'context_recall', 'answer_similarity', 'answer_correctness'];
            let html = '';
            
            metrics.forEach(metric => {
                if (evaluation[metric] !== undefined) {
                    const value = (evaluation[metric] * 100).toFixed(2);
                    html += `
                        <div class="metric">
                            <span class="metric-name">${metric.replace('_', ' ')}</span>
                            <span class="metric-value">${value}%</span>
                        </div>
                    `;
                }
            });
            
            return html;
        }

        function displayComparisonSummary(summary) {
            const summaryDiv = document.getElementById('comparisonSummary');
            const summaryContent = document.getElementById('summaryContent');
            
            let html = `<h4>🏆 Overall Winner: ${summary.overall_winner}</h4>`;
            html += `<p>Average Improvement: ${summary.avg_improvement.toFixed(2)}%</p>`;
            
            summaryContent.innerHTML = html;
            summaryDiv.classList.remove('hidden');
        }
    </script>
</body>
</html>'''
        
        with open(index_path, 'w') as f:
            f.write(html_content)
        print("📄 Created templates/index.html")
    else:
        print("📄 templates/index.html already exists")

def create_sample_files():
    """Create sample ground truth file"""
    sample_gt = [
        {
            "question": "What is RAG in Natural Language Processing?",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context to generate more accurate and informed responses."
        },
        {
            "question": "How does hybrid search improve retrieval accuracy?",
            "answer": "Hybrid search combines lexical search (like BM25) with semantic search (dense embeddings) to leverage both exact keyword matching and semantic understanding, resulting in more comprehensive and accurate retrieval results."
        },
        {
            "question": "What is the purpose of reranking in retrieval systems?",
            "answer": "Reranking is used to refine initial retrieval results by applying more sophisticated relevance scoring methods. It helps distinguish truly relevant passages from others and improves the precision of retrieval systems."
        }
    ]
    
    sample_path = Path('sample_ground_truth.json')
    if not sample_path.exists():
        with open(sample_path, 'w') as f:
            json.dump(sample_gt, f, indent=2)
        print("📋 Created sample_ground_truth.json")

def main():
    """Main setup function"""
    print("🚀 Setting up RAG Comparison Application")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create .env file
    print("\n🔧 Setting up configuration...")
    create_env_file()
    
    # Create HTML template
    print("\n🌐 Setting up web interface...")
    create_index_html()
    
    # Create sample files
    print("\n📋 Creating sample files...")
    create_sample_files()
    
    # Install dependencies
    print("\n📦 Installing Python dependencies...")
    if run_command("pip install -r requirements.txt", "Installing requirements"):
        print("✅ All dependencies installed successfully")
    else:
        print("⚠️  Some dependencies may have failed to install")
        print("   Try running: pip install -r requirements.txt manually")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update your .env file with actual API keys:")
    print("   - OPENAI_API_KEY (required)")
    print("   - COHERE_API_KEY (required for reranking)")
    print("\n2. Run the application:")
    print("   python app.py")
    print("\n3. Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n4. Upload your PDF document and ground truth JSON file")
    print("   (You can use sample_ground_truth.json as a template)")
    print("\n💡 Tips:")
    print("   - Ensure your PDF contains the text content (not just images)")
    print("   - Ground truth JSON should be a list of {question, answer} objects")
    print("   - The application will track token usage to help optimize costs")
    print("\n🆘 Need help? Check the README.md for detailed instructions")

if __name__ == "__main__":
    main()