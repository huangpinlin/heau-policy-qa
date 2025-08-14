export const filesData = [
    {
        name: 'README.md',
        description: 'Project overview, technical architecture, setup, and usage guide.',
        content: `# 河南大学政策问答系统

一个基于本地知识库的智能问答系统，采用检索增强生成（RAG）技术，为河南大学师生提供精准、可溯源的政策查询服务。

![河南大学政策问答系统界面](https://r2.flowith.net/files/85b35d1e-d6a4-458b-ab2b-2396aae76f0a/1754543336217-%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_2025-08-07_130813@1340x660.png)

## ✨ 功能特点

-   **自然语言理解**：用户可以使用日常对话方式进行提问，无需关心关键词或专业术语。
-   **精准信息检索**：基于向量化检索技术，快速从海量政策文件中定位最相关的条款和段落。
-   **智能答案生成**：结合大型语言模型（LLM）的推理和生成能力，将检索到的信息整合成通顺、易懂的回答。
-   **答案来源可追溯**：每个回答都会附上来源文档的标题、链接和原文片段，方便用户核实信息的准确性。
-   **友好交互界面**：提供基于 Streamlit 的 Web 界面，操作直观，上手简单。

## 🛠️ 技术架构

本系统采用检索增强生成（RAG）架构，结合了多种先进的AI技术和框架，以实现高效、准确的问答功能。**系统最近已重构，将嵌入模块从\`sentence-transformers\`迁移至\`fastembed\`，以提升性能和简化依赖。**

| 技术/组件 | 用途说明 |
| :--- | :--- |
| **LangChain** | 作为核心编排框架，整合了数据处理、模型调用、检索和问答链（QA Chain）的全部流程。 |
| **Qwen-7B-Chat** | 阿里巴巴通义千问70亿参数对话模型，作为大型语言模型（LLM），负责理解问题并基于检索内容生成最终答案。 |
| **fastembed (BAAI/bge-base-en-v1.5)** | 一个轻量级、高性能的嵌入模型库。通过\`ONNX Runtime\`优化，替代了原有的\`sentence-transformers\`方案，实现了更快的向量化速度和更低的资源占用，尤其在CPU上表现出色。 |
| **FAISS (Facebook AI Similarity Search)** | 高效的向量相似度搜索引擎，用于存储文本向量并实现快速检索。 |
| **Streamlit** | 一个开源的Python框架，用于快速构建和部署数据科学与机器学习的Web应用，是本系统的用户界面。 |

\`\`\`mermaid
flowchart LR
    A[用户在Streamlit界面输入问题] --> B{LangChain QA Chain};
    B --> C[使用fastembed将问题向量化];
    C --> D[在FAISS索引中检索相似文档];
    D --> E{Prompt构建};
    E -- 检索到的上下文 --> F[Qwen-7B LLM];
    E -- 原始问题 --> F;
    F --> G[生成最终答案];
    G --> H[在Streamlit界面展示答案及来源];
\`\`\`

## 📋 环境要求

-   **Python 版本**: 推荐 \`Python 3.8\` ~ \`3.11\`
-   **硬件要求**:
    -   **强烈推荐**：配备至少 16GB显存的 NVIDIA GPU，以保证 \`Qwen-7B\` 模型的高效推理。
    -   **最低配置**：可在 CPU 上运行。得益于\`fastembed\`，索引构建步骤在CPU上依然高效。但LLM推理速度会非常慢。
-   **依赖库**: 所有依赖项均已在 \`requirements.txt\` 文件中列出。

    \`\`\`
    streamlit==1.35.0
    langchain==0.1.16
    langchain-community==0.0.34
    faiss-cpu==1.8.0
    fastembed>=0.2.0
    onnxruntime>=1.16.0
    transformers==4.40.1
    torch==2.2.1
    accelerate==0.29.3
    numpy==1.26.4
    tqdm==4.66.4
    \`\`\`

## 🚀 快速开始

请按照以下步骤在您的本地环境中部署并启动系统。

### 1. 环境准备

克隆或下载本项目，并创建所需的目录结构。

\`\`\`bash
# 假设项目根目录为 henan_university_qa
mkdir -p henan_university_qa/data henan_university_qa/faiss_index
cd henan_university_qa
\`\`\`

### 2. 依赖安装

使用 \`pip\` 安装所有必需的 Python 库。

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. 数据准备

将包含政策文本的知识库文件 \`processed_policies.json\` 放置到 \`data/\` 目录下。

> 项目结构应如下所示：
> \`\`\`
> /henan_university_qa/
> ├── data/
> │   └── processed_policies.json
> ├── faiss_index/
> ├── app.py
> ├── create_index.py
> ├── qa_system_core.py
> └── requirements.txt
> \`\`\`

### 4. 构建知识库索引

此步骤会将 \`data/\` 目录下的政策文本转换为向量，并构建一个 FAISS 索引用于快速检索。

\`\`\`bash
python create_index.py
\`\`\`
> **注意**：首次运行此脚本会由\`fastembed\`自动下载嵌入模型 (\`BAAI/bge-base-en-v1.5\`)。此过程已优化，下载和处理速度较快。成功后，\`faiss_index/\` 目录下会生成 \`faiss_index.bin\` 和 \`documents_metadata.json\` 两个文件。

### 5. 启动系统

运行以下命令启动 Streamlit Web 应用。

\`\`\`bash
streamlit run app.py
\`\`\`

启动成功后，终端会显示本地访问地址（通常是 \`http://localhost:8501\`），在浏览器中打开此地址即可开始使用。

## 📂 项目结构

\`\`\`
.
├── data/
│   └── processed_policies.json   # 预处理后的政策知识库源文件
├── faiss_index/
│   ├── faiss_index.bin           # (自动生成) FAISS 向量索引文件
│   └── documents_metadata.json   # (自动生成) 与索引对应的文档元数据
├── app.py                      # Streamlit Web 应用主程序
├── create_index.py             # 用于构建 FAISS 索引的脚本
├── qa_system_core.py           # 封装了RAG核心逻辑的模块
└── requirements.txt            # Python 依赖项列表
\`\`\`

## 💬 使用说明

1.  在浏览器中打开 \`http://localhost:8501\`。
2.  在主界面的文本输入框中，输入您想咨询的关于学校政策的问题。例如：“What are the rewards for undergraduate research projects?”
3.  点击“提交问题”按钮。
4.  系统会进行思考，并在下方显示 AI 生成的回答。
5.  您可以展开“查看答案来源”部分，核对答案所依据的原始政策文件和相关片段。

## ⚙️ 配置说明

### 自定义模型和参数

系统的核心配置位于 \`qa_system_core.py\` 文件中的 \`QAConfig\` 类。您可以直接修改此类中的参数来自定义系统行为。

\`\`\`python
# qa_system_core.py

class QAConfig:
    """统一管理系统所需的所有配置参数。"""
    FAISS_INDEX_PATH = 'faiss_index/faiss_index.bin'
    METADATA_PATH = 'faiss_index/documents_metadata.json'
    # 嵌入模型已更新为fastembed支持的轻量级模型
    EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'  # 可更换为其他fastembed支持的模型
    LLM_MODEL_PATH = 'Qwen/Qwen-7B-Chat'             # 可更换为其他LLM
    SEARCH_TOP_K = 4                                # 调整检索文档的数量
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
\`\`\`

### 更新知识库

1.  **准备新数据**：将新的政策文本处理成与 \`processed_policies.json\` 相同格式的 JSON 文件，并替换掉旧文件。
2.  **删除旧索引**：删除 \`faiss_index/\` 目录下的所有文件。
3.  **重建索引**：重新运行 \`create_index.py\` 脚本，以根据新数据生成新的向量索引。

    \`\`\`bash
    python create_index.py
    \`\`\`

4.  **重启应用**：重启 Streamlit 应用即可加载新的知识库。

## ❓ 常见问题 (FAQ)

1.  **Q: 启动 \`app.py\` 时为什么非常缓慢或卡住？**
    *   **A:** 首次启动时，系统需要从 Hugging Face Hub 下载并加载大型语言模型（如 \`Qwen-7B-Chat\`）到内存/显存中。这个过程根据您的网络和硬件情况，可能需要几分钟到几十分钟。请耐心等待，后续启动会快得多（因为模型已有缓存）。嵌入模型的下载由\`fastembed\`处理，通常非常迅速。

2.  **Q: 运行时出现 \`FileNotFoundError\`，提示找不到 \`faiss_index.bin\` 文件。**
    *   **A:** 这个错误表示您尚未成功构建知识库索引。请确保您已经成功运行了 \`python create_index.py\` 脚本，并且 \`faiss_index\` 目录下已生成了 \`.bin\` 和 \`.json\` 文件。

3.  **Q: 可以在没有GPU的电脑上运行吗？**
    *   **A:** 可以。代码会自动检测是否存在 CUDA 设备，如果不存在，则会切换到 CPU 模式。在新的架构下，使用\`fastembed\`进行索引构建和 embedding 在CPU上非常高效。然而，运行大型语言模型（LLM）的推理部分在CPU上仍然会**极其缓慢**，可能无法获得流畅的问答体验。

## 📈 维护和扩展

-   **数据自动化更新**: 未来可以开发网络爬虫，定期从河南大学官网抓取最新的政策文件，实现知识库的自动化更新。
-   **用户反馈机制**: 在Web界面中增加“赞/踩”或评分功能，收集用户对答案质量的反馈，用于迭代优化检索策略和Prompt模板。
-   **模型升级**: 随着AI技术的发展，可以轻松地在 \`QAConfig\` 中替换为更先进、更高效的嵌入模型或大型语言模型。

## 📜 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。

## 🙌 贡献指南

欢迎任何形式的贡献！如果您有任何建议或发现了问题，请随时提交 Issues 或 Pull Requests。`
    },
    {
        name: 'requirements.txt',
        description: 'A list of all Python dependencies required to run the project.',
        content: `streamlit==1.35.0
langchain==0.1.16
langchain-community==0.0.34
faiss-cpu==1.8.0
fastembed>=0.2.0
onnxruntime>=1.16.0
transformers==4.40.1
torch==2.2.1
accelerate==0.29.3
numpy==1.26.4
tqdm==4.66.4`
    },
    {
        name: 'create_index.py',
        description: 'Script to generate vector embeddings and build the FAISS index from source data.',
        content: `import json
import os
import faiss
import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm

# --- Constants ---
INPUT_JSON_PATH = 'data/processed_policies.json'
FAISS_INDEX_DIR = 'faiss_index'
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, 'faiss_index.bin')
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, 'documents_metadata.json')
# The model is updated according to the refactoring requirements.
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

def load_and_filter_data(file_path: str) -> list:
    """
    Loads and filters documents from a JSON file.
    Only documents with non-empty 'full_cleaned_text' are kept.
    """
    print(f"正在从 {file_path} 加载数据...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 输入文件 {file_path} 未找到。请确保数据已放置在 'data/' 目录下。")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        all_documents = json.load(f)

    processed_docs = [doc for doc in all_documents if doc.get('full_cleaned_text', '').strip()]
    
    print(f"加载完成。共找到 {len(all_documents)} 条记录，其中有效记录 {len(processed_docs)} 条。")
    return processed_docs

def generate_embeddings(texts: list, model_name: str) -> np.ndarray:
    """
    Generates vector embeddings for a list of texts using the specified fastembed model.
    This version is lightweight and does not depend on PyTorch or Transformers.
    """
    print(f"正在加载轻量级嵌入模型: {model_name}...")
    # fastembed automatically handles device selection (CPU/GPU) via ONNX Runtime.
    # It's generally faster on CPU than sentence-transformers.
    embedding_model = TextEmbedding(model_name=model_name, cache_dir=os.getenv("HF_HOME"))
    
    print("开始生成文本向量嵌入...")
    # The \`embed\` method returns a generator of numpy arrays.
    embeddings_list = embedding_model.embed(texts, show_progress_bar=True)
    
    # Convert the generator of arrays into a single, stacked numpy matrix.
    embeddings = np.array(list(embeddings_list), dtype='float32')
    
    # --- CRITICAL STEP: Manual L2 Normalization ---
    # This step is essential to replicate the behavior of \`normalize_embeddings=True\`
    # from the previous SentenceTransformer implementation. BGE models are designed
    # to be used with normalized embeddings for accurate similarity search.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # To prevent division by zero for potential zero-vectors
    norms[norms == 0] = 1e-12
    normalized_embeddings = embeddings / norms
    
    print("向量嵌入生成完毕。")
    
    # FAISS requires float32 type.
    return normalized_embeddings.astype('float32')

def build_and_save_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    Builds a FAISS index from embeddings and saves it to disk.
    """
    if embeddings.shape[0] == 0:
        print("没有可用于构建索引的向量。")
        return

    dimension = embeddings.shape[1]
    print(f"正在构建 FAISS 索引 (IndexFlatL2)，向量维度: {dimension}...")
    # IndexFlatL2 is a simple index that performs exhaustive search, suitable for many use cases.
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"索引构建完成，共添加了 {index.ntotal} 个向量。")
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS 索引已成功保存到: {index_path}")

def save_metadata(documents: list, metadata_path: str):
    """
    Saves the metadata of the indexed documents to a JSON file.
    This metadata is used to retrieve the original text content after a search.
    """
    metadata_list = []
    for doc in documents:
        metadata_list.append({
            'source_url': doc.get('source_url'),
            'title': doc.get('metadata', {}).get('title'),
            'publish_date': doc.get('metadata', {}).get('publish_date'),
            'publish_dept': doc.get('metadata', {}).get('publish_dept'),
            'document_id': doc.get('metadata', {}).get('document_id'),
            'full_cleaned_text': doc.get('full_cleaned_text')
        })
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
        
    print(f"元数据已成功保存到: {metadata_path}")

if __name__ == "__main__":
    try:
        valid_documents = load_and_filter_data(INPUT_JSON_PATH)
        
        if not valid_documents:
            print("没有找到有效的文档来构建索引，程序退出。")
        else:
            texts_to_embed = [doc['full_cleaned_text'] for doc in valid_documents]
            
            embeddings = generate_embeddings(texts_to_embed, MODEL_NAME)
            
            build_and_save_faiss_index(embeddings, FAISS_INDEX_PATH)
            
            save_metadata(valid_documents, METADATA_PATH)
            
            print("\\n--- 索引构建流程全部完成 ---")
            
    except Exception as e:
        print(f"在构建索引过程中发生错误: {e}")
`
    },
    {
        name: 'qa_system_core.py',
        description: 'Core logic for the Q&A system, refactored to use FastEmbedEmbeddings.',
        content: `# qa_system_core_refactored.py