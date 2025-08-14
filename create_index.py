# create_index.py
# Last Reviewed: 2025-08-07
# This script has been reviewed and validated for Python 3.13 compatibility
# as per the "Python 3.13 Compatibility Upgrade Plan". Key dependencies
# (faiss-cpu, fastembed, numpy) are aligned with the new requirements.

import json
import os
import faiss
import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm
from typing import List, Dict, Any

# --- Constants ---
# These paths are aligned with the overall project structure.
INPUT_JSON_PATH: str = 'data/processed_policies.json'
FAISS_INDEX_DIR: str = 'faiss_index'
FAISS_INDEX_PATH: str = os.path.join(FAISS_INDEX_DIR, 'faiss_index.bin')
METADATA_PATH: str = os.path.join(FAISS_INDEX_DIR, 'documents_metadata.json')

# Model selection is consistent with the core QA system (qa_system_core.py).
MODEL_NAME: str = 'BAAI/bge-small-zh-v1.5'

def load_and_filter_data(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载和过滤文档。

    此函数仅保留具有非空'full_cleaned_text'的文档，
    以确保只索引有效内容。同时截断过长的文档，
    以防止问答系统中出现token长度问题。

    Args:
        file_path: 输入JSON文件的路径。

    Returns:
        有效文档列表。
    """
    print(f"正在从 {file_path} 加载数据...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 未找到输入文件 '{file_path}'。请确保数据放置正确。")

    with open(file_path, 'r', encoding='utf-8') as f:
        all_documents = json.load(f)

    # 过滤掉任何缺乏可处理文本内容的文档。
    processed_docs = []
    max_text_length = 800  # 进一步减少文档文本长度，避免token过长
    
    for doc in all_documents:
        text_content = doc.get('full_cleaned_text', '').strip()
        if text_content:
            # 如果文档过长，截断到合适长度
            if len(text_content) > max_text_length:
                text_content = text_content[:max_text_length] + "..."
                doc['full_cleaned_text'] = text_content
                print(f"文档已截断至 {max_text_length} 字符")
            
            processed_docs.append(doc)

    print(f"加载完成。找到 {len(all_documents)} 条总记录，其中 {len(processed_docs)} 条有效可索引。")
    return processed_docs

def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    使用fastembed模型为文本列表生成向量嵌入。

    此函数已在fastembed>=0.7.1和numpy>=2.0上验证。
    手动L2归一化对于BGE等模型至关重要，
    在最新的NumPy版本中功能保持正确。

    Args:
        texts: 要嵌入的文本字符串列表。
        model_name: 要使用的fastembed嵌入模型名称。

    Returns:
        具有float32数据类型的L2归一化嵌入的NumPy数组。
    """
    print(f"正在初始化轻量级嵌入模型: {model_name}...")
    # TextEmbedding接口稳定且与升级后的库版本兼容。
    # fastembed通过ONNX Runtime自动处理设备选择(CPU/GPU)。
    embedding_model = TextEmbedding(model_name=model_name, cache_dir=os.getenv("HF_HOME"))

    print("正在生成文本嵌入...")
    # .embed()方法返回生成器，内存效率高。
    embeddings_list = embedding_model.embed(texts, show_progress_bar=True)

    # 将生成器转换为密集NumPy数组。此调用与NumPy 2.0兼容。
    embeddings = np.array(list(embeddings_list), dtype='float32')

    # --- 关键步骤: 手动L2归一化 ---
    # 此步骤对于BGE模型确保准确的余弦相似性搜索至关重要。
    # np.linalg.norm函数行为在NumPy 2.0中对此用例保持一致。
    print("正在归一化嵌入(L2范数)...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 添加小的epsilon以防止潜在零向量的除零错误。
    norms[norms == 0] = 1e-12
    normalized_embeddings = embeddings / norms

    print("嵌入生成和归一化完成。")
    # FAISS要求其索引使用float32类型。
    return normalized_embeddings.astype('float32')

def build_and_save_faiss_index(embeddings: np.ndarray, index_path: str) -> None:
    """
    从嵌入构建FAISS索引并保存到磁盘。

    FAISS API调用(IndexFlatL2, add, write_index)已确认
    在faiss-cpu>=1.11.0.post1上稳定且兼容。

    Args:
        embeddings: 文档嵌入数组。
        index_path: 保存FAISS索引的文件路径。
    """
    if embeddings.shape[0] == 0:
        print("警告: 未提供嵌入。跳过FAISS索引创建。")
        return

    dimension = embeddings.shape[1]
    print(f"正在构建FAISS索引(IndexFlatL2)，维度: {dimension}...")
    # IndexFlatL2提供精确的暴力搜索。这是一个稳健的基线。
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"索引构建成功，包含 {index.ntotal} 个向量。")

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS索引已保存到: {index_path}")

def save_metadata(documents: List[Dict[str, Any]], metadata_path: str) -> None:
    """
    将文档元数据保存到JSON文件。

    元数据文件将FAISS索引中的向量链接回其
    原始源内容以供检索。确保文档内容不会过长，
    以防止问答系统中出现token长度问题。

    Args:
        documents: 已处理文档列表。
        metadata_path: 保存元数据JSON的文件路径。
    """
    metadata_list = []
    max_text_length = 600  # 进一步减少保存的文本长度
    
    for doc in documents:
        text_content = doc.get('full_cleaned_text', '')
        # 确保保存的文本不会过长
        if len(text_content) > max_text_length:
            text_content = text_content[:max_text_length] + "..."
        
        metadata_list.append({
            'source_url': doc.get('source_url'),
            'title': doc.get('metadata', {}).get('title'),
            'publish_date': doc.get('metadata', {}).get('publish_date'),
            'publish_dept': doc.get('metadata', {}).get('publish_dept'),
            'document_id': doc.get('metadata', {}).get('document_id'),
            'full_cleaned_text': text_content  # 使用截断后的文本
        })

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

    print(f"文档元数据已保存到: {metadata_path}")

if __name__ == "__main__":
    print("--- 开始文档索引和嵌入生成过程 ---")
    try:
        # 步骤1: 加载和验证源文档
        valid_documents = load_and_filter_data(INPUT_JSON_PATH)

        if not valid_documents:
            print("未找到有效文档。终止过程。")
        else:
            # 步骤2: 提取用于嵌入的文本内容
            texts_to_embed = [doc['full_cleaned_text'] for doc in valid_documents]

            # 步骤3: 生成归一化向量嵌入
            embeddings = generate_embeddings(texts_to_embed, MODEL_NAME)

            # 步骤4: 构建并保存FAISS向量索引
            build_and_save_faiss_index(embeddings, FAISS_INDEX_PATH)

            # 步骤5: 保存相应的元数据
            save_metadata(valid_documents, METADATA_PATH)

            print("\n--- 索引管道成功完成。 ---")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保数据文件存在且路径正确。")
    except Exception as e:
        print(f"\n索引过程中发生意外错误: {e}")
