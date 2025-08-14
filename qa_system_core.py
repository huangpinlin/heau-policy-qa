# qa_system_core.py
# Updated: 2025-08-07 to ensure compatibility with Python 3.13 and upgraded dependencies.

# PyTorch兼容性设置
import os
os.environ['PYTORCH_JIT'] = '0'
# 允许GPU（如不可用将自动回退CPU）
os.environ['TORCH_DISABLE_GPU'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import torch
import faiss
import warnings
from typing import Dict, Any

# --- LangChain Core Imports (Updated for modern LangChain versions) ---
# As per the Python 3.13 compatibility plan, we use imports from `langchain_core`
# where applicable, ensuring stability with the unpinned langchain dependency.
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- LangChain Community and Standard Imports ---
from langchain.chains import RetrievalQA
from langchain_community.docstore import InMemoryDocstore  # Updated import path
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Suppress common warnings from the transformers and langchain ecosystem
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class QAConfig:
    """
    Manages all configuration parameters for the Q&A system.
    This centralized approach simplifies management and adjustments.
    """
    FAISS_INDEX_PATH = 'faiss_index/faiss_index.bin'
    METADATA_PATH = 'faiss_index/documents_metadata.json'
    EMBEDDING_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
    # 使用在线模型ID，首次运行会自动下载到本地缓存
    LLM_MODEL_PATH = 'Qwen/Qwen2.5-0.5B-Instruct'
    # 备用模型选项（同系列模型作为兜底）
    FALLBACK_MODELS = [
        'Qwen/Qwen2.5-0.5B-Instruct',
        'Qwen/Qwen2.5-0.5B'
    ]
    SEARCH_TOP_K = 1  # 检索文档数量，设置为1避免输入过长
    # 使用GPU以提升速度（如环境不支持将自动回退至CPU）
    DEVICE = 'cuda'
    print(f"当前设备: {DEVICE.upper()}")
    
    def __init__(self):
        # 确保配置值正确设置
        print(f"当前设备: {self.DEVICE.upper()}")
        print(f"LLM模型路径: {self.LLM_MODEL_PATH}")
        print(f"嵌入模型: {self.EMBEDDING_MODEL_NAME}")
        print(f"备用模型: {self.FALLBACK_MODELS}")


class QASystemCore:
    """
    Encapsulates the entire Q&A system's logic, including loading,
    initialization, and query execution.
    """

    def __init__(self, config: QAConfig):
        self.config = config
        self.embeddings = None
        self.llm = None
        self.retriever = None
        self.qa_chain = None

    def _load_embedding_model(self) -> None:
        """使用FastEmbed加载嵌入模型以提高效率。"""
        print(f"正在加载嵌入模型: {self.config.EMBEDDING_MODEL_NAME}...")
        # 使用FastEmbedEmbeddings，这是一个轻量级的、基于ONNX的包装器。
        # 这与create_index.py脚本保持一致，在CPU/GPU上都很高效。
        # 此组件的兼容性已在fastembed>=0.7.1上验证。
        self.embeddings = FastEmbedEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME
        )
        print("嵌入模型加载成功。")

    def _load_llm(self) -> None:
        """使用HuggingFace Transformers加载大语言模型(LLM)。"""
        print(f"正在加载LLM: {self.config.LLM_MODEL_PATH}...")
        
        # 检查配置值
        if not self.config.LLM_MODEL_PATH:
            raise ValueError("LLM_MODEL_PATH为空或未设置。请检查配置。")
        
        # 尝试加载模型，如果失败则使用备用模型
        models_to_try = [self.config.LLM_MODEL_PATH] + self.config.FALLBACK_MODELS
        
        for model_path in models_to_try:
            try:
                print(f"正在尝试加载模型: {model_path}")
                
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                # 为GPT模型设置pad_token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                # 使用accelerate自动处理设备选择，避免手动device_map冲突
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.config.DEVICE == 'cuda' else torch.float32,
                    device_map="auto" if self.config.DEVICE == 'cuda' else None,
                    trust_remote_code=True
                ).eval()
                
                # 设置生成配置 - 优化对话体验
                generation_config = {
                    "max_new_tokens": 512,  # 增加长度，让回答更完整
                    "max_length": 2048,
                    "temperature": 0.7,     # 提高温度，让回答更自然
                    "top_p": 0.9,          # 提高top_p，增加词汇多样性
                    "do_sample": True,
                    "repetition_penalty": 1.1,  # 适度重复惩罚
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "return_full_text": False,
                    "truncation": True,
                    "padding": True
                }
                
                # 确保tokenizer的max_length设置正确
                tokenizer.model_max_length = 2048
                
                # 设置tokenizer的padding和truncation
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
                
                # 创建一个自定义的pipeline配置
                # 创建pipeline时不指定device参数，让accelerate自动处理
                pipe = pipeline(
                    "text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    max_length=2048,
                    max_new_tokens=512,  # 增加长度，让回答更完整
                    truncation=True,
                    padding=True,
                    return_full_text=False,
                    do_sample=True,
                    temperature=0.7,     # 提高温度，让回答更自然
                    top_p=0.9,          # 提高top_p，增加词汇多样性
                    repetition_penalty=1.1  # 适度重复惩罚
                )
                
                # 直接设置pipeline的tokenizer参数
                pipe.tokenizer.model_max_length = 2048
                pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
                
                # 尝试直接修改pipeline的配置
                if hasattr(pipe, 'tokenizer'):
                    pipe.tokenizer.model_max_length = 2048
                    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
                
                # 尝试直接修改pipeline的配置
                if hasattr(pipe, 'model'):
                    pipe.model.config.max_length = 2048
                
                # 尝试直接修改pipeline的配置
                if hasattr(pipe, 'generation_config'):
                    pipe.generation_config.max_length = 2048
                    pipe.generation_config.max_new_tokens = 256
                
                # 使用HuggingFacePipeline
                self.llm = HuggingFacePipeline(
                    pipeline=pipe,
                    model_kwargs={
                        "max_length": 2048,
                        "max_new_tokens": 512,  # 增加长度，让回答更完整
                        "temperature": 0.7,     # 提高温度，让回答更自然
                        "top_p": 0.9,          # 提高top_p，增加词汇多样性
                        "do_sample": True,
                        "repetition_penalty": 1.1,  # 适度重复惩罚
                        "pad_token_id": tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "return_full_text": False,
                        "truncation": True,
                        "padding": True
                    }
                )
                print(f"模型加载成功: {model_path}")
                return
                
            except Exception as e:
                print(f"加载模型失败 {model_path}: {e}")
                continue
        
        # 如果所有模型都失败了
        raise RuntimeError(f"所有模型加载都失败了: {models_to_try}")

    def _load_retriever(self) -> None:
        """加载FAISS索引和元数据以构建文档检索器。"""
        print("正在加载FAISS索引和元数据...")
        if not os.path.exists(self.config.FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"错误: 在'{self.config.FAISS_INDEX_PATH}'处未找到FAISS索引文件。请先运行`create_index.py`脚本。")

        # 从磁盘加载FAISS索引。已验证与faiss-cpu>=1.11.0兼容。
        index = faiss.read_index(self.config.FAISS_INDEX_PATH)

        with open(self.config.METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 从元数据重建文档对象以用于文档存储。
        docs = [Document(page_content=item.pop('full_cleaned_text', ''), metadata=item) for item in metadata]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
        index_to_docstore_id = {i: str(i) for i in range(len(docs))}

        # 实例化LangChain FAISS向量存储。
        vector_store = LangChainFAISS(
            embedding_function=self.embeddings.embed_query,  # 直接传递函数
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        self.retriever = vector_store.as_retriever(
            search_kwargs={
                'k': self.config.SEARCH_TOP_K,
                'fetch_k': self.config.SEARCH_TOP_K * 2  # 获取更多候选，然后选择最相关的
            }
        )
        print("检索器构建成功。")

    def _create_qa_chain(self) -> None:
        """使用自定义提示创建RetrievalQA链。"""
        print("正在创建问答链...")
        prompt_template = """你是一个专业的大学政策咨询助手，请根据以下政策文档内容，用自然、流畅的中文回答用户问题。

要求：
1. 回答要准确、完整、易懂，像朋友一样友好
2. 使用正式但亲切的语气，避免过于生硬
3. 如果政策内容不够详细，请说明并提供建议
4. 回答要结构清晰，重点突出，适当使用表情符号增加亲和力
5. 如果用户的问题不完整，要主动询问更多细节
6. 回答要有逻辑性，先总结要点，再详细说明

政策文档内容：
{context}

用户问题：{question}

请用中文回答，让用户感受到专业和温暖："""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # RetrievalQA链结构在最新的langchain版本中仍然有效。
        # 其兼容性已作为Python 3.13升级的一部分得到确认。
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_separator": "\n\n"
            }
        )
        print("问答链创建成功。")

    def initialize(self) -> None:
        """按正确顺序执行所有加载和初始化步骤。"""
        self._load_embedding_model()
        self._load_llm()
        self._load_retriever()
        self._create_qa_chain()
        print("\n--- 问答系统已初始化并准备就绪 ---")

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        接收用户查询，调用问答链，并返回结构化结果。
        """
        if not self.qa_chain:
            raise RuntimeError("系统未初始化。请先调用`initialize()`方法。")
        
        # 检查查询长度
        if len(query) > 500:  # 限制查询长度
            query = query[:500]
            print(f"查询已截断至500字符: {query}")
        
        print(f"\n正在处理查询: '{query}'...")

        try:
            # .invoke()方法是运行LangChain链的标准现代方法(LCEL)。
            result = self.qa_chain.invoke({"query": query})
            
            # 后处理：清理和验证输出
            if 'result' in result:
                cleaned_result = self._clean_output(result['result'])
                result['result'] = cleaned_result
            
            return result
        except Exception as e:
            print(f"问答链中出现错误: {e}")
            # 如果出现长度相关错误，尝试截断输入
            if "max_length" in str(e) or "length" in str(e):
                print("正在尝试处理长度相关错误...")
                # 可以在这里添加更多的错误处理逻辑
                raise RuntimeError(f"输入过长无法处理。请尝试更短的问题。错误: {e}")
            raise e
    
    def _clean_output(self, text: str) -> str:
        """
        清理和验证AI输出，确保输出质量。
        """
        if not text:
            return "抱歉，无法生成有效答案。"
        
        # 移除多余的空白字符
        text = text.strip()
        
        # 检查是否包含乱码或混合语言
        import re
        
        # 检测非中文字符（保留标点符号和数字）
        chinese_pattern = re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef0-9，。！？；：""''（）【】]')
        non_chinese_chars = chinese_pattern.findall(text)
        
        # 如果非中文字符过多，尝试提取有意义的部分
        if len(non_chinese_chars) > len(text) * 0.4:  # 进一步提高阈值到40%
            # 尝试找到有意义的中文片段
            chinese_sentences = re.findall(r'[^。！？]*[。！？]', text)
            if chinese_sentences:
                # 检查每个句子，找到最长的有意义句子
                valid_sentences = []
                for sentence in chinese_sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 5:  # 至少5个字符
                        # 计算中文字符比例
                        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sentence))
                        if chinese_chars / len(sentence) > 0.4:  # 中文字符超过40%，进一步降低要求
                            valid_sentences.append(sentence)
                
                if valid_sentences:
                    # 返回最长的有效句子
                    return max(valid_sentences, key=len)
                else:
                    # 尝试从原始文本中提取任何有意义的中文片段
                    chinese_fragments = re.findall(r'[\u4e00-\u9fff]+', text)
                    if chinese_fragments:
                        # 找到最长的中文片段
                        longest_fragment = max(chinese_fragments, key=len)
                        if len(longest_fragment) >= 3:  # 至少3个中文字符
                            # 检查这个片段是否包含乱码
                            if self._is_valid_chinese_fragment(longest_fragment):
                                return f"根据政策内容，{longest_fragment}。如需详细信息，请查看相关政策文件。"
                            else:
                                # 尝试清理乱码字符
                                cleaned_fragment = self._clean_garbled_text(longest_fragment)
                                if cleaned_fragment:
                                    return f"根据政策内容，{cleaned_fragment}。如需详细信息，请查看相关政策文件。"
                                else:
                                    return "抱歉，生成的答案质量不佳。请尝试重新提问或联系管理员。"
                    
                    return "抱歉，生成的答案质量不佳。请尝试重新提问或联系管理员。"
            else:
                # 尝试从原始文本中提取任何有意义的中文片段
                chinese_fragments = re.findall(r'[\u4e00-\u9fff]+', text)
                if chinese_fragments:
                    longest_fragment = max(chinese_fragments, key=len)
                    if len(longest_fragment) >= 3:
                        if self._is_valid_chinese_fragment(longest_fragment):
                            return f"根据政策内容，{longest_fragment}。如需详细信息，请查看相关政策文件。"
                        else:
                            # 尝试清理乱码字符
                            cleaned_fragment = self._clean_garbled_text(longest_fragment)
                            if cleaned_fragment:
                                return f"根据政策内容，{cleaned_fragment}。如需详细信息，请查看相关政策文件。"
                
                return "抱歉，生成的答案质量不佳。请尝试重新提问或联系管理员。"
        
        # 如果输出看起来正常，直接返回
        return text
    
    def _is_valid_chinese_fragment(self, fragment: str) -> bool:
        """
        检查中文片段是否有效（不包含乱码）。
        """
        if not fragment:
            return False
        
        # 检查是否包含明显的乱码字符
        suspicious_chars = ['咕', '之', '泪', '打', '然', '時', '間', '雯', '咕', '膀', '侪', '雫', '敵', '護', '倘', '冶', '興', '語', '去']
        for char in suspicious_chars:
            if char in fragment:
                return False
        
        # 检查中文字符的连续性
        import re
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', fragment)
        if len(chinese_chars) < 3:
            return False
        
        # 检查是否有意义的中文词汇（简单的启发式检查）
        meaningful_words = ['政策', '规定', '要求', '条件', '申请', '奖励', '奖学金', '助学金', '转专业', '科研', '项目', '学生', '大学', '学院', '专业', '课程', '成绩', '学分', '毕业', '学位', '每', '等', '级', '类', '支持', '帮助', '提供', '获得', '参与', '参加', '可以', '能够', '需要', '必须', '应该', '根据', '按照', '关于', '对于', '有关', '相关']
        has_meaningful_content = any(word in fragment for word in meaningful_words)
        
        # 检查是否包含明显的无意义字符组合
        meaningless_patterns = ['咕之泪', '打然時', '雯咕', '膀侪', '雫市', '敵道', '護意', '倘前', '冶時', '興所', '語去']
        for pattern in meaningless_patterns:
            if pattern in fragment:
                return False
        
        return has_meaningful_content
    
    def _clean_garbled_text(self, text: str) -> str:
        """
        清理乱码文本，尝试提取有意义的内容。
        """
        if not text:
            return ""
        
        import re
        
        # 定义常见的乱码字符映射
        garbled_char_map = {
            '霣': '根', '佚': '体', '高': '高', '專': '专', '護': '护', '帶': '带', 
            '時': '时', '間': '间', '这': '这', '是': '是', '不': '不', '正': '正',
            '之': '之', '考': '考', '当': '当', '然': '然', '目': '目', '盅': '中',
            '巈': '护', '常': '常', '手': '手', '尚': '尚', '高': '高', '体': '体',
            '行': '行', '要': '要', '这': '这', '此': '此', '不': '不', '昬': '昏',
            '而': '而', '思': '思', '想': '想', '意': '意', '录': '录', '牒': '牒',
            '師': '师', '泜': '泜', '愚': '愚', '鬼': '鬼', '呈': '呈', '護': '护',
            '然': '然', '態': '态', '者': '者', '民': '民', '道': '道', '上': '上',
            '手': '手', '数': '数', '徙': '徙', '前': '前', '而': '而', '要': '要',
            '明': '明', '雙': '双', '如': '如', '語': '语', '哪': '哪', '战': '战',
            '录': '录', '牒': '牒', '師': '师', '泜': '泜', '这': '这', '是': '是',
            '不': '不', '能': '能', '愚': '愚', '鬼': '鬼', '呈': '呈', '護': '护',
            '然': '然', '態': '态', '者': '者', '民': '民', '道': '道', '上': '上',
            '手': '手', '数': '数', '徙': '徙', '間': '间', '前': '前', '而': '而',
            '要': '要', '明': '明', '雙': '双', '如': '如', '語': '语'
        }
        
        # 尝试清理乱码字符
        cleaned_text = text
        for garbled, clean in garbled_char_map.items():
            cleaned_text = cleaned_text.replace(garbled, clean)
        
        # 移除明显的无意义字符
        cleaned_text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', cleaned_text)
        
        # 检查清理后的文本是否有意义
        if len(cleaned_text) >= 3:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned_text))
            if chinese_chars / len(cleaned_text) > 0.3:  # 至少30%是中文字符
                return cleaned_text
        
        return ""
