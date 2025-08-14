# web_interface.py
# 现代化Web界面 - 提供更好的用户体验

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import time
from datetime import datetime
import os
from typing import Dict, Any, List

from qa_system_core import QASystemCore, QAConfig
from links_config import get_links_for_title

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
qa_system = None
conversations = {}
user_sessions = {}

class WebChatInterface:
    """Web聊天界面类"""
    
    def __init__(self):
        self.config = QAConfig()
        self.qa_system = QASystemCore(self.config)
        self.initialize_system()
    
    def initialize_system(self):
        """初始化系统"""
        try:
            print("正在初始化Web聊天系统...")
            self.qa_system.initialize()
            print("Web聊天系统初始化完成！")
        except Exception as e:
            print(f"系统初始化失败: {e}")
            raise e
    
    def get_response(self, user_input: str, conversation_id: str) -> Dict[str, Any]:
        """获取AI回复"""
        try:
            result = self.qa_system.answer_query(user_input)
            
            # 处理source_documents，确保可以JSON序列化
            sources = []
            if result.get('source_documents'):
                for doc in result.get('source_documents', []):
                    if hasattr(doc, 'metadata'):
                        # 获取原始网页链接
                        original_url = doc.metadata.get('source_url', '')
                        
                        # 如果没有原始链接，尝试从文档内容中查找
                        if not original_url:
                            # 根据文档标题查找对应的原始文档
                            title = doc.metadata.get('title', '')
                            try:
                                import json
                                import os
                                # 使用正确的相对路径
                                if '奖学金' in title:
                                    file_path = os.path.join('..', 'data', 'raw', '奖学金管理办法.json')
                                    if os.path.exists(file_path):
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            raw_doc = json.load(f)
                                            original_url = raw_doc.get('source_url', '')
                                elif '资助' in title:
                                    file_path = os.path.join('..', 'data', 'raw', '学生资助工作实施细则.json')
                                    if os.path.exists(file_path):
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            raw_doc = json.load(f)
                                            original_url = raw_doc.get('source_url', '')
                                elif '转专业' in title:
                                    file_path = os.path.join('..', 'data', 'raw', '转专业管理办法.json')
                                    if os.path.exists(file_path):
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            raw_doc = json.load(f)
                                            original_url = raw_doc.get('source_url', '')
                            except Exception as e:
                                print(f"读取原始文档失败: {e}")
                                original_url = ''
                        
                        # 使用链接配置文件获取官网链接
                        title = doc.metadata.get('title', '')
                        links = get_links_for_title(title)
                        
                        sources.append({
                            'title': doc.metadata.get('title', '未知文档'),
                            'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                            'links': links,
                            'full_content': doc.page_content
                        })
                    else:
                        sources.append({
                            'title': '未知文档',
                            'content': str(doc)[:200] + '...' if len(str(doc)) > 200 else str(doc),
                            'original_url': '',
                            'full_content': str(doc)
                        })
            
            return {
                'status': 'success',
                'response': result.get('result', '抱歉，我无法生成有效回答。'),
                'sources': sources,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'response': f'抱歉，处理您的问题时出现了错误: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# 初始化Web聊天界面
try:
    web_chat = WebChatInterface()
except Exception as e:
    print(f"Web聊天界面初始化失败: {e}")
    web_chat = None

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天API"""
    if not web_chat:
        return jsonify({'error': '系统未初始化'}), 500
    
    data = request.get_json()
    user_input = data.get('message', '').strip()
    conversation_id = data.get('conversation_id', 'default')
    
    if not user_input:
        return jsonify({'error': '消息不能为空'}), 400
    
    # 获取AI回复
    result = web_chat.get_response(user_input, conversation_id)
    
    # 保存到对话历史
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    conversations[conversation_id].append({
        'role': 'assistant',
        'content': result['response'],
        'timestamp': result['timestamp'],
        'sources': result.get('sources', [])
    })
    
    return jsonify(result)

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """获取对话历史"""
    conversation_id = request.args.get('id', 'default')
    if conversation_id in conversations:
        return jsonify(conversations[conversation_id])
    return jsonify([])

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """创建新对话"""
    conversation_id = f"conv_{int(time.time())}"
    conversations[conversation_id] = []
    return jsonify({'conversation_id': conversation_id})

@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_ready': web_chat is not None
    })

@app.route('/api/document/<path:doc_path>')
def get_document(doc_path):
    """获取文档内容"""
    try:
        # 安全路径检查，防止目录遍历攻击
        if '..' in doc_path or doc_path.startswith('/'):
            return jsonify({'error': '无效的文档路径'}), 400
        
        # 构建完整文件路径
        full_path = os.path.join(os.getcwd(), doc_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            return jsonify({'error': '文档不存在'}), 404
        
        # 读取文档内容
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'title': os.path.basename(doc_path),
            'content': content,
            'path': doc_path
        })
        
    except Exception as e:
        return jsonify({'error': f'读取文档失败: {str(e)}'}), 500

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print(f'客户端已连接: {request.sid}')
    emit('connected', {'message': '连接成功'})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    print(f'客户端已断开: {request.sid}')

@socketio.on('join_room')
def handle_join_room(data):
    """加入房间"""
    room = data.get('room')
    join_room(room)
    emit('room_joined', {'room': room})

@socketio.on('leave_room')
def handle_leave_room(data):
    """离开房间"""
    room = data.get('room')
    leave_room(room)
    emit('room_left', {'room': room})

@socketio.on('chat_message')
def handle_chat_message(data):
    """处理聊天消息"""
    if not web_chat:
        emit('error', {'message': '系统未初始化'})
        return
    
    user_input = data.get('message', '').strip()
    conversation_id = data.get('conversation_id', 'default')
    
    if not user_input:
        emit('error', {'message': '消息不能为空'})
        return
    
    # 获取AI回复
    result = web_chat.get_response(user_input, conversation_id)
    
    # 保存到对话历史
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    conversations[conversation_id].append({
        'role': 'assistant',
        'content': result['response'],
        'timestamp': result['timestamp'],
        'sources': result.get('sources', [])
    })
    
    # 广播消息到房间
    emit('chat_response', result, room=conversation_id)

if __name__ == '__main__':
    print("启动Web聊天界面...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
