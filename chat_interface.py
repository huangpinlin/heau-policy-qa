# chat_interface.py
# 智能对话界面 - 让系统像主流AI对话一样智能

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from qa_system_core import QASystemCore, QAConfig


@dataclass
class Message:
    """对话消息数据结构"""
    role: str  # 'user' 或 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Conversation:
    """对话会话数据结构"""
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None


class ChatInterface:
    """智能对话界面类"""
    
    def __init__(self, config: QAConfig):
        self.config = config
        self.qa_system = QASystemCore(config)
        self.current_conversation: Optional[Conversation] = None
        self.conversation_history: List[Conversation] = []
        self.max_history_length = 10  # 最大历史对话数
        
        # 初始化QA系统
        print("正在初始化智能对话系统...")
        self.qa_system.initialize()
        print("智能对话系统初始化完成！")
        
        # 加载对话历史
        self._load_conversation_history()
    
    def _load_conversation_history(self):
        """加载对话历史"""
        history_file = "conversation_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = [
                        Conversation(**conv_data) for conv_data in data
                    ]
                print(f"已加载 {len(self.conversation_history)} 个历史对话")
            except Exception as e:
                print(f"加载对话历史失败: {e}")
    
    def _save_conversation_history(self):
        """保存对话历史"""
        history_file = "conversation_history.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(conv) for conv in self.conversation_history], f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话历史失败: {e}")
    
    def start_new_conversation(self, title: str = None) -> str:
        """开始新对话"""
        if not title:
            title = f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conversation_id = f"conv_{int(time.time())}"
        self.current_conversation = Conversation(
            id=conversation_id,
            title=title,
            messages=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # 添加欢迎消息
        welcome_msg = Message(
            role="assistant",
            content="您好！我是您的大学政策咨询助手。我可以帮您了解各种政策规定，包括奖学金、助学金、转专业等相关政策。请告诉我您想了解什么？",
            timestamp=datetime.now().isoformat()
        )
        self.current_conversation.messages.append(welcome_msg)
        
        print(f"\n=== 新对话已开始: {title} ===")
        print(welcome_msg.content)
        
        return conversation_id
    
    def send_message(self, user_input: str) -> str:
        """发送用户消息并获取回复"""
        if not self.current_conversation:
            self.start_new_conversation()
        
        # 添加用户消息
        user_msg = Message(
            role="user",
            content=user_input,
            timestamp=datetime.now().isoformat()
        )
        self.current_conversation.messages.append(user_msg)
        
        # 更新对话时间
        self.current_conversation.updated_at = datetime.now().isoformat()
        
        print(f"\n用户: {user_input}")
        
        # 获取AI回复
        try:
            result = self.qa_system.answer_query(user_input)
            ai_response = result.get('result', '抱歉，我无法生成有效回答。')
            
            # 添加AI回复
            ai_msg = Message(
                role="assistant",
                content=ai_response,
                timestamp=datetime.now().isoformat(),
                metadata={'source_documents': result.get('source_documents', [])}
            )
            self.current_conversation.messages.append(ai_msg)
            
            print(f"\n助手: {ai_response}")
            
            return ai_response
            
        except Exception as e:
            error_msg = f"抱歉，处理您的问题时出现了错误: {str(e)}"
            print(f"\n助手: {error_msg}")
            
            # 添加错误消息
            error_ai_msg = Message(
                role="assistant",
                content=error_msg,
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e)}
            )
            self.current_conversation.messages.append(error_ai_msg)
            
            return error_msg
    
    def end_conversation(self):
        """结束当前对话"""
        if self.current_conversation:
            # 保存到历史
            self.conversation_history.append(self.current_conversation)
            
            # 限制历史长度
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # 保存历史
            self._save_conversation_history()
            
            print(f"\n=== 对话已结束: {self.current_conversation.title} ===")
            print(f"本次对话共 {len(self.current_conversation.messages)} 条消息")
            
            self.current_conversation = None
    
    def get_conversation_summary(self) -> str:
        """获取当前对话摘要"""
        if not self.current_conversation:
            return "当前没有活跃对话"
        
        user_messages = [msg for msg in self.current_conversation.messages if msg.role == 'user']
        ai_messages = [msg for msg in self.current_conversation.messages if msg.role == 'assistant']
        
        summary = f"""
对话摘要: {self.current_conversation.title}
开始时间: {self.current_conversation.created_at}
消息总数: {len(self.current_conversation.messages)}
用户问题: {len(user_messages)} 个
AI回复: {len(ai_messages)} 个
        """.strip()
        
        return summary
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """列出所有对话"""
        conversations = []
        for conv in self.conversation_history:
            conversations.append({
                'id': conv.id,
                'title': conv.title,
                'message_count': len(conv.messages),
                'created_at': conv.created_at,
                'updated_at': conv.updated_at
            })
        return conversations
    
    def load_conversation(self, conversation_id: str) -> bool:
        """加载指定对话"""
        for conv in self.conversation_history:
            if conv.id == conversation_id:
                self.current_conversation = conv
                print(f"\n=== 已加载对话: {conv.title} ===")
                return True
        return False
    
    def interactive_chat(self):
        """交互式聊天模式"""
        print("\n" + "="*60)
        print("欢迎使用智能政策咨询系统！")
        print("="*60)
        print("\n可用命令:")
        print("- /new: 开始新对话")
        print("- /end: 结束当前对话")
        print("- /list: 查看对话历史")
        print("- /summary: 查看当前对话摘要")
        print("- /help: 显示帮助信息")
        print("- /quit: 退出系统")
        print("- 直接输入问题开始咨询")
        print("="*60)
        
        self.start_new_conversation()
        
        while True:
            try:
                user_input = input("\n您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        if self.current_conversation:
                            self.end_conversation()
                        print("\n感谢使用！再见！")
                        break
                    elif user_input == '/new':
                        self.end_conversation()
                        self.start_new_conversation()
                        continue
                    elif user_input == '/end':
                        self.end_conversation()
                        continue
                    elif user_input == '/list':
                        conversations = self.list_conversations()
                        if conversations:
                            print("\n对话历史:")
                            for conv in conversations:
                                print(f"- {conv['title']} ({conv['message_count']} 条消息)")
                        else:
                            print("\n暂无对话历史")
                        continue
                    elif user_input == '/summary':
                        print(self.get_conversation_summary())
                        continue
                    elif user_input == '/help':
                        print("\n可用命令:")
                        print("- /new: 开始新对话")
                        print("- /end: 结束当前对话")
                        print("- /list: 查看对话历史")
                        print("- /summary: 查看当前对话摘要")
                        print("- /help: 显示帮助信息")
                        print("- /quit: 退出系统")
                        continue
                    else:
                        print(f"未知命令: {user_input}")
                        continue
                
                # 发送消息
                self.send_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n检测到中断信号...")
                if self.current_conversation:
                    self.end_conversation()
                print("再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
                continue


def main():
    """主函数"""
    try:
        config = QAConfig()
        chat_interface = ChatInterface(config)
        chat_interface.interactive_chat()
    except Exception as e:
        print(f"系统启动失败: {e}")
        print("请检查配置和依赖是否正确安装")


if __name__ == "__main__":
    main()
