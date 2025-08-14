# start_chat.py
# 智能政策咨询系统启动脚本

import sys
import os
import webbrowser

def main():
    """主启动函数"""
    print("="*60)
    print("🤖 智能政策咨询系统")
    print("="*60)
    print("\n请选择启动方式:")
    print("1. 命令行聊天界面 (推荐)")
    print("2. Web浏览器界面")
    print("3. 退出")
    print("="*60)
    
    while True:
        try:
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                print("\n正在启动命令行聊天界面...")
                try:
                    from chat_interface import main as chat_main
                    chat_main()
                except ImportError as e:
                    print(f"导入失败: {e}")
                    print("请确保已安装所有依赖包")
                except Exception as e:
                    print(f"启动失败: {e}")
                break
                
            elif choice == '2':
                print("\n正在启动Web界面...")
                url = "http://localhost:5000"
                print(f"启动后将自动打开浏览器访问: {url}")
                try:
                    from web_interface import app, socketio
                    # 延迟1秒后打开浏览器
                    def open_browser():
                        webbrowser.open(url)
                    socketio.sleep(1)
                    socketio.start_background_task(open_browser)
                    print("Web界面启动成功！")
                    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
                except ImportError as e:
                    print(f"导入失败: {e}")
                    print("请确保已安装Flask相关依赖包")
                except Exception as e:
                    print(f"启动失败: {e}")
                break
                
            elif choice == '3':
                print("\n感谢使用！再见！")
                sys.exit(0)
                
            else:
                print("无效选择，请输入1、2或3")
                
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，退出...")
            sys.exit(0)
        except Exception as e:
            print(f"发生错误: {e}")
            continue

if __name__ == "__main__":
    main()

