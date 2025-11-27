"""
启动训练和TensorBoard监控的便捷脚本
"""
import subprocess
import sys
import os
import time
import webbrowser

def main():
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print("=" * 60)
    print("启动训练监控系统")
    print("=" * 60)
    
    # 1. 启动 TensorBoard
    tensorboard_dir = os.path.join(project_root, 'runs')
    print(f"\n[1/3] 启动 TensorBoard...")
    print(f"日志目录: {tensorboard_dir}")
    
    try:
        tb_process = subprocess.Popen(
            ['tensorboard', '--logdir', tensorboard_dir, '--port', '6006', '--host', '0.0.0.0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )
        print("✓ TensorBoard 启动成功")
        print("  访问地址: http://localhost:6006")
        time.sleep(3)
        
        # 自动打开浏览器
        try:
            webbrowser.open('http://localhost:6006')
            print("✓ 浏览器已自动打开")
        except:
            print("⚠ 无法自动打开浏览器，请手动访问: http://localhost:6006")
    except Exception as e:
        print(f"✗ TensorBoard 启动失败: {e}")
        return
    
    train_script = os.path.join(script_dir, 'train_hierarchical.py')
    
    try:
        print(f"\n{'='*60}")
        print("训练进程启动中...")
        print(f"{'='*60}\n")
        
        env = os.environ.copy()
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        train_process = subprocess.Popen(
            [sys.executable, train_script],
            cwd=project_root,
            env=env
        )
        
        print("✓ 训练进程已启动")
        print("\n监控提示:")
        print("  - TensorBoard: http://localhost:6006")
        print("  - 按 Ctrl+C 可以停止训练")
        print("\n" + "="*60 + "\n")
        # 等待训练完成，方便调试/查看日志
        train_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n中断训练...")
        train_process.terminate()
        print("✓ 训练已停止")
    except Exception as e:
        print(f"✗ 训练启动失败: {e}")
    finally:
        # 清理
        print("\n清理进程...")
        try:
            tb_process.terminate()
            print("✓ TensorBoard 已关闭")
        except:
            pass

if __name__ == '__main__':
    main()
