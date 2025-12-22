import os
import sys
import subprocess
import time

def check_data_exists():
    """检查数据集是否存在"""
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'

    if not os.path.exists(train_dir):
        print(f"❌ 训练数据目录不存在: {train_dir}")
        return False

    if not os.path.exists(val_dir):
        print(f"❌ 验证数据目录不存在: {val_dir}")
        return False

    # 检查类别目录
    train_classes = [d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))]
    val_classes = [d for d in os.listdir(val_dir)
                  if os.path.isdir(os.path.join(val_dir, d))]

    if len(train_classes) == 0:
        print(f"❌ 训练数据目录中没有找到类别子目录")
        return False

    if len(val_classes) == 0:
        print(f"❌ 验证数据目录中没有找到类别子目录")
        return False

    print(f"✅ 数据集检查通过")
    print(f"   训练集类别: {train_classes}")
    print(f"   验证集类别: {val_classes}")
    return True


def check_dependencies():
    """检查依赖包"""
    required_packages = ['torch', 'torchvision', 'PIL']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")

    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def run_training(config_name, args):
    """运行训练"""
    cmd = [sys.executable, 'simple_train.py'] + args
    print(f"\n🚀 开始训练: {config_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print(f"✅ 训练完成: {config_name}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"❌ 训练失败: {config_name}")
        print(f"错误代码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("-" * 60)
        print(f"⏹️ 训练被中断: {config_name}")
        return False


def main():
    print("=" * 60)
    print("🎯 RepGhost 快速训练启动器")
    print("💻 针对 MacBook Air M3 芯片优化")
    print("=" * 60)

    # 环境检查
    print("\n📋 环境检查...")

    print("\n1. 检查数据集...")
    if not check_data_exists():
        print("\n❌ 请先准备好数据集再运行训练")
        return

    print("\n2. 检查依赖包...")
    if not check_dependencies():
        print("\n❌ 请先安装必要的依赖包")
        return

    # 训练配置选择
    print("\n🎯 选择训练配置:")
    print("1. 快速测试 (5 epochs, batch_size=8)")
    print("2. 标准训练 (50 epochs, batch_size=16)")
    print("3. 完整训练 (100 epochs, batch_size=16)")
    print("4. 自定义训练")
    print("5. 仅推理测试")

    try:
        choice = input("\n请选择 (1-5): ").strip()

        if choice == '1':
            # 快速测试
            args = [
                '--epochs', '5',
                '--batch_size', '8',
                '--img_size', '128',  # 更小的图像尺寸
                '--output_dir', 'outputs/quick_test'
            ]
            config_name = "快速测试"
            run_training(config_name, args)

        elif choice == '2':
            # 标准训练
            args = [
                '--epochs', '50',
                '--batch_size', '16',
                '--output_dir', 'outputs/standard_train'
            ]
            config_name = "标准训练"
            run_training(config_name, args)

        elif choice == '3':
            # 完整训练
            args = [
                '--epochs', '100',
                '--batch_size', '16',
                '--output_dir', 'outputs/full_train'
            ]
            config_name = "完整训练"
            run_training(config_name, args)

        elif choice == '4':
            # 自定义训练
            print("\n自定义训练参数:")
            epochs = input("训练轮数 (默认100): ").strip() or "100"
            batch_size = input("批次大小 (默认16): ").strip() or "16"
            img_size = input("图像尺寸 (默认224): ").strip() or "224"
            model_size = input("模型大小 0_5x/0_8x/1_0x (默认0_5x): ").strip() or "0_5x"

            args = [
                '--epochs', epochs,
                '--batch_size', batch_size,
                '--img_size', img_size,
                '--model_size', model_size,
                '--output_dir', 'outputs/custom_train'
            ]
            config_name = f"自定义训练 ({model_size}, {epochs} epochs)"
            run_training(config_name, args)

        elif choice == '5':
            # 推理测试
            print("\n🔍 推理测试")

            # 检查可用的模型
            available_models = []
            model_dirs = ['outputs/train', 'outputs/quick_test', 'outputs/standard_train', 'outputs/full_train']

            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    model_path = os.path.join(model_dir, 'best_model.pth')
                    if os.path.exists(model_path):
                        available_models.append(model_path)

            if not available_models:
                print("❌ 没有找到训练好的模型，请先进行训练")
                return

            print("找到以下训练好的模型:")
            for i, model_path in enumerate(available_models, 1):
                print(f"  {i}. {model_path}")

            # 使用第一个找到的模型
            checkpoint_path = available_models[0]
            print(f"\n使用模型: {checkpoint_path}")

            image_path = input("输入图片路径 (推荐: inference/b1cd1e94-26dd524f.jpg): ").strip()

            if not image_path:
                # 使用默认图片
                default_image = "inference/b1cd1e94-26dd524f.jpg"
                if os.path.exists(default_image):
                    image_path = default_image
                    print(f"使用默认图片: {image_path}")
                else:
                    print("没有提供图片路径且默认图片不存在，跳过推理测试")
                    return

            if not os.path.exists(image_path):
                print(f"❌ 图片文件不存在: {image_path}")
                return

            args = [
                '--checkpoint', checkpoint_path,
                '--image', image_path
            ]

            cmd = [sys.executable, 'inference.py'] + args
            print(f"\n运行推理: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, check=True, capture_output=False)
                print("\n✅ 推理完成")
            except subprocess.CalledProcessError as e:
                print(f"❌ 推理失败，错误代码: {e.returncode}")
            except FileNotFoundError:
                print("❌ 找不到推理脚本 inference.py")

        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户取消操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")


if __name__ == '__main__':
    main()