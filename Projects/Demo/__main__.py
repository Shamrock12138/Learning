"""
充电机训练演示主入口
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Demo.simple_charging_demo import main as simple_main
from Demo.enhanced_charging_demo import main as enhanced_main


def run_demo(demo_type):
    """运行指定类型的演示"""
    if demo_type == 'simple':
        print("运行简单版充电机训练演示...")
        simple_main()
    elif demo_type == 'enhanced':
        print("运行增强版充电机训练演示...")
        enhanced_main()
    else:
        print(f"未知的演示类型: {demo_type}")
        print("可用类型: simple, enhanced")


def main():
    parser = argparse.ArgumentParser(description='充电机训练演示')
    parser.add_argument(
        '--type', 
        type=str, 
        default='enhanced',
        choices=['simple', 'enhanced'],
        help='演示类型: simple(简单版) 或 enhanced(增强版，默认)'
    )
    parser.add_argument(
        '--no-demo-after-train',
        action='store_true',
        help='训练完成后不进行动画演示'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("充电机训练演示")
    print("=" * 60)
    
    # 设置全局标志，控制训练后是否进行演示
    import os
    os.environ['DEMO_AFTER_TRAIN'] = 'false' if args.no_demo_after_train else 'true'
    
    run_demo(args.type)


if __name__ == "__main__":
    main()