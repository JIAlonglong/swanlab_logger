#!/usr/bin/env python3
"""Test for SwanLab Logger"""

import os
import shutil
from swanlab_logger import create_logger


if __name__ == "__main__":
    # 清理之前的测试日志目录
    log_dir = "./test_logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=== 测试SwanLab Logger ===")
    print(f"日志将保存到: {os.path.abspath(log_dir)}")
    
    # 创建日志记录器
    logger = create_logger(
        log_dir=log_dir,
        experiment_name="Test_Logger",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_name": "TestModel"
        },
        use_tensorboard=True,
        use_swanlab=True
    )
    
    print("\n=== 测试单个标量记录 ===")
    # 测试单个标量记录
    for step in range(1, 6):
        loss = 1.0 / step
        accuracy = 0.5 + step * 0.1
        logger.log_scalar("loss", loss, step=step)
        logger.log_scalar("accuracy", accuracy, step=step)
        print(f"Step {step}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    print("\n=== 测试兼容TensorBoard的add_scalar方法 ===")
    # 测试兼容TensorBoard的add_scalar方法
    for step in range(6, 11):
        reward = step * 2.5
        logger.add_scalar("reward", reward, step=step)
        print(f"Step {step}: reward={reward:.2f}")
    
    print("\n=== 测试批量记录多个指标 ===")
    # 测试批量记录多个指标
    for step in range(11, 16):
        metrics = {
            "train_loss": 0.5 / step,
            "val_loss": 0.8 / step,
            "train_acc": 0.7 + step * 0.02,
            "val_acc": 0.6 + step * 0.015
        }
        logger.log_dict(metrics, step=step)
        print(f"Step {step}: {metrics}")
    
    print("\n=== 测试超参数记录 ===")
    # 测试超参数记录
    hparams = {
        "optimizer": "Adam",
        "epochs": 100,
        "dropout_rate": 0.3
    }
    metrics = {
        "best_train_loss": 0.05,
        "best_val_acc": 0.92
    }
    logger.log_hparams(hparams, metrics)
    print(f"超参数: {hparams}")
    print(f"对应的指标: {metrics}")
    
    # 关闭日志记录器
    print("\n=== 关闭日志记录器 ===")
    logger.close()
    print("测试完成！请查看日志目录验证记录结果。")
    print("\n注意：")
    print("1. TensorBoard日志可通过 'tensorboard --logdir=./test_logs' 查看")
    print("2. SwanLab日志可在其Web界面查看（如果已安装并配置）")