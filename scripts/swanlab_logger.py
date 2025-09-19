# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SwanLab Logger Adapter Module

This module provides a unified logging interface that supports both TensorBoard and SwanLab logging
"""

import os
from typing import Dict, Any, Optional, Union

# 尝试导入SwanLab
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    print("Warning: SwanLab is not installed. Please run 'pip install swanlab' to enable SwanLab logging.")

# 导入TensorBoard
from torch.utils.tensorboard import SummaryWriter


class SwanLabLogger:
    """统一的日志记录器，支持TensorBoard和SwanLab"""
    
    def __init__(self, log_dir: Optional[str] = None, 
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 use_tensorboard: bool = True,
                 use_swanlab: bool = True):
        """初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
            config: 实验配置参数
            use_tensorboard: 是否使用TensorBoard
            use_swanlab: 是否使用SwanLab
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or (os.path.basename(log_dir) if log_dir else "DreamWaQ_Training")
        self.config = config or {}
        
        # 初始化TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard and log_dir is not None:
            try:
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
                print(f"TensorBoard logger initialized, logs will be saved to {log_dir}")
            except Exception as e:
                print(f"Failed to initialize TensorBoard: {e}")
        
        # 初始化SwanLab
        self.swanlab_run = None
        if use_swanlab and HAS_SWANLAB and log_dir is not None:
            try:
                self.swanlab_run = swanlab.init(
                    project="DreamWaQ",
                    experiment_name=self.experiment_name,
                    config=self.config,
                    logdir=log_dir
                )
                print(f"SwanLab logger initialized successfully for experiment: {self.experiment_name}")
            except Exception as e:
                print(f"Failed to initialize SwanLab: {e}")
    
    def log_scalar(self, tag: str, value: Union[float, int], step: int = None):
        """记录标量值
        
        Args:
            tag: 指标名称
            value: 指标值
            step: 步骤数
        """
        # 记录到TensorBoard
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.add_scalar(tag, value, step)
            except Exception as e:
                print(f"Failed to log to TensorBoard: {e}")
        
        # 记录到SwanLab
        if self.swanlab_run is not None:
            try:
                swanlab.log({tag: value}, step=step)
            except Exception as e:
                print(f"Failed to log to SwanLab: {e}")
                
    def add_scalar(self, tag: str, value: Union[float, int], step: int = None):
        """兼容TensorBoard的add_scalar方法，调用log_scalar
        
        Args:
            tag: 指标名称
            value: 指标值
            step: 步骤数
        """
        self.log_scalar(tag, value, step)
    
    def log_dict(self, metrics: Dict[str, Union[float, int]], step: int = None):
        """记录多个标量值
        
        Args:
            metrics: 指标字典 {指标名称: 指标值}
            step: 步骤数
        """
        # 记录到TensorBoard
        if self.tensorboard_writer is not None:
            try:
                for tag, value in metrics.items():
                    self.tensorboard_writer.add_scalar(tag, value, step)
            except Exception as e:
                print(f"Failed to log dict to TensorBoard: {e}")
        
        # 记录到SwanLab
        if self.swanlab_run is not None:
            try:
                swanlab.log(metrics, step=step)
            except Exception as e:
                print(f"Failed to log dict to SwanLab: {e}")
    
    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float] = None):
        """记录超参数
        
        Args:
            hparams: 超参数字典
            metrics: 对应的指标
        """
        # TensorBoard支持hparams
        if self.tensorboard_writer is not None:
            try:
                from torch.utils.tensorboard.summary import hparams
                # 简化的hparams记录
                if metrics is None:
                    metrics = {}
                self.tensorboard_writer.add_hparams(hparams, metrics)
            except Exception as e:
                print(f"Failed to log hparams to TensorBoard: {e}")
        
        # SwanLab通过config初始化时已记录超参数
        if self.swanlab_run is not None:
            try:
                # 更新配置
                for key, value in hparams.items():
                    swanlab.config[key] = value
                # 如果有指标，也记录
                if metrics is not None:
                    swanlab.log(metrics)
            except Exception as e:
                print(f"Failed to update config in SwanLab: {e}")
    
    def close(self):
        """关闭日志记录器"""
        # 关闭TensorBoard
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.close()
            except Exception as e:
                print(f"Failed to close TensorBoard writer: {e}")
        
        # SwanLab会自动关闭
        if self.swanlab_run is not None:
            try:
                swanlab.finish()
            except Exception as e:
                print(f"Failed to finish SwanLab run: {e}")
    
    def __del__(self):
        """析构函数，确保日志记录器被正确关闭"""
        self.close()


def create_logger(log_dir: Optional[str] = None, 
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 use_tensorboard: bool = True,
                 use_swanlab: bool = True) -> SwanLabLogger:
    """创建日志记录器的工厂函数
    
    Args:
        log_dir: 日志保存目录
        experiment_name: 实验名称
        config: 实验配置参数
        use_tensorboard: 是否使用TensorBoard
        use_swanlab: 是否使用SwanLab
        
    Returns:
        SwanLabLogger实例
    """
    return SwanLabLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config,
        use_tensorboard=use_tensorboard,
        use_swanlab=use_swanlab
    )