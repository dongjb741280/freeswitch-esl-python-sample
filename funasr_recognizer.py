#!/usr/bin/env python3
"""
FunASR 语音识别模块
提供实时语音识别功能
"""

import time
import logging
import threading
import queue
import numpy as np
from funasr import AutoModel
from typing import Optional

# 配置日志
logger = logging.getLogger(__name__)


class FunASRRecognizer:
    """FunASR实时语音识别器"""
    
    def __init__(self, model_name='paraformer-zh', device='cpu', model_revision='v2.0.4'):
        """
        初始化FunASR识别器
        
        Args:
            model_name (str): 模型名称
            device (str): 设备类型 ('cpu' 或 'cuda')
            model_revision (str): 模型版本
        """
        self.model_name = model_name
        self.device = device
        self.model_revision = model_revision
        self.model = None
        # 允许外部注入共享队列（生产/消费）
        self.audio_queue = queue.Queue()
        self.recognition_results = queue.Queue()
        self.is_running = False
        self.recognition_thread = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化FunASR模型"""
        try:
            logger.info(f"正在初始化FunASR模型: {self.model_name}")
            self.model = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                # model_revision=self.model_revision,
                # device=self.device,
                # disable_update=True,
                # disable_log=True
            )
            logger.info("FunASR模型初始化成功")
        except Exception as e:
            logger.error(f"FunASR模型初始化失败: {e}")
            self.model = None
    
    def start_recognition(self):
        """开始语音识别"""
        if not self.model:
            logger.error("FunASR模型未初始化，无法开始识别")
            return False
        
        if self.is_running:
            logger.warning("语音识别已在运行中")
            return True
        
        self.is_running = True
        self.recognition_thread = threading.Thread(target=self._recognition_worker)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        logger.info("语音识别已启动")
        return True
    
    def stop_recognition(self):
        """停止语音识别"""
        if not self.is_running:
            # logger.warning("语音识别未在运行")
            return
        
        self.is_running = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=5)
        logger.info("语音识别已停止")
    
    def add_audio_data(self, audio_data):
        """
        添加音频数据到识别队列
        
        Args:
            audio_data (list or np.ndarray): 音频数据
        """
        if not self.is_running:
            logger.warning("语音识别未启动，忽略音频数据")
            return
        
        try:
            # 确保音频数据是列表格式
            if isinstance(audio_data, np.ndarray):
                audio_data = audio_data.tolist()
            
            self.audio_queue.put(audio_data)
        except Exception as e:
            logger.error(f"添加音频数据失败: {e}")
    
    def get_recognition_result(self, timeout=1):
        """
        获取识别结果
        
        Args:
            timeout (float): 超时时间（秒）
            
        Returns:
            dict or None: 识别结果，包含text、timestamp、confidence字段
        """
        try:
            return self.recognition_results.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_results(self, timeout=1):
        """
        获取所有可用的识别结果
        
        Args:
            timeout (float): 超时时间（秒）
            
        Returns:
            list: 识别结果列表
        """
        results = []
        while True:
            result = self.get_recognition_result(timeout=timeout)
            if result is None:
                break
            results.append(result)
        return results
    
    def _recognition_worker(self):
        """识别工作线程"""
        audio_buffer = []
        buffer_size = 16000 * 2  # 2秒的音频缓冲（16kHz采样率）
        
        logger.info("识别工作线程已启动")
        
        while self.is_running:
            try:
                # 获取音频数据
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(audio_chunk)
                except queue.Empty:
                    continue
                
                # 当缓冲区有足够数据时进行识别
                if len(audio_buffer) >= buffer_size:
                    # 转换为numpy数组
                    audio_array = np.array(audio_buffer[:buffer_size], dtype=np.float32)
                    
                    # 进行语音识别
                    try:
                        result = self.model.generate(
                            input=audio_array,
                            cache={},
                            language="zh",
                            use_itn=True
                        )
                        
                        if result and len(result) > 0:
                            text = result[0].get("text", "").strip()
                            if text:
                                recognition_result = {
                                    'text': text,
                                    'timestamp': time.time(),
                                    'confidence': result[0].get("confidence", 0.0)
                                }
                                logger.info(f"识别结果: {text} (置信度: {recognition_result['confidence']:.2f})")
                                self.recognition_results.put(recognition_result)
                    except Exception as e:
                        logger.error(f"语音识别处理失败: {e}")
                    
                    # 保留部分音频数据用于连续识别
                    audio_buffer = audio_buffer[buffer_size:]
                
            except Exception as e:
                logger.error(f"识别工作线程出错: {e}")
                time.sleep(0.1)
        
        logger.info("识别工作线程已停止")
    
    def is_model_ready(self):
        """检查模型是否已准备就绪"""
        return self.model is not None
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_revision': self.model_revision,
            'is_ready': self.is_model_ready(),
            'is_running': self.is_running
        }
    
    def clear_queues(self):
        """清空队列"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.recognition_results.empty():
            try:
                self.recognition_results.get_nowait()
            except queue.Empty:
                break
        
        logger.info("队列已清空")


class FunASRConfig:
    """FunASR配置类"""
    
    def __init__(self):
        self.model_name = 'paraformer-zh'
        self.device = 'cpu'
        self.model_revision = 'v2.0.4'
        self.sample_rate = 16000
        self.buffer_duration = 2.0  # 秒
        self.language = 'zh'
        self.use_itn = True
    
    def to_dict(self):
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_revision': self.model_revision,
            'sample_rate': self.sample_rate,
            'buffer_duration': self.buffer_duration,
            'language': self.language,
            'use_itn': self.use_itn
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# 工厂函数
def create_recognizer(config=None):
    """
    创建FunASR识别器
    
    Args:
        config (FunASRConfig or dict): 配置对象或配置字典
        
    Returns:
        FunASRRecognizer: 识别器实例
    """
    if config is None:
        config = FunASRConfig()
    elif isinstance(config, dict):
        config = FunASRConfig.from_dict(config)
    
    return FunASRRecognizer(
        model_name=config.model_name,
        device=config.device,
        model_revision=config.model_revision
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建识别器
    recognizer = create_recognizer()
    
    if recognizer.is_model_ready():
        print("模型初始化成功")
        print(f"模型信息: {recognizer.get_model_info()}")
        
        # 启动识别
        if recognizer.start_recognition():
            print("识别已启动")
            
            # 模拟音频数据
            for i in range(10):
                audio_data = np.random.randn(1600).astype(np.float32) * 0.1
                recognizer.add_audio_data(audio_data)
                time.sleep(0.1)
            
            # 获取结果
            time.sleep(2)
            results = recognizer.get_all_results(timeout=0.1)
            print(f"识别结果数量: {len(results)}")
            
            # 停止识别
            recognizer.stop_recognition()
            print("识别已停止")
    else:
        print("模型初始化失败")
