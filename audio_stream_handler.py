#!/usr/bin/env python3
"""
音频流处理模块
处理FreeSWITCH音频流和WebSocket音频流
"""

import time
import logging
import threading
import queue
import numpy as np
import json
import asyncio
import websockets
import os
from typing import Optional, Callable, Dict, Any
from llm_qwen import generate_reply

# 配置日志
logger = logging.getLogger(__name__)


class AudioStreamHandler:
    """音频流处理器"""
    
    def __init__(self, recognizer, sample_rate=16000, chunk_duration=0.1):
        """
        初始化音频流处理器
        
        Args:
            recognizer: 语音识别器实例
            sample_rate (int): 音频采样率
            chunk_duration (float): 音频块持续时间（秒）
        """
        self.recognizer = recognizer
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.is_streaming = False
        self.stream_thread = None
        # 使用共享队列（若提供）作为生产缓冲区
        self.audio_buffer = queue.Queue()
        self.callbacks = {}
    
    def start_audio_stream(self):
        """开始音频流处理"""
        if not self.recognizer.is_running:
            if not self.recognizer.start_recognition():
                logger.error("无法启动语音识别器")
                return False
        
        if self.is_streaming:
            logger.warning("音频流处理已在运行中")
            return True
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("音频流处理已启动")
        return True
    
    def stop_audio_stream(self):
        """停止音频流处理"""
        if not self.is_streaming:
            logger.warning("音频流处理未在运行")
            return
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logger.info("音频流处理已停止")
    
    def add_audio_data(self, audio_data):
        """
        添加音频数据
        
        Args:
            audio_data (list or np.ndarray): 音频数据
        """
        if self.is_streaming:
            self.audio_buffer.put(audio_data)
    
    def _stream_worker(self):
        """音频流工作线程"""
        logger.info("音频流工作线程已启动")
        
        try:
            while self.is_streaming:
                try:
                    # 从缓冲区获取音频数据
                    audio_data = self.audio_buffer.get(timeout=0.1)
                    
                    # 处理音频数据
                    processed_data = self._process_audio_data(audio_data)
                    
                    # 发送到识别器
                    if processed_data is not None:
                        self.recognizer.add_audio_data(processed_data)
                    
                    # 触发回调
                    self._trigger_callbacks('audio_processed', {
                        'audio_data': processed_data,
                        'timestamp': time.time()
                    })
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"处理音频数据时出错: {e}")
        
        except Exception as e:
            logger.error(f"音频流工作线程出错: {e}")
        
        logger.info("音频流工作线程已停止")
    
    def _process_audio_data(self, audio_data):
        """
        处理音频数据
        
        Args:
            audio_data: 原始音频数据
            
        Returns:
            处理后的音频数据
        """
        try:
            # 转换为numpy数组
            if isinstance(audio_data, list):
                audio_array = np.array(audio_data, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data.astype(np.float32)
            else:
                logger.error(f"不支持的音频数据类型: {type(audio_data)}")
                return None
            
            # 音频预处理（可以在这里添加降噪、增益等处理）
            processed_audio = self._preprocess_audio(audio_array)
            
            return processed_audio.tolist()
        
        except Exception as e:
            logger.error(f"音频数据处理失败: {e}")
            return None
    
    def _preprocess_audio(self, audio_array):
        """
        音频预处理
        
        Args:
            audio_array (np.ndarray): 音频数组
            
        Returns:
            np.ndarray: 预处理后的音频数组
        """
        # 简单的音频预处理
        # 1. 归一化
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # 2. 简单的降噪（可以添加更复杂的算法）
        # 这里只是示例，实际应用中可以使用更复杂的降噪算法
        
        return audio_array
    
    def add_callback(self, event_name: str, callback: Callable):
        """
        添加事件回调
        
        Args:
            event_name (str): 事件名称
            callback (Callable): 回调函数
        """
        if event_name not in self.callbacks:
            self.callbacks[event_name] = []
        self.callbacks[event_name].append(callback)
    
    def remove_callback(self, event_name: str, callback: Callable):
        """
        移除事件回调
        
        Args:
            event_name (str): 事件名称
            callback (Callable): 回调函数
        """
        if event_name in self.callbacks:
            try:
                self.callbacks[event_name].remove(callback)
            except ValueError:
                pass
    
    def _trigger_callbacks(self, event_name: str, data: Dict[str, Any]):
        """
        触发事件回调
        
        Args:
            event_name (str): 事件名称
            data (dict): 事件数据
        """
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {e}")
    
    def get_stream_info(self):
        """获取流信息"""
        return {
            'is_streaming': self.is_streaming,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'chunk_size': self.chunk_size,
            'buffer_size': self.audio_buffer.qsize()
        }


class FreeSWITCHAudioHandler(AudioStreamHandler):
    """FreeSWITCH音频流处理器"""
    
    def __init__(self, conn, recognizer, sample_rate=16000, chunk_duration=0.1):
        """
        初始化FreeSWITCH音频处理器
        
        Args:
            conn: FreeSWITCH ESL连接
            recognizer: 语音识别器实例
            sample_rate (int): 音频采样率
            chunk_duration (float): 音频块持续时间（秒）
        """
        super().__init__(recognizer, sample_rate, chunk_duration)
        self.conn = conn
        self.recording_path = None
    
    def start_recording(self, recording_path="/Users/dongjb/IdeaProjects/freeswitch/freeswitch-esl-python-sample/tmp/call_audio.wav"):
        """
        开始录音
        
        Args:
            recording_path (str): 录音文件路径
        """
        self.recording_path = recording_path
        try:
            # 如果存在上一通的录音文件，先删除避免读取残留数据
            try:
                if os.path.exists(self.recording_path):
                    os.remove(self.recording_path)
            except Exception:
                pass
            # 使用FreeSWITCH的record_session命令开始录音
            # 格式: record_session /path/to/file.wav
            self.conn.execute("record_session", recording_path)
            logger.info(f"开始录音: {recording_path}")
            
            # 确保录音文件目录存在
            os.makedirs(os.path.dirname(recording_path), exist_ok=True)
            
        except Exception as e:
            logger.error(f"启动录音失败: {e}")
    
    def stop_recording(self):
        """停止录音"""
        try:
            if self.recording_path:
                # 使用FreeSWITCH的stop_record_session命令停止录音
                self.conn.execute("stop_record_session", self.recording_path)
                logger.info(f"录音已停止: {self.recording_path}")
        except Exception as e:
            logger.error(f"停止录音失败: {e}")
    
    def _stream_worker(self):
        """FreeSWITCH音频流工作线程"""
        logger.info("FreeSWITCH音频流工作线程已启动")
        
        # 启动录音
        self.start_recording()
        
        try:
            # 使用FreeSWITCH的录音功能获取实时音频流
            self._capture_realtime_audio()
        
        except Exception as e:
            logger.error(f"FreeSWITCH音频流处理出错: {e}")
        finally:
            self.stop_recording()
        
        logger.info("FreeSWITCH音频流工作线程已停止")
    
    def _capture_realtime_audio(self):
        """从FreeSWITCH捕获实时音频数据"""
        import subprocess
        import threading
        import queue
        import struct
        
        # 使用sox或ffmpeg从录音文件实时读取音频数据
        audio_queue = queue.Queue()
        
        def read_audio_file():
            """读取录音文件的线程"""
            try:
                # 使用tail -f 实时读取录音文件
                # 这里使用一个简单的文件监控方法
                last_size = 0
                while self.is_streaming:
                    try:
                        if os.path.exists(self.recording_path):
                            current_size = os.path.getsize(self.recording_path)
                            if current_size > last_size:
                                # 读取新增的音频数据
                                with open(self.recording_path, 'rb') as f:
                                    f.seek(last_size)
                                    new_data = f.read(current_size - last_size)
                                    logger.info(f"读取到新的音频数据: {len(new_data)} 字节")
                                    if new_data:
                                        audio_queue.put(new_data)
                                last_size = current_size
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"读取音频文件出错: {e}")
                        time.sleep(0.1)
            except Exception as e:
                logger.error(f"音频文件读取线程出错: {e}")
        
        def process_audio_data():
            """处理音频数据的线程"""
            try:
                buffer = b''
                header_parsed = False
                data_offset = 0
                input_sample_rate = self.sample_rate
                num_channels = 1
                bits_per_sample = 16
                header_buffer = b''
                while self.is_streaming:
                    try:
                        # 从队列获取音频数据
                        data = audio_queue.get(timeout=0.1)
                        buffer += data
                        
                        # 解析并跳过WAV头（仅一次）
                        if not header_parsed:
                            header_buffer += buffer
                            # 至少需要WAV基础头长度
                            if len(header_buffer) >= 44 and header_buffer[0:4] == b'RIFF' and header_buffer[8:12] == b'WAVE':
                                # 解析fmt chunk
                                # 偏移12处应为第一个子块ID
                                offset = 12
                                fmt_chunk_found = False
                                data_chunk_found = False
                                while offset + 8 <= len(header_buffer):
                                    chunk_id = header_buffer[offset:offset+4]
                                    chunk_size = struct.unpack('<I', header_buffer[offset+4:offset+8])[0]
                                    next_offset = offset + 8 + chunk_size
                                    if chunk_id == b'fmt ' and offset + 8 + 16 <= len(header_buffer):
                                        fmt_chunk_found = True
                                        audio_format = struct.unpack('<H', header_buffer[offset+8:offset+10])[0]
                                        num_channels = struct.unpack('<H', header_buffer[offset+10:offset+12])[0]
                                        input_sample_rate = struct.unpack('<I', header_buffer[offset+12:offset+16])[0]
                                        # bits_per_sample 通常在fmt chunk的第14-15字节（当PCM时，fmt chunk size >= 16）
                                        if offset + 8 + chunk_size >= offset + 8 + 16:
                                            bits_per_sample = struct.unpack('<H', header_buffer[offset+22:offset+24])[0]
                                    if chunk_id == b'data':
                                        data_chunk_found = True
                                        data_offset = offset + 8
                                        break
                                    offset = next_offset
                                if fmt_chunk_found and data_chunk_found and len(header_buffer) >= data_offset:
                                    # 丢弃头部
                                    buffer = header_buffer[data_offset:]
                                    header_parsed = True
                                else:
                                    # 头尚未完整，继续累积
                                    continue
                            else:
                                # 非WAV或头不完整，继续累积直到足够
                                if len(header_buffer) < 44:
                                    continue
                                # 如果不是WAV文件（极端情况），按裸PCM处理
                                header_parsed = True
                                buffer = header_buffer
                        
                        # 当缓冲区有足够数据时按输入采样率的时长块处理
                        bytes_per_sample = max(1, bits_per_sample // 8)
                        frame_size = bytes_per_sample * max(1, num_channels)
                        input_chunk_size = max(1, int(self.chunk_duration * max(1, input_sample_rate)))
                        need_bytes = input_chunk_size * frame_size
                        while len(buffer) >= need_bytes:
                            chunk_data = buffer[:need_bytes]
                            buffer = buffer[need_bytes:]
                            
                            # 转换为int16/对应位深
                            if bits_per_sample == 16:
                                audio_array = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32)
                                audio_array = audio_array.reshape(-1, num_channels) if num_channels > 1 else audio_array.reshape(-1, 1)
                                # 转单声道
                                if num_channels > 1:
                                    audio_array = audio_array.mean(axis=1)
                                else:
                                    audio_array = audio_array[:, 0]
                                # 归一化到[-1, 1]
                                audio_array = audio_array / 32768.0
                            else:
                                logger.warning(f"不支持的位深: {bits_per_sample}，按16位处理可能失真")
                                audio_array = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # 采样率对齐到处理器采样率
                            target_sr = self.sample_rate
                            if input_sample_rate != target_sr and len(audio_array) > 1:
                                src_len = len(audio_array)
                                dst_len = max(1, int(src_len * target_sr / max(1, input_sample_rate)))
                                x = np.linspace(0, src_len - 1, num=src_len)
                                x_new = np.linspace(0, src_len - 1, num=dst_len)
                                audio_array = np.interp(x_new, x, audio_array).astype(np.float32)
                            
                            # 送入识别器
                            self.recognizer.add_audio_data(audio_array.tolist())
                            logger.info(f"处理音频数据: {len(audio_array)} 样本")
                            
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"处理音频数据出错: {e}")
                        time.sleep(0.1)
            except Exception as e:
                logger.error(f"音频数据处理线程出错: {e}")
        
        # 启动读取和处理线程
        read_thread = threading.Thread(target=read_audio_file)
        process_thread = threading.Thread(target=process_audio_data)
        
        read_thread.daemon = True
        process_thread.daemon = True
        
        read_thread.start()
        process_thread.start()
        
        # 等待流结束
        while self.is_streaming:
            time.sleep(0.1)
        
        # 等待线程结束
        read_thread.join(timeout=1)
        process_thread.join(timeout=1)


class WebSocketAudioHandler:
    """WebSocket音频流处理器"""
    
    def __init__(self, recognizer, sample_rate=16000, chunk_duration=0.1):
        """
        初始化WebSocket音频处理器
        
        Args:
            recognizer: 语音识别器实例
            sample_rate (int): 音频采样率
            chunk_duration (float): 音频块持续时间（秒）
        """
        self.recognizer = recognizer
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.websocket = None
        self.is_connected = False
    
    async def handle_websocket_connection(self, websocket, path):
        """
        处理WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            path: 连接路径
        """
        self.websocket = websocket
        self.is_connected = True
        logger.info(f"WebSocket连接建立: {websocket.remote_address}")
        
        # 启动识别器
        if not self.recognizer.is_running:
            if not self.recognizer.start_recognition():
                await websocket.close()
                return
        
        try:
            async for message in websocket:
                try:
                    # 解析音频数据
                    audio_data = json.loads(message)
                    if 'audio' in audio_data:
                        # 处理音频数据
                        processed_audio = self._process_websocket_audio(audio_data['audio'])
                        if processed_audio is not None:
                            # 添加到识别器
                            self.recognizer.add_audio_data(processed_audio)
                            
                            # 获取识别结果
                            result = self.recognizer.get_recognition_result(timeout=0.1)
                            if result:
                                # 发送识别结果回客户端
                                await websocket.send(json.dumps({
                                    'type': 'recognition_result',
                                    'text': result['text'],
                                    'confidence': result['confidence'],
                                    'timestamp': result['timestamp']
                                }))

                                # 调用大模型获取回复（非阻塞方式）
                                try:
                                    reply = generate_reply(result['text'])
                                    if reply:
                                        await websocket.send(json.dumps({
                                            'type': 'llm_reply',
                                            'text': reply,
                                            'source': 'qwen'
                                        }))
                                except Exception as e:
                                    logger.error(f"大模型调用失败: {e}")
                except json.JSONDecodeError:
                    logger.error("收到无效的JSON数据")
                except Exception as e:
                    logger.error(f"处理WebSocket消息时出错: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket连接已关闭")
        except Exception as e:
            logger.error(f"WebSocket处理出错: {e}")
        finally:
            self.is_connected = False
            self.recognizer.stop_recognition()
    
    def _process_websocket_audio(self, audio_data):
        """
        处理WebSocket音频数据
        
        Args:
            audio_data: 音频数据
            
        Returns:
            处理后的音频数据
        """
        try:
            # 转换为numpy数组
            if isinstance(audio_data, list):
                audio_array = np.array(audio_data, dtype=np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data.astype(np.float32)
            else:
                logger.error(f"不支持的音频数据类型: {type(audio_data)}")
                return None
            
            # 音频预处理
            processed_audio = self._preprocess_audio(audio_array)
            
            return processed_audio.tolist()
        
        except Exception as e:
            logger.error(f"WebSocket音频数据处理失败: {e}")
            return None
    
    def _preprocess_audio(self, audio_array):
        """
        音频预处理
        
        Args:
            audio_array (np.ndarray): 音频数组
            
        Returns:
            np.ndarray: 预处理后的音频数组
        """
        # 简单的音频预处理
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        return audio_array
    
    def get_connection_info(self):
        """获取连接信息"""
        return {
            'is_connected': self.is_connected,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'chunk_size': self.chunk_size
        }


# 工厂函数
def create_freeswitch_handler(conn, recognizer, **kwargs):
    """
    创建FreeSWITCH音频处理器
    
    Args:
        conn: FreeSWITCH ESL连接
        recognizer: 语音识别器实例
        **kwargs: 其他参数
        
    Returns:
        FreeSWITCHAudioHandler: 音频处理器实例
    """
    return FreeSWITCHAudioHandler(conn, recognizer, **kwargs)


def create_websocket_handler(recognizer, **kwargs):
    """
    创建WebSocket音频处理器
    
    Args:
        recognizer: 语音识别器实例
        **kwargs: 其他参数
        
    Returns:
        WebSocketAudioHandler: 音频处理器实例
    """
    return WebSocketAudioHandler(recognizer, **kwargs)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟识别器
    class MockRecognizer:
        def __init__(self):
            self.is_running = False
        
        def start_recognition(self):
            self.is_running = True
            return True
        
        def stop_recognition(self):
            self.is_running = False
        
        def add_audio_data(self, data):
            pass
    
    # 测试音频流处理器
    mock_recognizer = MockRecognizer()
    handler = AudioStreamHandler(mock_recognizer)
    
    print(f"流信息: {handler.get_stream_info()}")
    
    # 测试启动和停止
    if handler.start_audio_stream():
        print("音频流处理已启动")
        time.sleep(1)
        handler.stop_audio_stream()
        print("音频流处理已停止")
    
    print("测试完成")
