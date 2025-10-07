#!/usr/bin/env python3
"""
AI 呼叫中心
基于 FunASR 的智能语音识别呼叫中心系统
"""

import time
import logging
import os
import threading
import asyncio
import websockets
import socket
import ESL
from funasr_recognizer import create_recognizer, FunASRConfig
from audio_stream_handler import create_freeswitch_handler, create_websocket_handler
from llm_qwen import generate_reply

import torch
import sys

# 修复torch.serialization.FILE_LIKE问题
if not hasattr(torch.serialization, 'FILE_LIKE'):
    torch.serialization.FILE_LIKE = torch.serialization.FileLike

# 现在可以安全导入ChatTTS
import ChatTTS
import torchaudio


# 配置日志（强制覆盖其他配置）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)


class AICallCenterConfig:
    """AI 呼叫中心配置类"""
    
    def __init__(self):
        self.blacklist_numbers = ['1003']
        self.welcome_audio_path = '/Users/dongjb/soft/freeswitch/share/freeswitch/sounds/freeswitch-sound-cn/zh/cn/callie/ivr/8000/ivr-welcome_to_freeswitch.wav'
        self.fallback_audio = '/Users/dongjb/soft/freeswitch/share/freeswitch/sounds/en/us/callie/voicemail/8000/vm-hello.wav'
        self.bridge_destination = 'user/1001@192.168.1.4'
        self.server_host = '0.0.0.0'
        self.server_port = 8086
        self.funasr_model = 'paraformer-zh'
        self.audio_sample_rate = 16000
        self.audio_chunk_size = 1024
        self.recognition_timeout = 30
        self.websocket_port = 8087
    
    def to_dict(self):
        """转换为字典"""
        return {
            'blacklist_numbers': self.blacklist_numbers,
            'welcome_audio_path': self.welcome_audio_path,
            'fallback_audio': self.fallback_audio,
            'bridge_destination': self.bridge_destination,
            'server_host': self.server_host,
            'server_port': self.server_port,
            'funasr_model': self.funasr_model,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_chunk_size': self.audio_chunk_size,
            'recognition_timeout': self.recognition_timeout,
            'websocket_port': self.websocket_port
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class AudioManager:
    """音频管理器"""
    
    def __init__(self, config):
        self.config = config
    
    def check_audio_file(self, file_path):
        """检查音频文件是否存在"""
        if os.path.exists(file_path):
            logger.info(f"音频文件存在: {file_path}")
            return True
        else:
            logger.warning(f"音频文件不存在: {file_path}")
            return False
    
    def play_audio_file(self, conn, file_path, description):
        """播放音频文件"""
        try:
            conn.execute("playback", file_path)
            logger.info(f"{description}播放完成")
            return True
        except Exception as e:
            logger.warning(f"{description}播放失败: {e}")
            return False
    
    def play_welcome_audio(self, conn, caller_number):
        """播放欢迎音频"""
        logger.info(f"为主叫 {caller_number} 播放欢迎音频")
        audio_played = False
        
        # 尝试播放主音频文件
        if self.check_audio_file(self.config.welcome_audio_path):
            audio_played = self.play_audio_file(conn, self.config.welcome_audio_path, "主音频文件")
        else:
            logger.warning("主音频文件不存在，跳过播放")
        
        # 如果主音频播放失败，尝试播放备用音频
        if not audio_played:
            if self.check_audio_file(self.config.fallback_audio):
                audio_played = self.play_audio_file(conn, self.config.fallback_audio, "备用音频")
            else:
                logger.warning("备用音频文件也不存在，使用音调")
                audio_played = self.play_audio_file(conn, "tone_stream://%(1000,0,800,1000)", "音调")
        
        # 最后的备用方案
        if not audio_played:
            audio_played = self.play_audio_file(conn, "tone_stream://%(500,500,480,620)", "最终备用音调")
        
        # 如果音频播放成功，等待一段时间
        if audio_played:
            time.sleep(3)  # 等待音频播放完成
        
        return audio_played


class CallHandler:
    """呼叫处理器"""
    
    def __init__(self, config, audio_manager):
        self.config = config
        self.audio_manager = audio_manager
    
    def establish_connection(self, conn):
        """建立ESL连接握手"""
        try:
            resp = conn.sendRecv("connect")
            logger.info(f"收到连接响应: {resp}")
            return True
        except Exception as e:
            logger.error(f"连接握手失败: {e}")
            try:
                conn.disconnect()
            except Exception as e:
                logger.error(f"断开连接时发生错误: {e}")
            return False
    
    def answer_call(self, conn):
        """应答呼叫"""
        try:
            conn.execute("answer")
            logger.info("呼叫应答成功")
            return True
        except Exception as e:
            logger.error(f"应答失败: {e}")
            return False
    
    def get_caller_info(self, conn):
        """获取主叫信息"""
        info = conn.getInfo()
        if not info:
            try:
                info = conn.recvEvent()
            except Exception as e:
                logger.error(f"获取呼叫信息时发生错误: {e}")
                info = None

        caller_number = None
        try:
            if info:
                caller_number = info.getHeader("Caller-Caller-ID-Number")
                logger.info(f"主叫号码: {caller_number}")
        except Exception as e:
            logger.error(f"获取主叫号码时发生错误: {e}")
            caller_number = None
        
        return caller_number
    
    def is_blacklisted(self, caller_number):
        """检查号码是否在黑名单中"""
        return caller_number in self.config.blacklist_numbers
    
    def handle_blacklisted_call(self, conn, caller_number):
        """处理黑名单呼叫"""
        logger.warning(f"号码 {caller_number} 在黑名单中，挂断呼叫")
        try:
            conn.execute("hangup")
        except Exception as e:
            logger.error(f"挂断失败: {e}")
    
    def bridge_call(self, conn):
        """桥接呼叫"""
        logger.info(f"桥接呼叫到 {self.config.bridge_destination}")
        try:
            conn.execute("bridge", self.config.bridge_destination)
            logger.info("桥接成功")
            return True
        except Exception as e:
            logger.error(f"桥接失败: {e}")
            try:
                conn.execute("playback", "tone_stream://%(500,500,480,620)")
                time.sleep(1)
            except Exception as e:
                logger.error(f"播放错误提示音失败: {e}")
            return False


class SpeechRecognitionHandler:
    """语音识别处理器"""
    
    def __init__(self, config, audio_manager):
        self.config = config
        self.audio_manager = audio_manager
        # 初始化FunASR配置和识别器
        funasr_config = FunASRConfig()
        funasr_config.model_name = self.config.funasr_model
        funasr_config.sample_rate = self.config.audio_sample_rate
        self.recognizer = create_recognizer({
            'model_name': funasr_config.model_name,
            'device': funasr_config.device,
            'model_revision': funasr_config.model_revision
        })
    
    def handle_speech_recognition(self, conn, caller_number):
        """处理语音识别"""
        logger.info(f"开始为主叫 {caller_number} 进行语音识别")
        
        # 启动前清空识别器的历史队列，避免上一通的残留数据被消费
        try:
            time.sleep(1)
            self.recognizer.clear_queues()
        except Exception:
            pass
        audio_handler = create_freeswitch_handler(
            conn,
            self.recognizer,
            sample_rate=self.config.audio_sample_rate
        )
        
        # 启动音频流处理
        if not audio_handler.start_audio_stream():
            logger.error("无法启动音频流处理")
            return None
        
        # 等待识别结果
        start_time = time.time()
        recognition_results = []
        
        try:
            while time.time() - start_time < self.config.recognition_timeout:
                result = self.recognizer.get_recognition_result(timeout=1)
                if result:
                    recognition_results.append(result)
                    logger.info(f"收到识别结果: {result['text']} (置信度: {result['confidence']:.2f})")
                
                # 检查是否有足够的结果
                if len(recognition_results) >= 3:  # 收集3个结果后停止
                    break
        
        except Exception as e:
            logger.error(f"语音识别过程中出错: {e}")
        
        finally:
            # 停止音频流处理
            audio_handler.stop_audio_stream()
            self.recognizer.stop_recognition()
        
        # 返回识别结果
        if recognition_results:
            # 顺序合并为完整文本（最长后缀重叠，避免重复与缺失）
            combined_text = ""
            unique_segments = []
            seen = set()
            max_conf = 0.0
            for r in recognition_results:
                seg = (r.get('text') or '').strip()
                conf = float(r.get('confidence') or 0.0)
                if not seg:
                    continue
                if seg not in seen:
                    seen.add(seg)
                    unique_segments.append(seg)
                combined_text = self._append_without_dup(combined_text, seg)
                if conf > max_conf:
                    max_conf = conf

            final_result = {
                'text': combined_text,
                'confidence': max_conf,
                'segments': unique_segments,
                'timestamp': time.time()
            }
            logger.info(f"最终完整文本: {final_result['text']} (max置信度: {final_result['confidence']:.2f})")

            # 调用大模型生成回复（日志记录）
            try:
                reply = generate_reply(final_result['text'])
                if reply:
                    logger.info(f"LLM 回复: {reply}")
            except Exception as e:
                logger.error(f"调用大模型失败: {e}")
            return final_result
        
        logger.warning("未获得有效的语音识别结果")
        return None
    
    def process_recognition_result(self, conn, result, caller_number):
        """处理识别结果"""
        if not result:
            logger.info("无识别结果，继续桥接")
            return True
        
        text = result['text'].lower()
        confidence = result['confidence']
        
        logger.info(f"处理识别结果: '{text}' (置信度: {confidence:.2f})")
        
        # 根据识别结果进行不同的处理
        if confidence < 0.5:
            logger.info("识别置信度较低，继续桥接")
            return True
        
        # 关键词匹配
        if any(keyword in text for keyword in ['转接', '转人工', '人工服务']):
            logger.info("用户请求转接人工服务")
            try:
                conn.execute("playback", "tone_stream://%(1000,0,800,1000)")
                time.sleep(1)
                conn.execute("playback", "tone_stream://%(1000,0,800,1000)")
                time.sleep(1)
            except Exception as e:
                logger.error(f"播放转接提示音失败: {e}")
            return True
        
        elif any(keyword in text for keyword in ['挂断', '结束', '再见']):
            logger.info("用户请求挂断")
            try:
                conn.execute("playback", "tone_stream://%(500,500,480,620)")
                time.sleep(1)
                conn.execute("hangup")
            except Exception as e:
                logger.error(f"挂断失败: {e}")
            return False
        
        elif any(keyword in text for keyword in ['重播', '再听一遍']):
            logger.info("用户请求重播欢迎音频")
            self.audio_manager.play_welcome_audio(conn, caller_number)
            return True
        
        else:
            logger.info(f"未识别的指令: {text}，继续桥接")
            return True

    def _append_without_dup(self, accumulated, new_text):
        """将 new_text 以最长后缀重叠方式追加到 accumulated，避免重复与缺失。"""
        if not accumulated:
            return new_text
        if not new_text:
            return accumulated
        if new_text in accumulated:
            return accumulated
        max_overlap = min(len(accumulated), len(new_text))
        for k in range(max_overlap, 0, -1):
            if accumulated.endswith(new_text[:k]):
                return accumulated + new_text[k:]
        return accumulated + new_text

    def simple_record_and_recognize(self, conn, duration=5,
                                    input_path="/Users/dongjb/IdeaProjects/freeswitch/freeswitch-esl-python-sample/tmp/call_audio.wav"):
        """
        简单录音并一次性识别函数

        Args:
            conn: FreeSWITCH ESL连接
            duration (int): 录音时长（秒）
            output_path (str): 录音文件路径

        Returns:
            str: 识别结果文本，失败返回None
        """
        try:
            logger.info(f"开始录音 {duration} 秒到 {input_path}")

            # 删除旧文件
            if os.path.exists(input_path):
                os.remove(input_path)

            time.sleep(1)
            # 开始录音
            conn.execute("record_session", input_path)
            logger.info(f"录音中... 等待 {duration} 秒")

            # 等待录音完成
            time.sleep(duration)

            # 停止录音
            conn.execute("stop_record_session", input_path)
            logger.info("录音完成")

            # 检查录音文件是否存在
            if not os.path.exists(input_path):
                logger.error("录音文件不存在")
                return None

            res = self.recognizer.model.generate(input=input_path,batch_size_s=300)
            print(res[0]["text"])
            return res[0]["text"]

        except Exception as e:
            logger.error(f"处理音频文件失败: {e}")
            return None

        finally:
            self.recognizer.stop_recognition()


def _append_without_dup(accumulated, new_text):
    """将 new_text 以最长后缀重叠方式追加到 accumulated，避免重复与缺失。"""
    if not accumulated:
        return new_text
    if not new_text:
        return accumulated
    if new_text in accumulated:
        return accumulated
    max_overlap = min(len(accumulated), len(new_text))
    for k in range(max_overlap, 0, -1):
        if accumulated.endswith(new_text[:k]):
            return accumulated + new_text[k:]
    return accumulated + new_text


class AICallCenter:
    """AI 呼叫中心主类"""
    
    def __init__(self, config=None):
        self.config = config or AICallCenterConfig()
        self.audio_manager = AudioManager(self.config)
        self.call_handler = CallHandler(self.config, self.audio_manager)
        self.speech_handler = SpeechRecognitionHandler(self.config, self.audio_manager)
        self.websocket_server = None
        self.esl_server = None
        self.is_running = False
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)  # Set to True for better performance

    def handle_call(self, conn):
        """处理呼叫 - 主流程"""
        logger.info("开始处理呼叫")
        
        # 1. 建立连接
        if not self.call_handler.establish_connection(conn):
            return
        
        # 2. 应答呼叫
        if not self.call_handler.answer_call(conn):
            return
        
        # 3. 获取主叫信息
        caller_number = self.call_handler.get_caller_info(conn)
        
        # 4. 检查黑名单
        if self.call_handler.is_blacklisted(caller_number):
            self.call_handler.handle_blacklisted_call(conn, caller_number)
            return
        
        # 5. 播放欢迎音频
        self.audio_manager.play_welcome_audio(conn, caller_number)

            # 6. 使用FunASR进行语音识别
            # recognition_result = self.speech_handler.handle_speech_recognition(conn, caller_number)

            # 7. 处理识别结果
            # should_continue = self.speech_handler.process_recognition_result(conn, recognition_result, caller_number)

        # 6. 使用FunASR进行语音识别
        recognize_text = self.speech_handler.simple_record_and_recognize(conn)

        # 7. 调用大模型生成回复
        try:
            reply = generate_reply(recognize_text)
            if reply:
                logger.info(f"LLM 回复: {reply}")
        except Exception as e:
            logger.error(f"调用大模型失败: {e}")

        # 8. tts 回复内容
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            temperature=0.3,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )

        wavs = self.chat.infer(reply,
                               stream=False,
                               skip_refine_text=True,  # 跳过润色，加快推理
                               max_split_batch=8,  # 并行批次加大
                               params_infer_code=params_infer_code)
        # 确保音频数据格式正确
        if wavs and len(wavs) > 0:
            torchaudio.save(
                "./tmp/output.wav",
                torch.from_numpy(wavs[0]).unsqueeze(0),
                24000
            )
            logger.info("✅ TTS合成成功，音频文件已保存为 output.wav")
        else:
            logger.error("❌ TTS合成失败：未生成音频数据")

        # 9. 播放回复音频
        self.audio_manager.play_audio_file(conn, "/Users/dongjb/IdeaProjects/freeswitch/freeswitch-esl-python-sample/tmp/output.wav", "回复音频")



        # 8. 根据处理结果决定是否桥接
        # if should_continue:
        #     self.call_handler.bridge_call(conn)
        # else:
        #     logger.info("根据语音识别结果，跳过桥接")

    async def websocket_audio_handler(self, websocket, path):

        """WebSocket音频流处理器"""
        logger.info(f"WebSocket连接建立: {websocket.remote_address}")
        
        # 创建FunASR配置
        funasr_config = FunASRConfig()
        funasr_config.model_name = self.config.funasr_model
        funasr_config.sample_rate = self.config.audio_sample_rate
        
        # 创建识别器和WebSocket处理器
        recognizer = create_recognizer(funasr_config)
        websocket_handler = create_websocket_handler(
            recognizer,
            sample_rate=self.config.audio_sample_rate
        )
        
        # 处理WebSocket连接
        await websocket_handler.handle_websocket_connection(websocket, path)
    
    def start_websocket_server(self):
        """启动WebSocket服务器"""
        async def run_server():
            logger.info(f"启动WebSocket服务器，监听端口 {self.config.websocket_port}")
            
            # 创建包装函数来处理WebSocket连接
            async def websocket_handler(websocket, path="/audio"):
                await self.websocket_audio_handler(websocket, path)
            
            # 使用正确的websockets.serve调用方式
            server = await websockets.serve(websocket_handler, "0.0.0.0", self.config.websocket_port)
            logger.info(f"WebSocket服务器已启动，监听 {server.sockets[0].getsockname()}")
            
            # 保持服务器运行
            await server.wait_closed()
        
        # 在单独线程中运行WebSocket服务器
        self.websocket_server = threading.Thread(target=lambda: asyncio.run(run_server()))
        self.websocket_server.daemon = True
        self.websocket_server.start()
    
    def start_esl_server(self):
        """启动ESL服务器"""
        host = self.config.server_host
        port = self.config.server_port
        
        logger.info(f"启动ESL服务器，监听 {host}:{port}")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            logger.info(f"等待呼叫连接...")
            
            while self.is_running:
                try:
                    client_socket, addr = s.accept()
                    logger.info(f"接受来自 {addr} 的连接")
                    conn = ESL.ESLconnection(client_socket.fileno())
                    conn.send("linger\n\n")
                    self.handle_call(conn)
                except KeyboardInterrupt:
                    logger.info("收到中断信号，正在关闭服务器...")
                    break
                except Exception as e:
                    logger.error(f"处理连接时出错: {e}")
                    continue
    
    def start(self):
        """启动呼叫中心"""
        self.is_running = True
        logger.info("开始启动 AI 呼叫中心...")
        
        # 启动WebSocket服务器
        logger.info("启动 WebSocket 服务器...")
        self.start_websocket_server()
        logger.info("WebSocket 服务器启动完成")
        
        # 启动ESL服务器
        logger.info("启动 ESL 服务器...")
        self.start_esl_server()
        logger.info("ESL 服务器启动完成")
    
    def stop(self):
        """停止呼叫中心"""
        self.is_running = False
        logger.info("AI 呼叫中心已停止")
    
    def get_status(self):
        """获取呼叫中心状态"""
        return {
            'is_running': self.is_running,
            'config': self.config.to_dict(),
            'websocket_port': self.config.websocket_port,
            'esl_port': self.config.server_port
        }


def main():
    """主函数 - 启动 AI 呼叫中心"""
    try:
        logger.info("程序启动: 进入 main()")
        # 创建呼叫中心实例
        call_center = AICallCenter()
        
        # 显示呼叫中心状态
        status = call_center.get_status()
        logger.info(f"AI 呼叫中心状态: {status}")
        
        # 启动呼叫中心
        call_center.start()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭 AI 呼叫中心...")
    except Exception as e:
        logger.error(f"AI 呼叫中心启动失败: {e}")
    finally:
        if 'call_center' in locals():
            call_center.stop()


if __name__ == "__main__":
    main()
