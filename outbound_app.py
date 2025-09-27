import time
import logging
import os
from freeswitchESL import ESL

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    'blacklist_numbers': ['1002'],
    'welcome_audio_path': '/Users/dongjb/soft/freeswitch/share/freeswitch/sounds/music/'
                          '48000/ponce-preludio-in-e-major.wav',
    'fallback_audio': '/Users/dongjb/soft/freeswitch/share/freeswitch/sounds/en/us/callie/voicemail/8000/vm-hello.wav',
    'bridge_destination': 'user/1001@192.168.1.4',
    'server_host': '0.0.0.0',
    'server_port': 8086
}


def check_audio_file(file_path):
    """检查音频文件是否存在"""
    if os.path.exists(file_path):
        logger.info(f"音频文件存在: {file_path}")
        return True
    else:
        logger.warning(f"音频文件不存在: {file_path}")
        return False


def establish_connection(conn):
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


def answer_call(conn):
    """应答呼叫"""
    try:
        conn.execute("answer")
        logger.info("呼叫应答成功")
        return True
    except Exception as e:
        logger.error(f"应答失败: {e}")
        return False


def get_caller_info(conn):
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


def is_blacklisted(caller_number):
    """检查号码是否在黑名单中"""
    return caller_number in CONFIG['blacklist_numbers']


def handle_blacklisted_call(conn, caller_number):
    """处理黑名单呼叫"""
    logger.warning(f"号码 {caller_number} 在黑名单中，挂断呼叫")
    try:
        conn.execute("hangup")
    except Exception as e:
        logger.error(f"挂断失败: {e}")


def play_audio_file(conn, file_path, description):
    """播放音频文件"""
    try:
        conn.execute("playback", file_path)
        logger.info(f"{description}播放完成")
        return True
    except Exception as e:
        logger.warning(f"{description}播放失败: {e}")
        return False


def play_welcome_audio(conn, caller_number):
    """播放欢迎音频"""
    logger.info(f"为主叫 {caller_number} 播放欢迎音频")
    audio_played = False
    
    # 尝试播放主音频文件
    if check_audio_file(CONFIG['welcome_audio_path']):
        audio_played = play_audio_file(conn, CONFIG['welcome_audio_path'], "主音频文件")
    else:
        logger.warning("主音频文件不存在，跳过播放")
    
    # 如果主音频播放失败，尝试播放备用音频
    if not audio_played:
        if check_audio_file(CONFIG['fallback_audio']):
            audio_played = play_audio_file(conn, CONFIG['fallback_audio'], "备用音频")
        else:
            logger.warning("备用音频文件也不存在，使用音调")
            audio_played = play_audio_file(conn, "tone_stream://%(1000,0,800,1000)", "音调")
    
    # 最后的备用方案
    if not audio_played:
        audio_played = play_audio_file(conn, "tone_stream://%(500,500,480,620)", "最终备用音调")
    
    # 如果音频播放成功，等待一段时间
    if audio_played:
        time.sleep(3)  # 等待音频播放完成
    
    return audio_played


def bridge_call(conn):
    """桥接呼叫"""
    logger.info(f"桥接呼叫到 {CONFIG['bridge_destination']}")
    try:
        conn.execute("bridge", CONFIG['bridge_destination'])
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


def handle_outbound_call(conn):
    """处理出站呼叫 - 主流程"""
    logger.info("开始处理出站呼叫")
    
    # 1. 建立连接
    if not establish_connection(conn):
        return
    
    # 2. 应答呼叫
    if not answer_call(conn):
        return
    
    # 3. 获取主叫信息
    caller_number = get_caller_info(conn)
    
    # 4. 检查黑名单
    if is_blacklisted(caller_number):
        handle_blacklisted_call(conn, caller_number)
        return
    
    # 5. 播放欢迎音频
    play_welcome_audio(conn, caller_number)
    
    # 6. 桥接呼叫
    bridge_call(conn)
    
    # 7. 清理连接
    # try:
    #     conn.disconnect()
    # except Exception as e:
    #     logger.error(f"断开连接时出错: {e}")


def main():
    """主函数 - 启动ESL服务器"""
    import socket

    host = CONFIG['server_host']
    port = CONFIG['server_port']

    logger.info(f"启动ESL服务器，监听 {host}:{port}")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        logger.info(f"等待出站连接...")
        
        while True:
            try:
                client_socket, addr = s.accept()
                logger.info(f"接受来自 {addr} 的连接")
                conn = ESL.ESLconnection(client_socket.fileno())
                handle_outbound_call(conn)
            except KeyboardInterrupt:
                logger.info("收到中断信号，正在关闭服务器...")
                break
            except Exception as e:
                logger.error(f"处理连接时出错: {e}")
                continue


if __name__ == "__main__":
    main()
