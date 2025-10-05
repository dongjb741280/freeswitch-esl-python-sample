#!/usr/bin/env python3
"""
简单的 HTTP 服务器
用于提供 Web 录音页面
"""

import http.server
import socketserver
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义 HTTP 请求处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        # 添加 CORS 头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        """处理 GET 请求"""
        if self.path == '/' or self.path == '/index.html':
            # 重定向到录音页面
            self.path = '/web_recorder.html'
        
        return super().do_GET()
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        logger.info(f"{self.address_string()} - {format % args}")


def start_http_server(port=8080):
    """启动 HTTP 服务器"""
    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            logger.info(f"HTTP 服务器启动，监听端口 {port}")
            logger.info(f"访问地址: http://localhost:{port}")
            logger.info("按 Ctrl+C 停止服务器")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")


if __name__ == "__main__":
    import sys
    
    # 获取端口参数
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error("无效的端口号")
            sys.exit(1)
    
    start_http_server(port)
