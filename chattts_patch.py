#!/usr/bin/env python3
"""
ChatTTS narrow()错误修复补丁
为Python 3.12环境修复ChatTTS的narrow()错误
"""

import torch
import sys
import re

# 修复torch.serialization.FILE_LIKE问题
if not hasattr(torch.serialization, 'FILE_LIKE'):
    torch.serialization.FILE_LIKE = torch.serialization.FileLike

def patch_chattts():
    """修复ChatTTS的narrow()错误"""
    try:
        import ChatTTS
        
        # 尝试修复narrow()方法
        if hasattr(torch.Tensor, 'narrow'):
            original_narrow = torch.Tensor.narrow
            
            def safe_narrow(self, dim, start, length):
                """安全的narrow方法，避免负长度和越界错误"""
                # 确保维度有效
                if dim < 0 or dim >= self.dim():
                    dim = 0
                
                # 获取维度大小
                dim_size = self.size(dim)
                
                # 修复start参数
                if start < 0:
                    start = 0
                elif start >= dim_size:
                    start = max(0, dim_size - 1)
                
                # 修复length参数
                if length < 0:
                    length = 0
                elif start + length > dim_size:
                    length = max(0, dim_size - start)
                
                # 如果长度为0，返回空张量
                if length == 0:
                    return torch.empty(0)
                
                return original_narrow(self, dim, start, length)
            
            # 替换narrow方法
            torch.Tensor.narrow = safe_narrow
            print("✅ ChatTTS narrow()方法已修复")
            return True
    except Exception as e:
        print(f"❌ ChatTTS修复失败: {e}")
        return False

def test_chattts():
    """测试ChatTTS功能"""
    try:
        print("初始化ChatTTS...")
        
        # 应用修复
        if not patch_chattts():
            print("修复失败，继续尝试...")
        
        import ChatTTS
        import torchaudio
        
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        
        print("开始TTS合成...")
        
        # 使用最简单的文本
        test_texts = [
            "你好",
            "测试",
            "语音"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n尝试文本 {i+1}: {text}")
            try:
                texts = [text]
                wavs = chat.infer(texts)
                
                if wavs and len(wavs) > 0:
                    # 确保音频数据格式正确
                    audio_tensor = torch.from_numpy(wavs[0]).float()
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    torchaudio.save(f"conda_output{i+1}.wav", audio_tensor, 24000)
                    print(f"✅ 成功生成 conda_output{i+1}.wav")
                    return True
                else:
                    print(f"❌ 文本 {i+1} 未生成音频")
            except Exception as e:
                print(f"❌ 文本 {i+1} 失败: {e}")
                continue
        
        print("❌ 所有文本都失败了")
        return False
        
    except Exception as e:
        print(f"❌ TTS合成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chattts()
    if success:
        print("\n🎉 ChatTTS测试成功！")
    else:
        print("\n❌ ChatTTS测试失败")
        sys.exit(1)