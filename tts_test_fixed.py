import ChatTTS
import torch
import torchaudio
import time


def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return result

    return wrapper


@time_execution
def process_tts(texts):
    chat = ChatTTS.Chat()
    print("Loading model...")
    chat.load(compile=False)

    print("Processing text to speech...")
    wavs = chat.infer(texts)
    return wavs


texts = [
    "毛泽东诞辰是指中国共产党、中国人民解放军和中华人民共和国的主要缔造者之一，中国各族人民的伟大领袖毛泽东同志的出生纪念日。毛泽东于1893年12月26日出生于湖南省湘潭县韶山冲的一个农民家庭。"]

wavs = process_tts(texts)

print("Saving audio file...")
torchaudio.save(
    "output1.wav",
    torch.from_numpy(wavs[0]).unsqueeze(0),
    24000
)