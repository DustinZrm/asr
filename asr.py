import logging  # 导入日志模块
import sys  # 导入系统模块
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)  # 配置日志输出格式

import os  # 导入os模块
import numpy as np  # 导入numpy库用于数值计算
import time  # 导入时间模块
import uuid  # 导入uuid库用于生成唯一标识符
from funasr import AutoModel  # 导入funasr的AutoModel类
from funasr.utils.postprocess_utils import rich_transcription_postprocess  # 导入语音转文本后处理函数
from modelscope.models.base import Model  # 导入modelscope模型基类
import pyaudio

# 常量定义
SAMPLE_RATE = 16000  # 采样率设置为16kHz

# 定义ModelManager类，用于加载和管理模型
class ModelManager:
    def __init__(self):
        self.vad_model = None  # 初始化VAD（语音活动检测）模型
        self.sense_model = None  # 初始化Sense模型
        self.sv_model = None  # 初始化语音识别（SV）模型

    def load_models(self):  # 加载所有模型
        # 加载VAD模型
        self.vad_model = AutoModel(
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            model_revision="v2.0.4",
            max_end_silence_time=240,  # 最大静默时间
            speech_noise_thres=0.8,  # 语音噪声阈值
            disable_update=True,  # 禁用更新
            disable_pbar=True,  # 禁用进度条
            device="cuda",  # 使用GPU
        )
        # 加载Sense模型
        self.sense_model = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cuda",  # 使用GPU
            disable_update=True,  # 禁用更新
            disable_pbar=True,  # 禁用进度条
        )

        # 加载SV模型
        self.sv_model = Model.from_pretrained("iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common", device="cuda")

# 定义AsrWorker类，用于处理ASR（自动语音识别）任务
class AsrWorker:
    def __init__(self, session_id, model_manager):
        self.session_id = session_id  # 会话ID
        self.model_manager = model_manager  # 模型管理器
        self.chunk_size_ms = 240  # VAD处理时长为240ms
        self.chunk_size = int(SAMPLE_RATE / 1000 * self.chunk_size_ms)  # 每个音频块的大小
        self.fast_reply_silence_duration = 240  # 快速回复静默时长
        self.reply_silence_duration = 960  # 回复静默时长
        self.truncate_silence_duration = 1440  # 截断静默时长
        self.max_audio_duration = 120000  # 最大音频时长（120秒）
        self.mode = 'auto'  # 默认模式为自动应答
        self.reset()  # 初始化工作

    def reset(self):  # 重置工作
        self.audio_buffer = np.array([], dtype=np.float32)  # 初始化音频缓冲区
        self.audio_process_last_pos_ms = 0  # 初始化音频处理的最后位置
        self.vad_cache = {}  # 初始化VAD缓存
        self.vad_last_pos_ms = -1  # 初始化VAD最后位置
        self.vad_cached_segments = []  # 初始化VAD缓存段
        self.fast_reply_checked = False  # 是否检查快速回复
        self.listening = False  # 是否正在监听
        self.content = ''  # 初始化内容

    def truncate(self):  # 截断音频
        if self.audio_process_last_pos_ms < self.truncate_silence_duration:  # 如果音频处理位置小于截断时长，不做处理
            return
        self.audio_buffer = self.audio_buffer[-self.chunk_size_ms * 16:]  # 保留最后的音频块
        self.audio_process_last_pos_ms = 0  # 重置音频处理位置
        self.vad_cache = {}  # 清空VAD缓存

    def get_unprocessed_duration(self):  # 获取未处理的音频时长
        return self.audio_buffer.shape[0] / 16 - self.audio_process_last_pos_ms  # 计算未处理时长

    def get_silence_duration(self):  # 获取静默时长
        if self.vad_last_pos_ms == -1:
            return 0
        return self.audio_buffer.shape[0] / 16 - self.vad_last_pos_ms  # 计算静默时长

    def is_question(self):  # 检测是否为问题
        # TODO: 使用模型检测问题
        match_tokens = ['吗', '嘛', '么', '呢', '吧', '啦', '？', '?', '拜拜', '再见', '晚安', '退下']  # 问题标识符
        last_part = self.content[-3:]  # 获取内容的最后三部分
        for token in match_tokens:
            if token in last_part:  # 如果匹配到问题标识符，则认为是问题
                return True
        return False

    def on_audio_frame(self, frame):  # 处理音频帧
        frame_fp32 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768  # 将音频数据转换为浮点数
        self.audio_buffer = np.concatenate([self.audio_buffer, frame_fp32])  # 将音频帧加入缓冲区
        current_duration = self.audio_buffer.shape[0] / 16  # 计算当前音频时长

        if not self.listening:  # 如果没有在监听，则返回
            return

        if self.mode == 'manual':  # 如果是手动应答模式
            if current_duration >= self.max_audio_duration:  # 如果音频超过最大时长，则停止
                self.stop()
            return

        if self.mode == 'realtime':  # 实时应答模式（尚未实现）
            print('realtime mode not implemented')
            return

        if self.get_unprocessed_duration() < self.chunk_size_ms:  # 如果未处理音频时长小于每个块的大小，则返回
            return
        self.generate_vad_segments()  # 生成VAD段

        if len(self.vad_cached_segments) == 0:  # 如果VAD段为空，则截断音频
            self.truncate()
            return

        if self.vad_last_pos_ms == -1:  # 如果VAD检测仍在进行中，则返回
            return

        silence_duration = self.get_silence_duration()  # 获取静默时长

        if not self.fast_reply_checked and silence_duration >= self.fast_reply_silence_duration:  # 如果静默时长超过快速回复静默时长
            start_time = time.time()
            self.fast_reply_checked = True
            self.content = self.generate_text()  # 生成文本
            if self.is_question():  # 如果是问题，则快速回复
                logging.info(f'Fast reply detected: {self.content} (time: {time.time() - start_time:.3f}s)')
                self.reply()
            return

        if silence_duration >= self.reply_silence_duration:  # 如果静默时长超过回复静默时长，则回复
            start_time = time.time()
            self.content = self.generate_text()  # 生成文本
            logging.info(f'Silence detected: {self.content} (time: {time.time() - start_time:.3f}s)')
            self.reply()
            return

        if current_duration >= self.max_audio_duration:  # 如果音频时长超过最大值，则回复
            start_time = time.time()
            self.content = self.generate_text()  # 生成文本
            logging.info(f'Max audio duration reached: {self.content} (time: {time.time() - start_time:.3f}s)')
            self.reply()
            return

    def generate_vad_segments(self):  # 生成VAD段
        beg = self.audio_process_last_pos_ms * 16  # 计算起始位置
        end = beg + self.chunk_size  # 计算结束位置
        chunk = self.audio_buffer[beg:end]  # 获取当前音频块
        self.audio_process_last_pos_ms += self.chunk_size_ms  # 更新音频处理位置

        result = self.model_manager.vad_model.generate(input=chunk, cache=self.vad_cache, chunk_size=self.chunk_size_ms)  # 调用VAD模型进行处理
        if len(result[0]['value']) > 0:  # 如果有VAD检测结果
            self.vad_cached_segments.extend(result[0]['value'])  # 将检测结果加入缓存
            self.vad_last_pos_ms = self.vad_cached_segments[-1][1]  # 更新VAD最后位置
            if self.vad_last_pos_ms != -1:
                self.fast_reply_checked = False

    def generate_text(self):  # 生成文本
        result = self.model_manager.sense_model.generate(input=self.audio_buffer, cache={}, language='zh', use_itn=True)  # 调用Sense模型生成文本
        return rich_transcription_postprocess(result[0]['text'])  # 后处理并返回文本

    def generate_embedding(self):  # 生成嵌入向量
        if self.audio_buffer.shape[0] == 0:  # 如果没有音频输入，返回空列表
            return []
        result = self.model_manager.sv_model(self.audio_buffer[-SAMPLE_RATE * 10:])  # 处理最后10秒音频
        return result.tolist()  # 返回嵌入向量列表

    def reply(self):  # 回复
        if self.content == '。':  # 如果生成的内容为空，忽略
            logging.info(f'Ignore empty content {self.content}')
            self.reset()  # 重置
            self.start(self.mode)  # 启动当前模式
            return

        message = {
            'type': 'chat',
            'session_id': self.session_id,
            'content': self.content,
            'embedding': self.generate_embedding(),  # 嵌入向量，用来做声纹识别
        }
        print("Replying:", message)
        # TODO: 接入大模型
        self.reset()  # 重置

    def detect(self, words):  # 直接使用文字内容进行识别
        self.content = words  # 直接设置内容
        self.reply()  # 回复

    def start(self, mode):  # 启动工作
        self.reset()  # 重置
        self.mode = mode  # 设置模式
        self.listening = True  # 开始监听

    def stop(self):  # 停止工作
        self.listening = False  # 停止监听
        if self.audio_buffer.shape[0] > 0:  # 如果有音频缓冲区内容，则生成文本并回复
            self.content = self.generate_text()
            self.reply()

# 定义麦克风音频流
class MicrophoneAudioStream:
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None

    def start_stream(self):
        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,  # 16-bit PCM
            channels=1,  # 单声道
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def get_audio_frame(self):
        if self.stream is not None:
            return self.stream.read(self.chunk_size)
        return None

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

# 用于音频流处理的线程函数
def audio_stream_thread(microphone, asr_worker):
    while True:
        frame = microphone.get_audio_frame()  # 获取麦克风音频帧
        if frame:
            asr_worker.on_audio_frame(frame)  # 传递音频帧给 AsrWorker 进行处理
        time.sleep(0.01)  # 适当的延迟，避免过度占用 CPU

    # 用于启动ASR识别的线程函数
def start_asr_thread(asr_worker):
    asr_worker.start(mode='auto')  # 启动自动模式

# 主程序逻辑，异步处理用户输入
def test_with_microphone():
    # 初始化模型管理器
    model_manager = ModelManager()
    model_manager.load_models()

    # 初始化AsrWorker
    session_id = str(uuid.uuid4())  # 生成唯一会话ID
    asr_worker = AsrWorker(session_id, model_manager)

    # 初始化麦克风音频流
    microphone = MicrophoneAudioStream()
    microphone.start_stream()

    # 启动音频流处理线程
    audio_thread = threading.Thread(target=audio_stream_thread, args=(microphone, asr_worker))
    audio_thread.daemon = True  # 将线程设置为守护线程，主程序退出时自动结束
    audio_thread.start()

    while True:  # 循环等待每次按回车后启动识别
        input()
        print("Starting ASR...")
        asr_thread = threading.Thread(target=start_asr_thread, args=(asr_worker,))
        asr_thread.start()


if __name__ == '__main__':
    test_with_microphone()
