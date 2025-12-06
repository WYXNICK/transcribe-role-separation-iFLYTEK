#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音角色分离和转写的级联管线
VAD -> Speaker Diarization -> ASR
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
import logging
from dataclasses import dataclass


@dataclass
class SpeechSegment:
    """语音段数据结构"""
    start: float
    end: float
    speaker: str
    audio_path: Optional[str] = None
    text: Optional[str] = None
    confidence: float = 1.0


class VADProcessor:
    """语音活动检测处理器"""
    
    def __init__(self, model_name: str = "pyannote/voice-activity-detection"):
        """
        初始化VAD处理器
        
        Args:
            model_name: VAD模型名称
        """
        self.model_name = model_name
        self.pipeline = None
        self.audio_loader = None
        
        try:
            from pyannote.audio import Pipeline, Audio
            
            # 尝试从本地或Hugging Face加载模型
            print(f"正在加载VAD模型: {model_name}")
            self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=os.environ.get("HF_TOKEN"))
            self.audio_loader = Audio()
            
            # 设置设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print(f"将VAD模型移至GPU: {self.device}")
                self.pipeline = self.pipeline.to(self.device)
            
            print("VAD模型加载成功")
            
        except Exception as e:
            print(f"VAD模型加载失败: {e}")
            print("将使用fallback能量检测方法")
            self.pipeline = None
            self.audio_loader = None
    
    def detect_speech(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        检测语音活动段
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[Tuple[float, float]]: 语音段的(开始时间, 结束时间)列表
        """
        if self.pipeline is None:
            # 使用简单的能量检测作为fallback
            return self._energy_based_vad(audio_path)
        
        try:
            print(f"使用pyannote VAD处理: {audio_path}")
            
            # 使用官方API方式进行VAD检测
            vad_result = self.pipeline(audio_path)
            speech_segments = []
            
            # 提取语音段
            for segment in vad_result.get_timeline():
                speech_segments.append((segment.start, segment.end))

            # 合并与最短时长约束
            merged = self._merge_vad_segments(speech_segments,
                                              max_gap=self.config.get('max_vad_gap', 0.3) if hasattr(self, 'config') else 0.3,
                                              min_duration=self.config.get('min_vad_duration', 0.6) if hasattr(self, 'config') else 0.6)
            print(f"VAD检测完成，原始 {len(speech_segments)} 段，合并后 {len(merged)} 段")
            return merged
            
        except Exception as e:
            print(f"pyannote VAD检测失败: {e}")
            print("降级使用能量检测方法")
            return self._energy_based_vad(audio_path)

    def _merge_vad_segments(self, segments: List[Tuple[float, float]], max_gap: float = 0.3,
                            min_duration: float = 0.6) -> List[Tuple[float, float]]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        cs, ce = segments[0]
        for s, e in segments[1:]:
            if s - ce <= max_gap:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))

        refined: List[Tuple[float, float]] = []
        for s, e in merged:
            if e - s >= min_duration or not refined:
                refined.append((s, e))
            else:
                ps, pe = refined[-1]
                if s - pe <= max_gap:
                    refined[-1] = (ps, max(pe, e))
                else:
                    refined.append((s, e))
        return refined
    
    def detect_speech_segment(self, audio_path: str, start: float, end: float) -> List[Tuple[float, float]]:
        """
        检测指定时间段内的语音活动
        
        Args:
            audio_path: 音频文件路径
            start: 开始时间(秒)
            end: 结束时间(秒)
            
        Returns:
            List[Tuple[float, float]]: 语音段的(开始时间, 结束时间)列表
        """
        if self.pipeline is None or self.audio_loader is None:
            return self._energy_based_vad(audio_path)
        
        try:
            from pyannote.core import Segment
            
            # 创建时间段
            excerpt = Segment(start=start, end=end)
            
            # 加载指定时间段的音频
            waveform, sample_rate = self.audio_loader.crop(audio_path, excerpt)
            
            # 进行VAD检测
            vad_result = self.pipeline({
                "waveform": waveform, 
                "sample_rate": sample_rate
            })
            
            speech_segments = []
            for segment in vad_result.get_timeline():
                # 调整时间偏移
                abs_start = segment.start + start
                abs_end = segment.end + start
                speech_segments.append((abs_start, abs_end))
            
            return speech_segments
            
        except Exception as e:
            print(f"VAD段检测失败: {e}")
            return self._energy_based_vad(audio_path)
    
    def _energy_based_vad(self, audio_path: str, 
                         frame_length: int = 2048,
                         hop_length: int = 512) -> List[Tuple[float, float]]:
        """基于能量的简单VAD"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 计算短时能量
            frame_energy = librosa.feature.rms(y=y, 
                                             frame_length=frame_length,
                                             hop_length=hop_length)[0]
            
            # 动态阈值
            energy_threshold = np.percentile(frame_energy, 30)
            
            # 检测语音段
            speech_frames = frame_energy > energy_threshold
            frame_times = librosa.frames_to_time(np.arange(len(speech_frames)),
                                               sr=sr, hop_length=hop_length)
            
            # 合并连续的语音帧
            segments = []
            in_speech = False
            start_time = 0
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    start_time = frame_times[i]
                    in_speech = True
                elif not is_speech and in_speech:
                    end_time = frame_times[i]
                    if end_time - start_time > 0.5:  # 最小语音段长度
                        segments.append((start_time, end_time))
                    in_speech = False
            
            # 处理最后一个段
            if in_speech:
                segments.append((start_time, frame_times[-1]))
            
            return segments
            
        except Exception as e:
            print(f"基于能量的VAD失败: {e}")
            return []


class SpeakerDiarization:
    """说话人分离处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化说话人分离器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.clustering_threshold = self.config.get('clustering_threshold', 0.5)
        self.min_speakers = self.config.get('min_speakers', 2)
        self.max_speakers = self.config.get('max_speakers', 10)
        
        # 尝试加载3D-Speaker-Toolkit
        self.speaker_model = self._load_speaker_model()
    
    def _load_speaker_model(self):
        """加载说话人识别模型"""
        try:
            # 尝试加载 3D-Speaker
            print("正在尝试加载 3D-Speaker...")
            local_3d_path = os.path.abspath('3D-Speaker-main')
            
            if not os.path.isdir(local_3d_path):
                print(f"3D-Speaker 目录不存在: {local_3d_path}")
                return None
                
            if local_3d_path not in sys.path:
                sys.path.insert(0, local_3d_path)
            
            # 尝试直接导入，捕获具体错误
            try:
                print("导入 Diarization3Dspeaker...")
                from speakerlab.bin.infer_diarization import Diarization3Dspeaker
                
                print("初始化 3D-Speaker 模型...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                diar = Diarization3Dspeaker(
                    device=device, 
                    include_overlap=False, 
                    hf_access_token=os.environ.get("HF_TOKEN")
                )
                print("3D-Speaker 模型加载成功")
                return diar
                
            except ImportError as e:
                print(f"3D-Speaker 导入失败 (可能是依赖问题): {e}")
                return None
            except Exception as e:
                print(f"3D-Speaker 初始化失败: {e}")
                return None

            
        except Exception as e:
            print(f"3D-Speaker 加载过程出错: {e}")
            return None
    
    def diarize(self, audio_path: str, 
                speech_segments: List[Tuple[float, float]] = None) -> List[SpeechSegment]:
        """
        进行说话人分离
        
        Args:
            audio_path: 音频文件路径
            speech_segments: VAD检测到的语音段
            
        Returns:
            List[SpeechSegment]: 说话人分离结果
        """
        if self.speaker_model is None:
            return self._fallback_diarization(audio_path, speech_segments)
        
        try:
            # 情况1：本地 3D-Speaker 的 Diarization3Dspeaker
            if self.speaker_model.__class__.__name__ == 'Diarization3Dspeaker':
                diar_out = self.speaker_model(audio_path)
                segments: List[SpeechSegment] = []
                for st, ed, spk in diar_out:
                    segments.append(SpeechSegment(
                        start=float(st),
                        end=float(ed),
                        speaker=f"spk{int(spk)}",
                    ))
                return self._merge_same_speaker_segments(
                    segments,
                    max_gap=self.config.get('max_spk_gap', 0.3),
                    max_window=self.config.get('max_spk_window', 8.0),
                    min_duration=self.config.get('min_spk_duration', 0.6),
                )

            # 情况2：ModelScope pipeline
            result = self.speaker_model(audio_path)
            segments: List[SpeechSegment] = []
            for segment in result.get('segments', []):
                segments.append(SpeechSegment(
                    start=segment['start'],
                    end=segment['end'],
                    speaker=f"spk{segment['speaker']}",
                    confidence=segment.get('confidence', 1.0)
                ))
            return self._merge_same_speaker_segments(
                segments,
                max_gap=self.config.get('max_spk_gap', 0.3),
                max_window=self.config.get('max_spk_window', 8.0),
                min_duration=self.config.get('min_spk_duration', 0.6),
            )
            
        except Exception as e:
            print(f"说话人分离失败: {e}")
            return self._fallback_diarization(audio_path, speech_segments)
    
    def _fallback_diarization(self, audio_path: str, 
                            speech_segments: List[Tuple[float, float]]) -> List[SpeechSegment]:
        """简单的fallback说话人分离"""
        segments = []
        
        if speech_segments is None:
            # 简单分割
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                duration = len(y) / sr
                
                # 简单的基于时间的分割，假设2个说话人轮流
                num_segments = max(4, int(duration / 5))  # 每5秒一个段
                segment_duration = duration / num_segments
                
                for i in range(num_segments):
                    start = i * segment_duration
                    end = min((i + 1) * segment_duration, duration)
                    speaker = f"spk{i % 2 + 1}"  # 简单轮流
                    
                    segments.append(SpeechSegment(
                        start=start,
                        end=end,
                        speaker=speaker
                    ))
                    
            except Exception as e:
                print(f"Fallback分离失败: {e}")
        else:
            # 基于VAD段进行简单分配
            for i, (start, end) in enumerate(speech_segments):
                speaker = f"spk{i % 2 + 1}"  # 简单轮流
                segments.append(SpeechSegment(
                    start=start,
                    end=end,
                    speaker=speaker
                ))
        
        return self._merge_same_speaker_segments(
            segments,
            max_gap=self.config.get('max_spk_gap', 0.3),
            max_window=self.config.get('max_spk_window', 8.0),
            min_duration=self.config.get('min_spk_duration', 0.6),
        )

    def _merge_same_speaker_segments(self, segments: List[SpeechSegment], max_gap: float = 0.3,
                                     max_window: float = 8.0, min_duration: float = 0.6) -> List[SpeechSegment]:
        if not segments:
            return []
        segs = sorted(segments, key=lambda s: s.start)
        out: List[SpeechSegment] = []
        cur: Optional[SpeechSegment] = None
        for s in segs:
            if cur is None:
                cur = SpeechSegment(start=s.start, end=s.end, speaker=s.speaker)
                continue
            if s.speaker == cur.speaker and s.start - cur.end <= max_gap and (s.end - cur.start) <= max_window:
                cur.end = max(cur.end, s.end)
            else:
                if cur.end - cur.start >= min_duration:
                    out.append(cur)
                else:
                    if out and out[-1].speaker == cur.speaker and cur.start - out[-1].end <= max_gap:
                        out[-1].end = max(out[-1].end, cur.end)
                    else:
                        out.append(cur)
                cur = SpeechSegment(start=s.start, end=s.end, speaker=s.speaker)
        if cur is not None:
            if cur.end - cur.start >= min_duration:
                out.append(cur)
            else:
                if out and out[-1].speaker == cur.speaker and cur.start - out[-1].end <= max_gap:
                    out[-1].end = max(out[-1].end, cur.end)
                else:
                    out.append(cur)
        return sorted(out, key=lambda s: s.start)


class ASRProcessor:
    """FireRedASR 自动语音识别处理器"""
    
    def __init__(self, model_name: str = "FireRedASR-AED-L", config: Dict[str, Any] = None):
        """
        初始化ASR处理器
        
        Args:
            model_name: ASR模型名称（支持 FireRedASR-AED-L 和 FireRedASR-LLM-L）
            config: 模型配置
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = self._load_firered_model()
        
        # 解码参数
        self.beam_size = self.config.get('beam_size', 5)
        self.length_penalty = self.config.get('length_penalty', 1.0)
        self.repetition_penalty = self.config.get('repetition_penalty', 1.1)
        self.temperature = self.config.get('temperature', 0.3)
    
    def _load_firered_model(self):
        """加载 FireRedASR 模型"""
        try:
            print(f"正在加载 FireRedASR 模型: {self.model_name}")
            
            # 设置 FireRedASR 路径
            base_dir = os.path.abspath("FireRedASR-main")
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            
            # 导入 FireRedASR
            from fireredasr.models.fireredasr import FireRedAsr
            
            # 确定模型变体
            if "AED" in self.model_name.upper():
                variant = "aed"
                model_folder = "FireRedASR-AED-L"
            elif "LLM" in self.model_name.upper():
                variant = "llm"
                model_folder = "FireRedASR-LLM-L"
            else:
                variant = "aed"  # 默认使用 AED 变体
                model_folder = "FireRedASR-AED-L"
            
            # 构建模型路径
            model_dir = os.path.join(base_dir, "pretrained_models", model_folder)
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"FireRedASR 模型目录不存在: {model_dir}")
            
            # 加载模型
            fr_model = FireRedAsr.from_pretrained(variant, model_dir)
            
            print(f"FireRedASR 模型加载成功: {variant} 变体")
            return {
                "type": "firered",
                "variant": variant,
                "model": fr_model
            }
            
        except Exception as e:
            print(f"FireRedASR 模型加载失败: {e}")
            raise RuntimeError(f"无法加载 FireRedASR 模型: {e}")
    
    def transcribe_segment(self, audio_path: str) -> str:
        """
        使用 FireRedASR 转写单个音频段
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            str: 转写文本
        """
        if self.model is None:
            return ""
        
        try:
            if not os.path.exists(audio_path):
                print(f"警告: 音频文件不存在 {audio_path}")
                return ""
            
            variant = self.model.get("variant", "aed")
            use_gpu = 1 if torch.cuda.is_available() else 0
            
            batch_uttid = [Path(audio_path).stem]
            batch_wav_path = [audio_path]
            
            # 根据变体设置解码参数
            if variant == "aed":
                decode_args = {
                    "use_gpu": use_gpu,
                    "beam_size": int(self.beam_size),
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.25,
                    "aed_length_penalty": float(self.length_penalty),
                    "eos_penalty": 1.0,
                }
            else:  # llm
                decode_args = {
                    "use_gpu": use_gpu,
                    "beam_size": int(self.beam_size),
                    "decode_max_len": 0,
                    "decode_min_len": 0,
                    "repetition_penalty": float(self.repetition_penalty),
                    "llm_length_penalty": float(self.length_penalty),
                    "temperature": float(self.temperature),
                }
            
            # 执行转写
            results = self.model["model"].transcribe(batch_uttid, batch_wav_path, decode_args)
            
            if results and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    text = first_result.get("text", "")
                else:
                    text = str(first_result)
                
                return self._post_process_text(text)
            
            return ""
            
        except Exception as e:
            print(f"FireRedASR 转写失败 {audio_path}: {e}")
            return ""
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """使用 FireRedASR 批量转写"""
        if self.model is None:
            return [""] * len(audio_paths)
        
        variant = self.model.get("variant", "aed")
        
        # AED 变体支持真正的批量处理，LLM 变体建议逐个处理避免重复问题
        if variant == "aed":
            return self._batch_transcribe_aed(audio_paths)
        else:
            return self._sequential_transcribe_llm(audio_paths)
    
    def _batch_transcribe_aed(self, audio_paths: List[str]) -> List[str]:
        """AED 变体批量转写（支持内存优化）"""
        try:
            # 过滤有效的音频路径
            valid_paths = []
            valid_indices = []
            
            for i, path in enumerate(audio_paths):
                if path and os.path.exists(path):
                    valid_paths.append(path)
                    valid_indices.append(i)
            
            if not valid_paths:
                return [""] * len(audio_paths)
            
            # 内存优化：大批量时分块处理
            max_batch_size = 8 if torch.cuda.is_available() else 4
            if len(valid_paths) > max_batch_size:
                print(f"大批量 ({len(valid_paths)}) 分块处理，避免 GPU 内存溢出")
                return self._chunked_batch_transcribe(audio_paths, max_batch_size)
            
            # 准备批量输入
            uttids = [Path(p).stem for p in valid_paths]
            decode_args = {
                "use_gpu": 1 if torch.cuda.is_available() else 0,
                "beam_size": int(self.beam_size),
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": float(self.length_penalty),
                "eos_penalty": 1.0,
            }
            
            # 执行批量转写
            results = self.model["model"].transcribe(uttids, valid_paths, decode_args)
            
            # 构建完整的结果列表
            full_results = [""] * len(audio_paths)
            
            for i, result in enumerate(results):
                if i < len(valid_indices):
                    original_idx = valid_indices[i]
                    if isinstance(result, dict):
                        text = result.get("text", "")
                    else:
                        text = str(result)
                    full_results[original_idx] = self._post_process_text(text)
            
            return full_results
            
        except Exception as e:
            print(f"FireRedASR AED 批量转写失败，改为逐个处理: {e}")
            return [self.transcribe_segment(p) for p in audio_paths]
    
    def _chunked_batch_transcribe(self, audio_paths: List[str], chunk_size: int) -> List[str]:
        """分块批量转写"""
        all_results = [""] * len(audio_paths)
        
        for i in range(0, len(audio_paths), chunk_size):
            chunk = audio_paths[i:i + chunk_size]
            chunk_results = self._batch_transcribe_aed(chunk)
            
            for j, result in enumerate(chunk_results):
                if i + j < len(all_results):
                    all_results[i + j] = result
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results
    
    def _sequential_transcribe_llm(self, audio_paths: List[str]) -> List[str]:
        """LLM 变体逐个转写"""
        return [self.transcribe_segment(p) for p in tqdm(audio_paths, desc="FireRedASR-LLM 转写")]

    def _post_process_text(self, text: str) -> str:
        import re
        if not text:
            return ""
        # 去除中文字符间多余空格
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        text = text.strip()
        # 轻度重复字清洗
        if len(text) <= 6:
            text = re.sub(r'([\u4e00-\u9fff])\1{1,}', r'\1', text)
        # 语气词弱化
        text = re.sub(r'(啊|哦|嗯|呃|唉)\1{1,}', r'\1', text)
        return text


class RoleSeparationPipeline:
    """语音角色分离和转写管线"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化管线
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 初始化各个组件
        self.vad = VADProcessor()
        self.diarization = SpeakerDiarization(self.config.get('diarization', {}))
        asr_config = self.config.get('asr', {})
        model_name = asr_config.get('model_name', 'FireRedASR-AED-L')
        self.asr = ASRProcessor(model_name=model_name, config=asr_config)
        
        # 临时目录
        self.temp_dir = self.config.get('temp_dir', 'temp_segments')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def process_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        处理单个音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[Dict]: 处理结果
        """
        print(f"处理音频文件: {audio_path}")
        
        # 步骤1: VAD检测
        print("步骤1: 语音活动检测...")
        speech_segments = self.vad.detect_speech(audio_path)
        print(f"检测到 {len(speech_segments)} 个语音段")
        
        # 步骤2: 说话人分离
        print("步骤2: 说话人分离...")
        diarization_result = self.diarization.diarize(audio_path, speech_segments)
        print(f"分离出 {len(diarization_result)} 个说话人段")
        
        # 步骤3: 音频切分
        print("步骤3: 音频切分...")
        segment_paths = self._cut_audio_segments(audio_path, diarization_result)
        
        # 步骤4: ASR转写
        print("步骤4: 语音识别...")
        transcriptions = self.asr.transcribe_batch(segment_paths)
        
        # 步骤5: 整合结果
        results = []
        for i, (segment, transcription) in enumerate(zip(diarization_result, transcriptions)):
            if transcription.strip():  # 只保留非空转写
                results.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': segment.speaker,
                    'text': transcription.strip(),
                    'confidence': segment.confidence
                })
        
        # 清理临时文件
        self._cleanup_temp_files(segment_paths)
        
        return results
    
    def _cut_audio_segments(self, audio_path: str, 
                          segments: List[SpeechSegment]) -> List[str]:
        """切分音频段"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            segment_paths = []
            
            file_stem = Path(audio_path).stem
            
            for i, segment in enumerate(segments):
                start_sample = int(segment.start * sr)
                end_sample = int(segment.end * sr)
                
                # 确保索引有效
                start_sample = max(0, start_sample)
                end_sample = min(len(y), end_sample)
                
                if end_sample > start_sample:
                    audio_segment = y[start_sample:end_sample]
                    
                    # 生成临时文件路径
                    segment_filename = f"{file_stem}_seg_{i:04d}_{segment.speaker}.wav"
                    segment_path = os.path.join(self.temp_dir, segment_filename)
                    
                    # 保存音频段
                    sf.write(segment_path, audio_segment, sr)
                    segment_paths.append(segment_path)
                else:
                    segment_paths.append("")  # 空路径表示无效段
            
            return segment_paths
            
        except Exception as e:
            print(f"音频切分失败: {e}")
            return [""] * len(segments)
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """清理临时文件"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    def process_test_data(self, test_dir: str, output_dir: str) -> Dict[str, str]:
        """
        处理测试数据
        
        Args:
            test_dir: 测试数据目录
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 文件名到结果文本的映射
        """
        os.makedirs(output_dir, exist_ok=True)
        
        test_files = list(Path(test_dir).glob("*.wav"))
        results = {}
        
        for audio_file in tqdm(test_files, desc="处理测试文件"):
            try:
                # 处理音频文件
                segments = self.process_audio(str(audio_file))
                
                # 格式化输出
                output_lines = [f"{audio_file.name}"]
                for segment in segments:
                    output_lines.append(f"{segment['speaker']}: {segment['text']}")
                output_lines.append(".")  # 结束标记
                
                result_text = "\n".join(output_lines)
                results[audio_file.name] = result_text
                
                # 保存单个结果文件
                output_file = os.path.join(output_dir, f"{audio_file.stem}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result_text)
                    
            except Exception as e:
                print(f"处理文件失败 {audio_file}: {e}")
                results[audio_file.name] = f"{audio_file.name}\n."
        
        return results


def main():
    """测试管线"""
    config = {
        'diarization': {
            'clustering_threshold': 0.5,
            'min_speakers': 2,
            'max_speakers': 8
        },
        'asr': {
            'model_name': 'FireRedASR-AED-L',  # 可选: FireRedASR-AED-L 或 FireRedASR-LLM-L
            'beam_size': 5,
            'length_penalty': 1.0,
            'repetition_penalty': 1.1,
            'temperature': 0.3
        },
        'temp_dir': 'temp_segments'
    }
    
    pipeline = RoleSeparationPipeline(config)
    
    # 测试单个文件
    test_audio = "test_data/T1.wav"
    if os.path.exists(test_audio):
        results = pipeline.process_audio(test_audio)
        print("处理结果:")
        for result in results:
            print(f"{result['speaker']}: {result['text']}")


if __name__ == "__main__":
    main()
