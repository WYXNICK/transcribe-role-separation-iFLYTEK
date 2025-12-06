#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
处理TextGrid文件解析、音频切分、数据集构建
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import librosa
import soundfile as sf
from pydub import AudioSegment
import textgrid
from tqdm import tqdm


class TextGridParser:
    """TextGrid文件解析器"""
    
    def __init__(self):
        pass
    
    def parse_textgrid(self, textgrid_path: str) -> List[Dict]:
        """
        解析单个TextGrid文件
        
        Args:
            textgrid_path: TextGrid文件路径
            
        Returns:
            List[Dict]: 包含说话人信息的列表
            格式: [{'start': float, 'end': float, 'speaker': str, 'text': str}]
        """
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            segments = []
            
            for tier in tg.tiers:
                # 检查是否是说话人层或文本层
                tier_name = tier.name.lower()
                
                for interval in tier.intervals:
                    if interval.mark and interval.mark.strip():
                        # 清理文本
                        text = interval.mark.strip()
                        
                        # 提取说话人信息
                        speaker = self._extract_speaker_from_text(text, tier_name)
                        
                        if speaker and text:
                            segments.append({
                                'start': float(interval.minTime),
                                'end': float(interval.maxTime),
                                'speaker': speaker,
                                'text': text,
                                'duration': float(interval.maxTime - interval.minTime)
                            })
            
            return sorted(segments, key=lambda x: x['start'])
            
        except Exception as e:
            print(f"解析TextGrid文件失败 {textgrid_path}: {e}")
            return []
    
    def _extract_speaker_from_text(self, text: str, tier_name: str) -> Optional[str]:
        """从文本或层名中提取说话人信息"""
        # 如果文本包含说话人标签
        if ':' in text:
            parts = text.split(':', 1)
            speaker_part = parts[0].strip()
            if speaker_part.startswith('spk') or 'speaker' in speaker_part.lower():
                return speaker_part
        
        # 从层名提取 - 只要是spk开头的层名都认为是说话人
        if tier_name.startswith('spk') or 'speaker' in tier_name.lower():
            return tier_name
            
        # 默认返回
        return 'spk_unknown'
    
    def parse_all_textgrids(self, eval_dir: str) -> pd.DataFrame:
        """
        解析eval目录下所有TextGrid文件
        
        Args:
            eval_dir: eval目录路径
            
        Returns:
            pd.DataFrame: 包含所有音频段信息的DataFrame
        """
        textgrid_files = list(Path(eval_dir).glob("*.TextGrid"))
        all_segments = []
        
        for tg_file in tqdm(textgrid_files, desc="解析TextGrid文件"):
            audio_file = tg_file.with_suffix('.wav')
            if audio_file.exists():
                segments = self.parse_textgrid(str(tg_file))
                # 为每个段落添加音频文件路径和文件ID
                for seg in segments:
                    seg['audio_file'] = str(audio_file)
                    seg['file_id'] = tg_file.stem
                all_segments.extend(segments)
        return pd.DataFrame(all_segments)


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            return None, None
    
    def segment_audio(self, audio_path: str, segments: List[Dict], 
                     output_dir: str, min_duration: float = 0.5,
                     max_duration: float = 30.0) -> List[Dict]:
        """
        根据时间段切分音频
        
        Args:
            audio_path: 原始音频路径
            segments: 时间段列表
            output_dir: 输出目录
            min_duration: 最小时长(秒)
            max_duration: 最大时长(秒)
            
        Returns:
            List[Dict]: 切分后的音频段信息
        """
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        segmented_info = []
        file_id = Path(audio_path).stem
        
        for i, seg in enumerate(segments):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            duration = seg['end'] - seg['start']
            
            # 过滤太短或太长的片段
            if duration < min_duration or duration > max_duration:
                continue
            
            audio_segment = audio[start_sample:end_sample]
            
            # 生成输出文件名
            output_filename = f"{file_id}_seg_{i:04d}_{seg['speaker']}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存音频段
            sf.write(output_path, audio_segment, sr)
            
            segmented_info.append({
                'segment_path': output_path,
                'original_path': audio_path,
                'start': seg['start'],
                'end': seg['end'],
                'duration': duration,
                'speaker': seg['speaker'],
                'text': seg['text'],
                'file_id': file_id,
                'segment_id': i
            })
        
        return segmented_info


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, eval_dir: str, output_dir: str):
        self.eval_dir = eval_dir
        self.output_dir = output_dir
        self.parser = TextGridParser()
        self.audio_processor = AudioProcessor()
        
    def build_dataset(self, val_split: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建训练和验证数据集
        
        Args:
            val_split: 验证集比例
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 验证集)
        """
        print("开始构建数据集...")
        
        # 解析所有TextGrid文件
        df = self.parser.parse_all_textgrids(self.eval_dir)
        print(f"解析得到 {len(df)} 个音频段")
        
        # 按文件分组，确保同一文件的段都在同一数据集中
        file_groups = df.groupby('file_id')
        file_ids = list(file_groups.groups.keys())
        
        # 随机划分文件
        np.random.seed(42)
        np.random.shuffle(file_ids)
        val_size = int(len(file_ids) * val_split)
        val_files = file_ids[:val_size]
        train_files = file_ids[val_size:]
        
        train_df = df[df['file_id'].isin(train_files)].copy()
        val_df = df[df['file_id'].isin(val_files)].copy()
        
        print(f"训练集: {len(train_df)} 段 ({len(train_files)} 文件)")
        print(f"验证集: {len(val_df)} 段 ({len(val_files)} 文件)")
        
        # 保存数据集信息
        os.makedirs(self.output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(self.output_dir, 'train_metadata.csv'), index=False)
        val_df.to_csv(os.path.join(self.output_dir, 'val_metadata.csv'), index=False)
        
        return train_df, val_df
    
    def prepare_segmented_audio(self, df: pd.DataFrame, output_subdir: str) -> pd.DataFrame:
        """
        为数据集准备切分后的音频文件
        
        Args:
            df: 数据集DataFrame
            output_subdir: 输出子目录名
            
        Returns:
            pd.DataFrame: 包含切分音频路径的DataFrame
        """
        output_dir = os.path.join(self.output_dir, output_subdir)
        all_segments = []
        
        # 按文件分组处理
        for file_id, group in tqdm(df.groupby('file_id'), desc=f"切分{output_subdir}音频"):
            audio_path = group.iloc[0]['audio_file']
            segments = group.to_dict('records')
            
            segmented_info = self.audio_processor.segment_audio(
                audio_path, segments, output_dir
            )
            all_segments.extend(segmented_info)
        
        segmented_df = pd.DataFrame(all_segments)
        segmented_df.to_csv(os.path.join(self.output_dir, f'{output_subdir}_segments.csv'), index=False)
        
        return segmented_df
    
    def create_speaker_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """创建说话人到ID的映射"""
        speakers = sorted(df['speaker'].unique())
        speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
        
        # 保存映射
        with open(os.path.join(self.output_dir, 'speaker_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(speaker_to_id, f, ensure_ascii=False, indent=2)
        
        return speaker_to_id


def main():
    """主函数 - 数据预处理入口"""
    eval_dir = "eval"
    output_dir = "processed_data"
    
    # 创建数据集构建器
    builder = DatasetBuilder(eval_dir, output_dir)
    
    # 构建数据集
    train_df, val_df = builder.build_dataset(val_split=0.1)
    
    # 准备切分音频
    print("准备训练集音频...")
    train_segments = builder.prepare_segmented_audio(train_df, 'train')
    
    print("准备验证集音频...")
    val_segments = builder.prepare_segmented_audio(val_df, 'val')
    
    # 创建说话人映射
    all_df = pd.concat([train_df, val_df])
    speaker_mapping = builder.create_speaker_mapping(all_df)
    
    print(f"数据预处理完成！")
    print(f"说话人数量: {len(speaker_mapping)}")
    print(f"训练段数: {len(train_segments)}")
    print(f"验证段数: {len(val_segments)}")


if __name__ == "__main__":
    main()
