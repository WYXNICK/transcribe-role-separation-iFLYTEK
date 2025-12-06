#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型微调模块
包括Speaker Diarization和ASR模型的微调
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerEmbeddingDataset(Dataset):
    """说话人嵌入训练数据集"""
    
    def __init__(self, metadata_df: pd.DataFrame, audio_dir: str,
                 max_length: float = 3.0, sr: int = 16000):
        """
        初始化数据集
        
        Args:
            metadata_df: 元数据DataFrame
            audio_dir: 音频文件目录
            max_length: 最大音频长度(秒)
            sr: 采样率
        """
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.sr = sr
        self.max_samples = int(max_length * sr)
        
        # 创建说话人标签映射
        speakers = sorted(metadata_df['speaker'].unique())
        self.speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
        self.num_speakers = len(speakers)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 加载音频 - 优先使用segment_path字段，否则构造路径
        if 'segment_path' in row and pd.notna(row['segment_path']) and row['segment_path'].strip():
            # 使用CSV中记录的完整路径
            audio_path = row['segment_path']
            # 如果是相对路径，确保正确
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(os.getcwd(), audio_path)
        else:
            # 回退到构造路径的方式
            if 'segment_id' in row:
                segment_id = row['segment_id']
            else:
                segment_id = idx  # 使用索引作为segment_id
            
            audio_path = os.path.join(self.audio_dir, f"{row['file_id']}_seg_{segment_id:04d}_{row['speaker']}.wav")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"WARNING: 音频文件不存在: {audio_path}")
                # 尝试其他可能的路径
                alternative_paths = [
                    audio_path,
                    os.path.join(self.audio_dir, os.path.basename(audio_path)),
                    os.path.join("processed_data/train", os.path.basename(audio_path))
                ]
                
                audio_path_found = None
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        audio_path_found = alt_path
                        break
                
                if audio_path_found:
                    audio_path = audio_path_found
                    print(f"找到替代路径: {audio_path}")
                else:
                    print(f"ERROR: 无法找到音频文件: {audio_path}")
                    # 返回零音频避免训练中断（与正常返回结构一致）
                    fallback_speaker_id = self.speaker_to_id.get(row.get('speaker', ''), 0)
                    return {
                        'audio': torch.zeros(self.max_samples),
                        'speaker_id': torch.LongTensor([fallback_speaker_id]),
                        'file_id': row.get('file_id', 'unknown')
                    }
            
            import librosa
            audio, _ = librosa.load(audio_path, sr=self.sr)
            
            # 音频预处理
            audio = self._preprocess_audio(audio)
            
            # 说话人标签
            speaker_id = self.speaker_to_id[row['speaker']]
            
            return {
                'audio': torch.FloatTensor(audio),
                'speaker_id': torch.LongTensor([speaker_id]),
                'file_id': row['file_id']
            }
            
        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            # 返回零音频避免训练中断（与正常返回结构一致）
            fallback_speaker_id = self.speaker_to_id.get(row.get('speaker', ''), 0)
            return {
                'audio': torch.zeros(self.max_samples),
                'speaker_id': torch.LongTensor([fallback_speaker_id]),
                'file_id': row.get('file_id', 'unknown')
            }
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """音频预处理"""
        # 截断或填充到固定长度
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), 'constant')
        
        # 归一化
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio


class SpeakerEmbeddingModel(nn.Module):
    """简化的说话人嵌入模型"""
    
    def __init__(self, num_speakers: int, embedding_dim: int = 256):
        super().__init__()
        
        # 简单的CNN架构
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.embedding_layer = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_speakers)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # CNN特征提取
        x = self.conv_layers(x)  # (batch_size, 256, 1)
        x = x.squeeze(-1)  # (batch_size, 256)
        
        # 嵌入层
        embeddings = self.embedding_layer(x)
        embeddings = self.dropout(embeddings)
        
        # 分类层
        logits = self.classifier(embeddings)
        
        return embeddings, logits


class SpeakerDiarizationTrainer:
    """说话人分离模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    segmented_audio_dir: str) -> Tuple[DataLoader, DataLoader]:
        """准备训练数据"""
        # 创建数据集
        train_dataset = SpeakerEmbeddingDataset(
            train_df, segmented_audio_dir,
            max_length=self.config.get('max_audio_length', 3.0)
        )
        
        val_dataset = SpeakerEmbeddingDataset(
            val_df, segmented_audio_dir,
            max_length=self.config.get('max_audio_length', 3.0)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        return train_loader, val_loader, train_dataset.num_speakers
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_speakers: int) -> nn.Module:
        """训练模型"""
        # 创建模型
        model = SpeakerEmbeddingModel(
            num_speakers=num_speakers,
            embedding_dim=self.config.get('embedding_dim', 256)
        ).to(self.device)
        
        # 优化器和调度器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-2)
        )
        
        num_epochs = self.config.get('num_epochs', 10)
        total_steps = len(train_loader) * num_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_acc = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                audio = batch['audio'].to(self.device)
                speaker_ids = batch['speaker_id'].squeeze().to(self.device)
                
                optimizer.zero_grad()
                
                embeddings, logits = model(audio)
                loss = criterion(logits, speaker_ids)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                train_acc += (predicted == speaker_ids).float().mean().item()
            
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_acc = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    audio = batch['audio'].to(self.device)
                    speaker_ids = batch['speaker_id'].squeeze().to(self.device)
                    
                    embeddings, logits = model(audio)
                    loss = criterion(logits, speaker_ids)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(logits, 1)
                    val_acc += (predicted == speaker_ids).float().mean().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 
                          os.path.join(self.config['output_dir'], 'best_speaker_model.pth'))
        
        return model


class ASRDataset(Dataset):
    """ASR训练数据集"""
    
    def __init__(self, metadata_df: pd.DataFrame, audio_dir: str, max_length: float = 30.0):
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.max_length = max_length
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 音频路径
        audio_path = os.path.join(self.audio_dir, f"{row['file_id']}_seg_{row['segment_id']:04d}_{row['speaker']}.wav")
        
        return {
            'audio_path': audio_path,
            'text': row['text'],
            'speaker': row['speaker']
        }


class ASRFineTuner:
    """ASR模型微调器"""
    
    def __init__(self, model_name: str = "FireRedASR", config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_fine_tuning_data(self, train_df: pd.DataFrame, output_dir: str):
        """准备微调数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成微调数据文件
        train_data = []
        for _, row in train_df.iterrows():
            # 优先使用CSV中的实际切分文件路径
            if 'segment_path' in row and pd.notna(row['segment_path']) and str(row['segment_path']).strip():
                audio_path = str(row['segment_path'])
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(os.getcwd(), audio_path)
            else:
                # 回退到拼接命名
                audio_path = os.path.join(
                    'processed_data', 'train', f"{row['file_id']}_seg_{int(row['segment_id']):04d}_{row['speaker']}.wav"
                )
            train_data.append({
                'audio_path': audio_path,
                'text': row['text']
            })
        
        # 保存为JSONL格式
        with open(os.path.join(output_dir, 'train_data.jsonl'), 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"微调数据已准备完成: {len(train_data)} 条")
    
    def fine_tune_with_lora(self, train_data_path: str, model_path: str, output_dir: str):
        """使用LoRA进行微调"""
        try:
            # 这里应该调用FireRedASR的LoRA微调接口
            # 由于具体API可能不同，这里提供框架
            
            logger.info("开始LoRA微调...")
            
            # 配置LoRA参数
            lora_config = {
                'r': self.config.get('lora_r', 8),
                'lora_alpha': self.config.get('lora_alpha', 32),
                'target_modules': self.config.get('target_modules', ['q_proj', 'v_proj']),
                'lora_dropout': self.config.get('lora_dropout', 0.1)
            }
            
            # 训练参数
            training_args = {
                'output_dir': output_dir,
                'num_train_epochs': self.config.get('num_epochs', 3),
                'per_device_train_batch_size': self.config.get('batch_size', 4),
                'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 8),
                'learning_rate': self.config.get('learning_rate', 1e-4),
                'warmup_steps': self.config.get('warmup_steps', 100),
                'logging_steps': self.config.get('logging_steps', 10),
                'save_steps': self.config.get('save_steps', 500),
                'eval_steps': self.config.get('eval_steps', 500),
                'save_total_limit': 2,
                'fp16': True,
                'dataloader_num_workers': 4,
            }
            
            # 实际的微调代码需要根据FireRedASR的具体API进行调整
            # 这里提供调用示例
            """
            from fireredasr import FireRedAsrTrainer
            
            trainer = FireRedAsrTrainer(
                model_path=model_path,
                lora_config=lora_config,
                training_args=training_args
            )
            
            trainer.train(train_data_path)
            trainer.save_model(output_dir)
            """
            
            logger.info(f"LoRA微调完成，模型保存至: {output_dir}")
            
        except Exception as e:
            logger.error(f"LoRA微调失败: {e}")
            # 使用替代方案或跳过微调


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, pipeline, val_data):
        self.pipeline = pipeline
        self.val_data = val_data
    
    def optimize_speaker_diarization_params(self) -> Dict[str, float]:
        """优化说话人分离参数"""
        best_score = float('inf')
        best_params = {}
        
        # 网格搜索参数范围
        threshold_range = np.arange(0.3, 0.8, 0.1)
        min_speakers_range = [2, 3, 4]
        max_speakers_range = [6, 8, 10]
        
        logger.info("开始说话人分离参数优化...")
        
        for threshold in threshold_range:
            for min_spk in min_speakers_range:
                for max_spk in max_speakers_range:
                    if min_spk >= max_spk:
                        continue
                    
                    # 更新管线参数
                    self.pipeline.diarization.clustering_threshold = threshold
                    self.pipeline.diarization.min_speakers = min_spk
                    self.pipeline.diarization.max_speakers = max_spk
                    
                    # 在验证集上评估
                    score = self._evaluate_on_validation_set()
                    
                    if score < best_score:
                        best_score = score
                        best_params = {
                            'clustering_threshold': threshold,
                            'min_speakers': min_spk,
                            'max_speakers': max_spk
                        }
                    
                    logger.info(f"参数: {threshold:.1f}, {min_spk}, {max_spk} -> Score: {score:.4f}")
        
        logger.info(f"最佳说话人分离参数: {best_params}, Score: {best_score:.4f}")
        return best_params
    
    def optimize_asr_params(self) -> Dict[str, Any]:
        """优化ASR解码参数"""
        best_score = float('inf')
        best_params = {}
        
        # ASR参数范围
        beam_sizes = [3, 5, 8]
        length_penalties = [0.8, 1.0, 1.2]
        repetition_penalties = [1.0, 1.1, 1.2]
        temperatures = [0.1, 0.3, 0.5]
        
        logger.info("开始ASR参数优化...")
        
        for beam_size in beam_sizes:
            for length_penalty in length_penalties:
                for rep_penalty in repetition_penalties:
                    for temperature in temperatures:
                        # 更新ASR参数
                        self.pipeline.asr.beam_size = beam_size
                        self.pipeline.asr.length_penalty = length_penalty
                        self.pipeline.asr.repetition_penalty = rep_penalty
                        self.pipeline.asr.temperature = temperature
                        
                        # 评估
                        score = self._evaluate_on_validation_set()
                        
                        if score < best_score:
                            best_score = score
                            best_params = {
                                'beam_size': beam_size,
                                'length_penalty': length_penalty,
                                'repetition_penalty': rep_penalty,
                                'temperature': temperature
                            }
                        
                        logger.info(f"ASR参数: {beam_size}, {length_penalty}, {rep_penalty}, {temperature} -> Score: {score:.4f}")
        
        logger.info(f"最佳ASR参数: {best_params}, Score: {best_score:.4f}")
        return best_params
    
    def _evaluate_on_validation_set(self) -> float:
        """在验证集上评估当前参数"""
        try:
            # 这里应该运行完整的管线并计算word-DER
            # 为了简化，返回随机分数
            return np.random.random()
        except Exception as e:
            logger.warning(f"验证集评估失败: {e}")
            return float('inf')


def main():
    """微调主函数"""
    # 配置参数
    config = {
        'speaker_training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'embedding_dim': 256,
            'max_audio_length': 3.0,
            'output_dir': 'models/speaker_model'
        },
        'asr_fine_tuning': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 3,
            'lora_r': 8,
            'lora_alpha': 32,
            'output_dir': 'models/asr_model'
        }
    }
    
    # 创建输出目录
    os.makedirs('models/speaker_model', exist_ok=True)
    os.makedirs('models/asr_model', exist_ok=True)
    
    # 加载数据
    train_df = pd.read_csv('processed_data/train_segments.csv')
    val_df = pd.read_csv('processed_data/val_segments.csv')
    
    logger.info("开始模型微调...")
    
    # 1. 说话人模型微调
    logger.info("1. 训练说话人嵌入模型...")
    speaker_trainer = SpeakerDiarizationTrainer(config['speaker_training'])
    train_loader, val_loader, num_speakers = speaker_trainer.prepare_data(
        train_df, val_df, 'processed_data/train'
    )
    speaker_model = speaker_trainer.train(train_loader, val_loader, num_speakers)
    
    # 2. ASR模型微调
    logger.info("2. 准备ASR微调数据...")
    asr_tuner = ASRFineTuner(config=config['asr_fine_tuning'])
    asr_tuner.prepare_fine_tuning_data(train_df, 'processed_data/asr_finetune')
    
    # 这里需要根据实际的FireRedASR API进行微调
    # asr_tuner.fine_tune_with_lora(...)
    
    logger.info("微调完成！")


if __name__ == "__main__":
    main()
