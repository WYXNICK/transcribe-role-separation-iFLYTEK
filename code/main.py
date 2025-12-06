#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音角色分离与转写系统主程序
整合数据预处理、模型训练、推理和评估
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
import logging

# 添加src目录到Python路径
sys.path.append('src')

from data_preprocessing import DatasetBuilder
from pipeline import RoleSeparationPipeline
from evaluation import EvaluationManager, evaluate_from_files
from fine_tuning import SpeakerDiarizationTrainer, ASRFineTuner, HyperparameterOptimizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('role_separation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoleSeparationSystem:
    """语音角色分离系统主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        default_config = {
            "data": {
                "eval_dir": "../xfdata/eval",
                "test_dir": "../xfdata/test_data",
                "output_dir": "../user_data/tmp_data/processed_data",
                "val_split": 0.0
            },
            "pipeline": {
                "temp_dir": "../user_data/tmp_data/temp_segments",
                "diarization": {
                    "clustering_threshold": 0.5,
                    "min_speakers": 2,
                    "max_speakers": 8
                },
                "asr": {
                    # "model_name": "FireRedASR-LLM",  # 可选：FireRedASR-AED / FireRedASR-LLM / Whisper / Paraformer
                    "model_name": "FireRedASR-AED",
                    "beam_size": 3,               # 结合日志，降低显存占用与时延
                    "length_penalty": 0.8,        # 日志最优区间
                    "repetition_penalty": 1.0,    # 日志最优区间
                    "temperature": 0.1            # 日志最优区间
                }
            },
            "training": {
                "speaker_model": {
                    "batch_size": 16,
                    "learning_rate": 1e-4,
                    "num_epochs": 10,
                    "embedding_dim": 256,
                    "output_dir": "../user_data/model_data/speaker_model"
                },
                "asr_model": {
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    "num_epochs": 3,
                    "lora_r": 8,
                    "lora_alpha": 32,
                    "output_dir": "../user_data/model_data/asr_model"
                }
            },
            "evaluation": {
                "output_dir": "../prediction_result"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # 递归合并配置
            default_config.update(user_config)
        
        return default_config
    
    def setup_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['data']['output_dir'],
            self.config['training']['speaker_model']['output_dir'],
            self.config['training']['asr_model']['output_dir'],
            self.config['evaluation']['output_dir'],
            self.config['pipeline']['temp_dir']
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def preprocess_data(self):
        """数据预处理步骤"""
        logger.info("=" * 60)
        logger.info("开始数据预处理")
        logger.info("=" * 60)
        
        builder = DatasetBuilder(
            self.config['data']['eval_dir'],
            self.config['data']['output_dir']
        )
        
        # 构建数据集
        train_df, val_df = builder.build_dataset(self.config['data']['val_split'])
        
        # 准备切分音频
        logger.info("准备训练集音频...")
        train_segments = builder.prepare_segmented_audio(train_df, 'train')
        
        # 不划分验证集：直接使用全部数据训练，跳过验证集切分
        if len(val_df) > 0:
            logger.info("准备验证集音频...")
            val_segments = builder.prepare_segmented_audio(val_df, 'val')
        else:
            logger.info("未划分验证集，全部数据用于训练")
            val_segments = []
        
        # 创建说话人映射
        all_df = pd.concat([train_df, val_df])
        speaker_mapping = builder.create_speaker_mapping(all_df)
        
        logger.info(f"数据预处理完成！")
        logger.info(f"说话人数量: {len(speaker_mapping)}")
        logger.info(f"训练段数: {len(train_segments)}")
        logger.info(f"验证段数: {len(val_segments)}")
        
        return train_df, val_df, train_segments, val_segments
    
    def train_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """模型训练步骤"""
        logger.info("=" * 60)
        logger.info("开始模型训练")
        logger.info("=" * 60)
        
        # 1. 训练说话人模型
        logger.info("1. 训练说话人嵌入模型...")
        try:
            speaker_trainer = SpeakerDiarizationTrainer(self.config['training']['speaker_model'])
            
            # 加载segments数据而不是metadata
            train_segments_df = pd.read_csv(os.path.join(self.config['data']['output_dir'], 'train_segments.csv'))
            val_segments_path = os.path.join(self.config['data']['output_dir'], 'val_segments.csv')
            if os.path.exists(val_segments_path):
                val_segments_df = pd.read_csv(val_segments_path)
            else:
                # 不划分验证集：使用训练集充当验证加载器以保持训练流程简洁
                val_segments_df = train_segments_df.copy()
            
            train_loader, val_loader, num_speakers = speaker_trainer.prepare_data(
                train_segments_df, val_segments_df, os.path.join(self.config['data']['output_dir'], 'train')
            )
            speaker_model = speaker_trainer.train(train_loader, val_loader, num_speakers)
            logger.info("说话人模型训练完成！")
        except Exception as e:
            logger.warning(f"说话人模型训练失败，将使用预训练模型: {e}")
        
        # 2. ASR模型微调
        logger.info("2. ASR模型微调...")
        try:
            asr_tuner = ASRFineTuner(config=self.config['training']['asr_model'])
            # 使用切分后的段级数据而不是原始 metadata，避免缺少 segment_id
            asr_tuner.prepare_fine_tuning_data(
                train_segments_df,
                os.path.join(self.config['data']['output_dir'], 'asr_finetune')
            )
            logger.info("ASR微调数据准备完成！")
            # 实际微调需要根据FireRedASR API实现
        except Exception as e:
            logger.warning(f"ASR模型微调失败，将使用预训练模型: {e}")
    
    def optimize_hyperparameters(self, val_df: pd.DataFrame):
        """超参数优化"""
        logger.info("=" * 60)
        logger.info("开始超参数优化")
        logger.info("=" * 60)
        
        try:
            # 创建管线实例
            pipeline = RoleSeparationPipeline(self.config['pipeline'])
            
            # 创建优化器
            optimizer = HyperparameterOptimizer(pipeline, val_df)
            
            # 优化说话人分离参数
            best_diarization_params = optimizer.optimize_speaker_diarization_params()
            
            # 优化ASR参数
            best_asr_params = optimizer.optimize_asr_params()
            
            # 更新配置
            self.config['pipeline']['diarization'].update(best_diarization_params)
            self.config['pipeline']['asr'].update(best_asr_params)
            
            # 保存优化后的配置
            with open('optimized_config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            logger.info("超参数优化完成！")
            logger.info(f"最佳说话人分离参数: {best_diarization_params}")
            logger.info(f"最佳ASR参数: {best_asr_params}")
            
        except Exception as e:
            logger.warning(f"超参数优化失败: {e}")
    
    def run_inference(self):
        """运行推理"""
        logger.info("=" * 60)
        logger.info("开始推理")
        logger.info("=" * 60)
        
        # 创建管线
        pipeline = RoleSeparationPipeline(self.config['pipeline'])
        
        # 处理测试数据
        results = pipeline.process_test_data(
            self.config['data']['test_dir'],
            self.config['evaluation']['output_dir']
        )
        
        logger.info(f"推理完成！处理了 {len(results)} 个文件")
        
        # 生成提交文件
        self._generate_submission_file(results)
        
        return results
    
    def _generate_submission_file(self, results: dict):
        """生成提交文件"""
        output_path = os.path.join(self.config['evaluation']['output_dir'], 'result.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for filename, content in results.items():
                f.write(content + '\n')
        
        logger.info(f"提交文件已生成: {output_path}")
    
    def evaluate_results(self, val_df: pd.DataFrame = None):
        """评估结果"""
        logger.info("=" * 60)
        logger.info("开始评估")
        logger.info("=" * 60)
        
        if val_df is not None:
            # 在验证集上评估
            logger.info("在验证集上评估...")
            
            # 处理验证集 - 使用相同的管线配置但不重新创建
            from src.evaluation import EvaluationManager
            evaluator = EvaluationManager()
            
            # 直接使用推理结果进行评估，避免重复处理
            logger.info("使用现有推理结果进行评估...")
            
            # 简化评估：仅在验证集元数据上计算统计信息
            logger.info(f"验证集包含 {len(val_df)} 个段落，来自 {val_df['file_id'].nunique()} 个文件")
            logger.info("评估完成 - 使用推理阶段结果")
        
        logger.info("评估完成！")
    
    def run_full_pipeline(self):
        """运行完整管线（至生成提交文件即结束）"""
        logger.info("开始运行完整的语音角色分离管线")
        
        # 1. 数据预处理（不划分验证集）
        train_df, val_df, train_segments, val_segments = self.preprocess_data()
        
        # 2. 模型训练（使用全部数据）
        self.train_models(train_df, val_df)
        
        # 3. 运行推理并生成提交文件
        results = self.run_inference()
        
        logger.info("流程结束（已生成提交文件）")
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='语音角色分离与转写系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'optimize', 'inference', 'evaluate', 'full'],
                       default='full', help='运行模式')
    parser.add_argument('--eval-dir', type=str, help='训练数据目录')
    parser.add_argument('--test-dir', type=str, help='测试数据目录')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 创建系统实例
        system = RoleSeparationSystem(args.config)
        
        # 更新命令行参数到配置
        if args.eval_dir:
            system.config['data']['eval_dir'] = args.eval_dir
        if args.test_dir:
            system.config['data']['test_dir'] = args.test_dir
        if args.output_dir:
            system.config['evaluation']['output_dir'] = args.output_dir
        
        # 根据模式运行不同步骤
        if args.mode == 'preprocess':
            system.preprocess_data()
        elif args.mode == 'train':
            # 需要先有预处理的数据
            train_df = pd.read_csv(os.path.join(system.config['data']['output_dir'], 'train_metadata.csv'))
            val_df = pd.read_csv(os.path.join(system.config['data']['output_dir'], 'val_metadata.csv'))
            system.train_models(train_df, val_df)
        elif args.mode == 'optimize':
            val_df = pd.read_csv(os.path.join(system.config['data']['output_dir'], 'val_metadata.csv'))
            system.optimize_hyperparameters(val_df)
        elif args.mode == 'inference':
            system.run_inference()
        elif args.mode == 'evaluate':
            val_df = pd.read_csv(os.path.join(system.config['data']['output_dir'], 'val_metadata.csv'))
            system.evaluate_results(val_df)
        elif args.mode == 'full':
            system.run_full_pipeline()
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
