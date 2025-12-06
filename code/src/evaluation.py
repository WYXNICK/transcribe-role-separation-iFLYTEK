#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块
主要实现word-DER (word-level Diarization Error Rate) 计算
"""

import os
import re
import jieba
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
import jiwer
from dataclasses import dataclass


@dataclass
class WordSegment:
    """词级别的语音段"""
    word: str
    start: float
    end: float
    speaker: str
    file_id: str


class ChineseTextProcessor:
    """中文文本处理器"""
    
    def __init__(self):
        # 初始化jieba分词
        jieba.initialize()
    
    def segment_text(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 清理文本
        text = self.clean_text(text)
        
        # 使用jieba分词
        words = list(jieba.cut(text))
        
        # 过滤空词和标点
        words = [w.strip() for w in words if w.strip() and not self.is_punctuation(w)]
        
        return words
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s]', '', text)
        return text.strip()
    
    def is_punctuation(self, char: str) -> bool:
        """判断是否为标点符号"""
        punctuation_ranges = [
            (0x3000, 0x303F),  # CJK符号和标点
            (0xFF00, 0xFFEF),  # 全角ASCII、全角标点
        ]
        
        if len(char) == 1:
            code = ord(char)
            for start, end in punctuation_ranges:
                if start <= code <= end:
                    return True
        
        return char in '，。！？；：""''（）【】《》〈〉、'


class WordLevelAligner:
    """词级别时间对齐器"""
    
    def __init__(self):
        self.text_processor = ChineseTextProcessor()
    
    def align_words_to_time(self, segments: List[Dict]) -> List[WordSegment]:
        """
        将文本段落对齐到词级别的时间戳
        
        Args:
            segments: 包含时间戳和文本的段落列表
                     格式: [{'start': float, 'end': float, 'speaker': str, 'text': str, 'file_id': str}]
        
        Returns:
            List[WordSegment]: 词级别的时间对齐结果
        """
        word_segments = []
        
        for segment in segments:
            words = self.text_processor.segment_text(segment['text'])
            if not words:
                continue
            
            # 简单的均匀时间分配
            duration = segment['end'] - segment['start']
            word_duration = duration / len(words)
            
            for i, word in enumerate(words):
                word_start = segment['start'] + i * word_duration
                word_end = segment['start'] + (i + 1) * word_duration
                
                word_segments.append(WordSegment(
                    word=word,
                    start=word_start,
                    end=word_end,
                    speaker=segment['speaker'],
                    file_id=segment.get('file_id', 'unknown')
                ))
        
        return word_segments


class WordDERCalculator:
    """word-DER计算器"""
    
    def __init__(self):
        self.aligner = WordLevelAligner()
        self.text_processor = ChineseTextProcessor()
    
    def calculate_word_der(self, reference_segments: List[Dict], 
                          hypothesis_segments: List[Dict],
                          file_id: str = None) -> Dict[str, float]:
        """
        计算word-DER指标
        
        Args:
            reference_segments: 参考标注段
            hypothesis_segments: 系统输出段
            file_id: 文件ID
            
        Returns:
            Dict[str, float]: DER指标结果
        """
        # 词级别对齐
        ref_words = self.aligner.align_words_to_time(reference_segments)
        hyp_words = self.aligner.align_words_to_time(hypothesis_segments)
        
        if file_id:
            ref_words = [w for w in ref_words if w.file_id == file_id]
            hyp_words = [w for w in hyp_words if w.file_id == file_id]
        
        # 计算各项错误
        w_insert, w_delete, w_confusion, w_label = self._compute_word_errors(ref_words, hyp_words)
        
        # 计算word-DER
        if w_label == 0:
            word_der = float('inf') if (w_insert + w_delete + w_confusion) > 0 else 0.0
        else:
            word_der = (w_insert + w_delete + w_confusion) / w_label
        
        return {
            'word_der': word_der,
            'w_insert': w_insert,
            'w_delete': w_delete,
            'w_confusion': w_confusion,
            'w_label': w_label,
            'insertion_rate': w_insert / w_label if w_label > 0 else 0,
            'deletion_rate': w_delete / w_label if w_label > 0 else 0,
            'confusion_rate': w_confusion / w_label if w_label > 0 else 0
        }
    
    def _compute_word_errors(self, ref_words: List[WordSegment], 
                           hyp_words: List[WordSegment]) -> Tuple[int, int, int, int]:
        """
        计算词级别的插入、删除、混淆错误
        
        Returns:
            Tuple[int, int, int, int]: (插入, 删除, 混淆, 总词数)
        """
        # 构建参考词典 {time_range: (word, speaker)}
        ref_word_map = {}
        for word_seg in ref_words:
            time_key = self._time_to_key(word_seg.start, word_seg.end)
            ref_word_map[time_key] = (word_seg.word, word_seg.speaker)
        
        # 构建假设词典
        hyp_word_map = {}
        for word_seg in hyp_words:
            time_key = self._time_to_key(word_seg.start, word_seg.end)
            hyp_word_map[time_key] = (word_seg.word, word_seg.speaker)
        
        # 使用更精确的对齐策略
        return self._detailed_alignment_errors(ref_words, hyp_words)
    
    def _detailed_alignment_errors(self, ref_words: List[WordSegment], 
                                 hyp_words: List[WordSegment]) -> Tuple[int, int, int, int]:
        """详细的词对齐错误计算"""
        # 提取词序列
        ref_text_words = [w.word for w in ref_words]
        hyp_text_words = [w.word for w in hyp_words]
        
        # 使用编辑距离进行词对齐
        alignment = self._align_word_sequences(ref_text_words, hyp_text_words)
        
        w_insert = 0
        w_delete = 0
        w_confusion = 0
        w_label = len(ref_text_words)
        
        ref_idx = 0
        hyp_idx = 0
        
        for op, ref_word, hyp_word in alignment:
            if op == 'equal':
                # 词相同，检查说话人是否一致
                if ref_idx < len(ref_words) and hyp_idx < len(hyp_words):
                    if ref_words[ref_idx].speaker != hyp_words[hyp_idx].speaker:
                        w_confusion += 1
                ref_idx += 1
                hyp_idx += 1
                
            elif op == 'substitute':
                # 词替换，计为混淆
                w_confusion += 1
                ref_idx += 1
                hyp_idx += 1
                
            elif op == 'delete':
                # 词删除
                w_delete += 1
                ref_idx += 1
                
            elif op == 'insert':
                # 词插入
                w_insert += 1
                hyp_idx += 1
        
        return w_insert, w_delete, w_confusion, w_label
    
    def _align_word_sequences(self, ref_words: List[str], 
                            hyp_words: List[str]) -> List[Tuple[str, str, str]]:
        """使用动态规划进行词序列对齐"""
        m, n = len(ref_words), len(hyp_words)
        
        # DP表格
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充DP表格
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # 匹配
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )
        
        # 回溯对齐路径
        alignment = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if ref_words[i-1] == hyp_words[j-1]:
                    alignment.append(('equal', ref_words[i-1], hyp_words[j-1]))
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j-1] + 1:
                    alignment.append(('substitute', ref_words[i-1], hyp_words[j-1]))
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j] + 1:
                    alignment.append(('delete', ref_words[i-1], ''))
                    i -= 1
                else:
                    alignment.append(('insert', '', hyp_words[j-1]))
                    j -= 1
            elif i > 0:
                alignment.append(('delete', ref_words[i-1], ''))
                i -= 1
            else:
                alignment.append(('insert', '', hyp_words[j-1]))
                j -= 1
        
        return list(reversed(alignment))
    
    def _time_to_key(self, start: float, end: float, precision: int = 2) -> str:
        """将时间范围转换为键"""
        return f"{start:.{precision}f}-{end:.{precision}f}"


class EvaluationManager:
    """评估管理器"""
    
    def __init__(self):
        self.der_calculator = WordDERCalculator()
    
    def evaluate_pipeline_results(self, ground_truth_df: pd.DataFrame,
                                 predictions: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        评估管线结果
        
        Args:
            ground_truth_df: 真值数据DataFrame
            predictions: 预测结果字典 {file_id: [segments]}
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        results = {
            'file_results': {},
            'overall_metrics': {}
        }
        
        total_der_sum = 0
        total_files = 0
        
        for file_id in predictions:
            # 获取该文件的真值
            file_ground_truth = ground_truth_df[ground_truth_df['file_id'] == file_id]
            
            if len(file_ground_truth) == 0:
                continue
            
            # 转换为统一格式
            ref_segments = []
            for _, row in file_ground_truth.iterrows():
                ref_segments.append({
                    'start': row['start'],
                    'end': row['end'],
                    'speaker': row['speaker'],
                    'text': row['text'],
                    'file_id': file_id
                })
            
            # 预测结果
            hyp_segments = predictions[file_id]
            for seg in hyp_segments:
                seg['file_id'] = file_id
            
            # 计算该文件的word-DER
            file_metrics = self.der_calculator.calculate_word_der(
                ref_segments, hyp_segments, file_id
            )
            
            results['file_results'][file_id] = file_metrics
            
            if not np.isinf(file_metrics['word_der']):
                total_der_sum += file_metrics['word_der']
                total_files += 1
        
        # 计算总体指标
        if total_files > 0:
            results['overall_metrics'] = {
                'mean_word_der': total_der_sum / total_files,
                'num_files': total_files,
                'processed_files': len(predictions)
            }
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """打印评估报告"""
        print("\n" + "="*60)
        print("Word-DER 评估报告")
        print("="*60)
        
        overall = results['overall_metrics']
        print(f"总体指标:")
        print(f"  平均 word-DER: {overall.get('mean_word_der', 0):.4f}")
        print(f"  处理文件数: {overall.get('processed_files', 0)}")
        print(f"  有效文件数: {overall.get('num_files', 0)}")
        
        print(f"\n各文件详细结果:")
        for file_id, metrics in results['file_results'].items():
            print(f"  {file_id}:")
            print(f"    word-DER: {metrics['word_der']:.4f}")
            print(f"    插入错误: {metrics['w_insert']}")
            print(f"    删除错误: {metrics['w_delete']}")
            print(f"    混淆错误: {metrics['w_confusion']}")
            print(f"    总词数: {metrics['w_label']}")
        
        print("="*60)


def evaluate_from_files(ground_truth_file: str, predictions_dir: str) -> Dict[str, Any]:
    """
    从文件评估结果
    
    Args:
        ground_truth_file: 真值CSV文件路径
        predictions_dir: 预测结果目录
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    # 加载真值数据
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    # 加载预测结果
    predictions = {}
    
    for pred_file in os.listdir(predictions_dir):
        if pred_file.endswith('.txt'):
            file_id = pred_file.replace('.txt', '')
            pred_path = os.path.join(predictions_dir, pred_file)
            
            # 解析预测文件
            segments = parse_prediction_file(pred_path)
            if segments:
                predictions[file_id] = segments
    
    # 进行评估
    evaluator = EvaluationManager()
    results = evaluator.evaluate_pipeline_results(ground_truth_df, predictions)
    evaluator.print_evaluation_report(results)
    
    return results


def parse_prediction_file(file_path: str) -> List[Dict]:
    """解析预测结果文件"""
    segments = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_start = 0.0
        current_duration = 5.0  # 假设每行5秒，实际需要更精确的时间戳
        
        for line in lines:
            line = line.strip()
            if ':' in line and not line.endswith('.wav'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    
                    segments.append({
                        'start': current_start,
                        'end': current_start + current_duration,
                        'speaker': speaker,
                        'text': text
                    })
                    
                    current_start += current_duration
    
    except Exception as e:
        print(f"解析预测文件失败 {file_path}: {e}")
    
    return segments


def main():
    """测试评估功能"""
    # 示例数据
    ref_segments = [
        {'start': 0.0, 'end': 2.0, 'speaker': 'spk1', 'text': '你好，我是小明', 'file_id': 'test'},
        {'start': 2.0, 'end': 4.0, 'speaker': 'spk2', 'text': '很高兴认识你', 'file_id': 'test'},
    ]
    
    hyp_segments = [
        {'start': 0.0, 'end': 2.0, 'speaker': 'spk1', 'text': '你好我是小明', 'file_id': 'test'},
        {'start': 2.0, 'end': 4.0, 'speaker': 'spk1', 'text': '很高兴认识你', 'file_id': 'test'},  # 说话人错误
    ]
    
    calculator = WordDERCalculator()
    results = calculator.calculate_word_der(ref_segments, hyp_segments, 'test')
    
    print("测试结果:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
