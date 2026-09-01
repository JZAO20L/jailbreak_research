"""
记忆机制（可选）

提供长期记忆功能，积累攻击经验。
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class Memory:
    """
    长期记忆机制
    
    存储和检索攻击经验，帮助 Agent 做出更好的决策。
    """
    
    def __init__(
        self,
        memory_path: Optional[str] = None,
        max_memories: int = 1000,
        similarity_threshold: float = 0.7,
    ):
        """
        初始化记忆
        
        Args:
            memory_path: 记忆存储路径（可选）
            max_memories: 最大记忆数量
            similarity_threshold: 相似度阈值
        """
        self.memory_path = memory_path
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        
        self.memories: List[Dict] = []
        
        # 加载已有记忆
        if memory_path and Path(memory_path).exists():
            self._load_memories()
    
    def update(self, experience: Dict[str, Any]):
        """
        更新记忆(瘦身契约:不存完整 action/skill 内容,只记"用了哪个 skill + 结果")

        Args:
            experience: {
                "prompt": str,
                "turn": int,
                "skill_idx": int,
                "skill_name": str,
                "success": bool,
                "summary": str,   # ≤1 句话的结果摘要(agent.py 展示用)
            }
        """
        # 添加时间戳
        import datetime
        experience["timestamp"] = datetime.datetime.now().isoformat()
        
        # 添加记忆
        self.memories.append(experience)
        
        # 限制记忆数量
        if len(self.memories) > self.max_memories:
            # 保留最新的
            self.memories = self.memories[-self.max_memories:]
        
        # 保存到文件
        if self.memory_path:
            self._save_memories()
    
    def retrieve(self, prompt: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关记忆
        
        Args:
            prompt: 当前 prompt
            top_k: 返回 top-k 条记忆
        
        Returns:
            相关记忆列表
        """
        if not self.memories:
            return []
        
        # 简单相似度计算（基于关键词）
        prompt_keywords = set(prompt.lower().split())
        
        scored_memories = []
        for mem in self.memories:
            mem_keywords = set(mem["prompt"].lower().split())
            
            # Jaccard 相似度
            intersection = len(prompt_keywords & mem_keywords)
            union = len(prompt_keywords | mem_keywords)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= self.similarity_threshold:
                scored_memories.append((similarity, mem))
        
        # 按相似度排序
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # 返回 top-k
        return [mem for _, mem in scored_memories[:top_k]]
    
    def _load_memories(self):
        """从文件加载记忆"""
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.memories = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load memories: {e}")
            self.memories = []
    
    def _save_memories(self):
        """保存记忆到文件"""
        try:
            Path(self.memory_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save memories: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if not self.memories:
            return {"count": 0, "success_rate": 0.0}
        
        success_count = sum(1 for mem in self.memories if mem.get("success", False))
        
        return {
            "count": len(self.memories),
            "success_count": success_count,
            "success_rate": success_count / len(self.memories),
            "avg_turns": sum(mem.get("turns", 0) for mem in self.memories) / len(self.memories),
        }
