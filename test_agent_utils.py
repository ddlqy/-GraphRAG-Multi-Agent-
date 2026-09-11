"""测试Agent工具函数"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
utils = importlib.import_module("src.agents.utils")
parse_action = utils.parse_action
get_action_list = utils.get_action_list
split_compound_func = utils.split_compound_func
get_compound_func = utils.get_compound_func
normalize_answer = utils.normalize_answer
EM = utils.EM


class TestAgentUtils:
    """Agent工具函数测试"""
    
    def test_parse_action(self):
        """测试解析动作"""
        # 简单动作
        action_type, argument = parse_action("Retrieve[国家奖学金]")
        assert action_type == "Retrieve"
        assert argument == "国家奖学金"
        
        # 复杂动作
        action_type, argument = parse_action("Feature[scholarship_001, name]")
        assert action_type == "Feature"
        assert argument == "scholarship_001, name"
        
        # Finish动作
        action_type, argument = parse_action("Finish[8000元/年]")
        assert action_type == "Finish"
        assert argument == "8000元/年"
    
    def test_get_action_list(self):
        """测试获取动作列表"""
        # 单个动作
        actions = get_action_list("Retrieve[国家奖学金]")
        assert len(actions) == 1
        assert actions[0] == "Retrieve[国家奖学金]"
        
        # Finish动作
        actions = get_action_list("Finish[8000元/年]")
        assert len(actions) == 1
    
    def test_split_compound_func(self):
        """测试拆分复合函数"""
        mid, outer = split_compound_func("Feature[Retrieve[Caffeine], inchikey]")
        assert mid == "Retrieve[Caffeine]"
        assert outer == "Feature[mid, inchikey]"
    
    def test_get_compound_func(self):
        """测试获取复合函数"""
        actions = get_compound_func("Feature[Retrieve[国家奖学金], amount]")
        assert actions is not None
        assert len(actions) == 2
        assert "Retrieve[国家奖学金]" in actions
    
    def test_normalize_answer(self):
        """测试答案标准化"""
        assert normalize_answer("The answer is 42") == "answer is 42"
        assert normalize_answer("Hello World!") == "hello world"
        assert normalize_answer("8000元/年") == "8000元年"
    
    def test_em(self):
        """测试精确匹配"""
        assert EM("8000元/年", "8000元/年") == True
        assert EM("The answer is 42", "the answer is 42") == True
        assert EM("hello", "world") == False
