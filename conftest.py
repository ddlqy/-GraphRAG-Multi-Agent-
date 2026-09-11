"""测试配置"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_graph():
    """示例图数据"""
    return {
        "Scholarship_nodes": {
            "scholarship_001": {
                "features": {
                    "name": "国家奖学金",
                    "amount": "8000元/年",
                    "conditions": "GPA前10%，无挂科，综合素质优秀",
                    "quota": "按学院分配",
                    "deadline": "每年10月"
                },
                "neighbors": {
                    "MANAGED_BY": ["dept_jwc"],
                    "REQUIRES": ["condition_gpa", "condition_no_fail"]
                }
            },
            "scholarship_002": {
                "features": {
                    "name": "校级一等奖学金",
                    "amount": "3000元/年",
                    "conditions": "GPA前20%，无挂科",
                    "quota": "每班2人",
                    "deadline": "每年11月"
                },
                "neighbors": {
                    "MANAGED_BY": ["dept_jwc"]
                }
            }
        },
        "Department_nodes": {
            "dept_jwc": {
                "features": {
                    "name": "教务处",
                    "dean": "张主任",
                    "contact": "010-12345678",
                    "office": "行政楼3层"
                },
                "neighbors": {
                    "MANAGES": ["scholarship_001", "scholarship_002"]
                }
            },
            "dept_cs": {
                "features": {
                    "name": "计算机学院",
                    "dean": "李院长",
                    "contact": "010-87654321",
                    "office": "理工楼5层"
                },
                "neighbors": {}
            }
        },
        "Condition_nodes": {
            "condition_gpa": {
                "features": {
                    "name": "GPA要求",
                    "requirement": "GPA前10%",
                    "description": "综合绩点排名在专业前10%"
                },
                "neighbors": {
                    "REQUIRED_BY": ["scholarship_001"]
                }
            },
            "condition_no_fail": {
                "features": {
                    "name": "无挂科",
                    "requirement": "无挂科记录",
                    "description": "所有课程均及格"
                },
                "neighbors": {
                    "REQUIRED_BY": ["scholarship_001", "scholarship_002"]
                }
            }
        },
        "Location_nodes": {
            "loc_library": {
                "features": {
                    "name": "图书馆",
                    "address": "校园中心",
                    "hours": "周一至周日 8:00-22:00",
                    "type": "学习场所",
                    "contact": "010-11111111"
                },
                "neighbors": {}
            }
        },
        "Competition_nodes": {
            "comp_lanqiao": {
                "features": {
                    "name": "蓝桥杯",
                    "level": "省级",
                    "category": "B类",
                    "deadline": "每年3月",
                    "organizer": "蓝桥杯组委会"
                },
                "neighbors": {
                    "RELATED_TO": ["scholarship_001"]
                }
            }
        }
    }


@pytest.fixture
def sample_question():
    """示例问题"""
    return "国家奖学金的申请条件是什么？"
