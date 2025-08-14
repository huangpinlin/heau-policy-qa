#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
链接配置文件
包含河南大学官网确实存在的页面链接
"""

# 河南大学官网链接配置
HEANU_LINKS = {
    'scholarship': {
        'name': '奖学金相关',
        'links': [
            {
                'name': '河南大学官网',
                'url': 'https://www.henu.edu.cn/',
                'type': 'main',
                'description': '河南大学官方网站主页'
            }
        ]
    },
    'financial_aid': {
        'name': '资助相关',
        'links': [
            {
                'name': '河南大学官网',
                'url': 'https://www.henu.edu.cn/',
                'type': 'main',
                'description': '河南大学官方网站主页'
            }
        ]
    },
    'major_change': {
        'name': '转专业相关',
        'links': [
            {
                'name': '河南大学官网',
                'url': 'https://www.henu.edu.cn/',
                'type': 'main',
                'description': '河南大学官方网站主页'
            }
        ]
    }
}

def get_links_for_title(title):
    """根据文档标题获取对应的链接"""
    if '奖学金' in title:
        return HEANU_LINKS['scholarship']['links']
    elif '资助' in title:
        return HEANU_LINKS['financial_aid']['links']
    elif '转专业' in title:
        return HEANU_LINKS['major_change']['links']
    else:
        # 默认返回官网链接
        return [
            {
                'name': '河南大学官网',
                'url': 'https://www.henu.edu.cn/',
                'type': 'main',
                'description': '河南大学官方网站主页'
            }
        ]
