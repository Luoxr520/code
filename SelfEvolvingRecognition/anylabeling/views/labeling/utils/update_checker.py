# -*- coding: utf-8 -*-
"""
update_checker.py — 自进化侦测:更新检查(已关闭对上游的联网检查)

本项目是基于 SelfEvolvingRecognition 的个人/演示用 fork,不跟随上游(Luoxr520)的发布节奏。
为避免启动时联网询问上游、并去掉那条误导性的"发现新版本 -> CVHub releases"提示,
这里把更新检查改为本地无操作:始终返回"已是最新"。保留原有函数签名,
调用方(app.py / about_dialog.py)无需任何改动。

如果将来你给自己的仓库(Luoxr520/code)打了 GitHub Release 并想恢复联网检查,
按文件末尾注释里的旧逻辑改回、把仓库地址换成你自己的即可。
"""
from anylabeling.app_info import __version__


def _no_update():
    """统一的"已是最新"返回值,字段与原实现保持一致,避免调用方取键报错。"""
    return {
        "has_update": False,
        "current_version": __version__,
        "latest_version": __version__,
        "download_url": "",
        "release_notes": "",
        "published_at": "",
    }


def check_for_updates_async(callback=None, timeout=10):
    """已关闭联网检查:不查询上游,直接回调"已是最新"(不阻塞、不联网)。"""
    if callback:
        callback(_no_update())


def check_for_updates_sync(timeout=10):
    """已关闭联网检查:始终返回"已是最新"。"""
    return _no_update()
