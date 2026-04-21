# Virtual Touch Demo (MediaPipe)

一个用于 **Windows** 的 Python 示例：通过摄像头识别手部食指位置，在预览画面上渲染多个“无效”半透明悬浮按钮；当食指进入按钮区域时显示 **hover** 状态。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 运行

```bash
python virtual_touch_demo.py
```

## 3. 操作说明

- 对准摄像头伸出手掌。
- 移动食指，指尖进入任意按钮区域时，按钮会高亮并显示 `HOVER`。
- 按 `Q` 或 `Esc` 退出。

> 该示例仅做 hover 命中演示，不执行真实按钮功能。
