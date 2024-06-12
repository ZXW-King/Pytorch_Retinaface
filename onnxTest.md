# onnx测试说明

## 测试文件说明

| 文件                       | 说明                               |
| -------------------------- | ---------------------------------- |
| onnx_export/detect_onnx.py | 单个图像测试                       |
| onnx_export/export_onnx.py | 导出onnx模型                       |
| onnx_export/test_all.py    | 多个图像测试【支持图像，视频测试】 |

## 代码运行

```python
python --trained_model FaceDetector_padding.onnx --network mobile0.25 --test_path data/test 
```

