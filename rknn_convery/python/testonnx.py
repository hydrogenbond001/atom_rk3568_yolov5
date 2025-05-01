import onnx
onnx_model = onnx.load("../exp64/weights/best.onnx")  # 会直接报错如果文件损坏
onnx.checker.check_model(onnx_model)
print("Model is valid!")
