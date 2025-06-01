import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

# 导入 YOLOv5 类
from yolov5 import YoloV5

# 标签映射
label_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'F', 15: 'H', 16: 'L', 17: 'M', 18: 'O',
    19: 'P', 20: 'R', 21: 'S', 22: 'T', 23: 'V', 24: 'W', 25: 'X', 26: 'Z', 27: 'Y', 28: 'Q'
}


# 初始化 Flask app
app = Flask(__name__)

# 初始化 YOLOv5 模型
model = YoloV5(yolov5_yaml_path='config/yolov5.yaml')
print("[INFO] 完成YoloV5模型加载")

def base64_to_image(base64_str):
    """将base64字符串转换为OpenCV图像"""
    image_data = base64.b64decode(base64_str.split(",")[1])
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# 首页路由，返回index.html
@app.route('/')
def index():
    return render_template('index.html')

# 定义上传图片并进行目标检测的路由
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # 将Base64图像转换为OpenCV格式
    img = base64_to_image(data['image'])

    # 进行目标检测
    canvas, class_id_list, xyxy_list, conf_list = model.detect(img)

    # 将检测结果打包返回
    results = []
    for class_id, bbox in zip(class_id_list, xyxy_list):
        result = {
            'label': label_map[int(class_id)],  # 获取字符
            'bbox': [int(x) for x in bbox]  # bounding box
        }
        results.append(result)

    return jsonify({'detections': results})


# 启动 Flask 服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
