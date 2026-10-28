import cv2
import numpy as np
import gradio as gr
from rapidocr_onnxruntime import RapidOCR

# 初始化RapidOCR识别引擎
ocr_engine = RapidOCR()


def ocr_recognize(img):
    """
    对矫正后的文档图片执行OCR文字识别
    :param img: 矫正完成的BGR图像
    :return: 拼接后的识别文本字符串
    """
    result, _ = ocr_engine(img)
    text_list = []
    if result:
        # result里面每条是 [坐标,文字,置信度]，取文字部分
        for item in result:
            txt = item[1]
            text_list.append(txt)
    # 拼接文字，没有识别内容则返回提示
    return "\n".join(text_list) if text_list else "未识别到文字"


def scan_document(img, canny_low=50, canny_high=150, area_ratio=0.25):
    """
    文档扫描核心函数：边缘检测、寻找文档四边形、透视变换矫正、二值化
    :param img: 原始输入BGR图像
    :param canny_low: Canny边缘检测低阈值
    :param canny_high: Canny边缘检测高阈值
    :param area_ratio: 最小轮廓面积占整张图面积比例，过滤小噪点轮廓
    :return: warped:矫正后二值图；contour_img:绘制了文档轮廓的原图
             如果找不到四边形，返回 None, contour_img
    """
    contour_img = img.copy()  # 拷贝原图用来画轮廓，不修改原图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊降噪
    edged = cv2.Canny(blur, canny_low, canny_high)  # Canny边缘检测

    # 寻找外部轮廓
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, contour_img

    # 轮廓按面积从大到小排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * area_ratio

    screenCnt = None
    # 遍历轮廓，寻找4边形（文档）
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 多边形逼近
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None, contour_img

    # 在图上画出找到的文档轮廓，绿色，线宽3
    cv2.drawContours(contour_img, [screenCnt], -1, (0, 255, 0), 3)

    # 四点坐标排序：左上、右上、右下、左下
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上：x+y最小
    rect[2] = pts[np.argmax(s)]  # 右下：x+y最大

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：y-x最小
    rect[3] = pts[np.argmax(diff)]  # 左下：y-x最大

    tl, tr, br, bl = rect
    # 计算矫正后图片宽高
    maxWidth = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    # 目标四点坐标
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # 计算透视变换矩阵，并执行变换
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # OTSU自动阈值二值化，增强文字对比度
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warped_bin = cv2.threshold(
        warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    warped = cv2.cvtColor(warped_bin, cv2.COLOR_GRAY2BGR)

    return warped, contour_img


def process_image(input_img, canny_low, canny_high, area_ratio):
    """
    Gradio回调入口函数
    :param input_img: gradio传入的numpy图片
    :param canny_low: 滑动条参数
    :param canny_high: 滑动条参数
    :param area_ratio: 滑动条参数
    :return: 矫正图，轮廓图，识别文本
    """
    # 判断是否上传图片
    if input_img is None:
        return None, None, "识别失败：请先上传图片！"

    # 兼容灰度图、RGBA透明图，统一转为BGR
    if len(input_img.shape) == 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    elif input_img.shape[2] == 4:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGBA2BGR)

    warped, contour_img = scan_document(input_img, canny_low, canny_high, area_ratio)

    # 没有找到文档四边形
    if warped is None:
        return (
            None,
            contour_img,
            "识别失败：没有检测到文档四边形，请调整参数或更换图片！",
        )

    text = ocr_recognize(warped)
    return warped, contour_img, text


# Gradio界面搭建
with gr.Blocks(title="文档扫描矫正 + OCR识别", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
        # 📄 OpenCV 文档扫描矫正 + RapidOCR 文字识别

        使用步骤：上传文档照片 → 点击开始扫描识别 → 查看轮廓检测、矫正结果和识别文字。
        """)

    with gr.Row():
        img_input = gr.Image(label="上传文档照片", type="numpy")

    with gr.Row():
        img_contour = gr.Image(label="轮廓检测图", type="numpy")
        img_result = gr.Image(label="矫正二值化结果", type="numpy")

    with gr.Row():
        txt_out = gr.Textbox(label="OCR 识别文字", lines=10)

    with gr.Row():
        canny_low = gr.Slider(
            minimum=10, maximum=200, value=50, step=1, label="Canny 下限"
        )
        canny_high = gr.Slider(
            minimum=50, maximum=300, value=150, step=1, label="Canny 上限"
        )
        area_ratio = gr.Slider(
            minimum=0.05, maximum=0.8, value=0.25, step=0.01, label="最小面积比例"
        )

    with gr.Row():
        btn_run = gr.Button("开始扫描识别", variant="primary")

    gr.Markdown("""
        ## 参数说明

        - **Canny 下限**：边缘检测的最低强度阈值，值越低越容易检测到边缘。
        - **Canny 上限**：边缘检测的最高强度阈值，值越高只保留明显边缘。
        - **最小面积比例**：文档轮廓占整张图片的最小面积比例，用于过滤小轮廓噪点。

        如果识别失败，请尝试：
        1. 上传背景干净、文档四角完整清晰的照片；
        2. 调低 Canny 下限；
        3. 减小最小面积比例。
        """)

    # 绑定按钮点击事件
    btn_run.click(
        fn=process_image,
        inputs=[img_input, canny_low, canny_high, area_ratio],
        outputs=[img_result, img_contour, txt_out],
    )

if __name__ == "__main__":
    demo.launch()
