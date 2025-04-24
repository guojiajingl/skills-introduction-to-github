import sys
import os
import torch
import cv2
import numpy as np
import warnings
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFileDialog,
                               QSpinBox, QMessageBox, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO
import torch.nn as nn

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置环境变量
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), 'torch_cache')


class FruitDetectionSystem(QMainWindow):
    # 添加样式表常量
    STYLE_SHEET = """
        QMainWindow {
            background-color: #f0f0f0;
        }

        QLabel {
            color: #2c3e50;
            font-size: 13px;
        }

        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-size: 13px;
            min-width: 100px;
        }

        QPushButton:hover {
            background-color: #2980b9;
        }

        QPushButton:disabled {
            background-color: #bdc3c7;
        }

        QSpinBox {
            padding: 5px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            background: white;
        }

        QScrollArea {
            border: none;
        }

        #title_label {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
        }

        #device_info {
            color: #27ae60;
            font-weight: bold;
        }

        #status_label {
            color: #e67e22;
            font-weight: bold;
        }

        #result_panel {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            margin: 5px;
        }

        #image_panel {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            margin: 5px;
        }
    """

    def __init__(self):
        super().__init__()
        # 首先初始化所有变量
        self.init_variables()
        # 设置样式表
        self.setStyleSheet(self.STYLE_SHEET)
        # 然后初始化UI
        self.initUI()

    def init_variables(self):
        """初始化所有变量"""
        self.model = None
        self.current_image = None
        self.confidence_threshold = 0.5
        self.model_path = None
        self.class_names = ['apple', 'banana', 'orange']
        self.original_image = None  # 保存原始图片
        self.detected_image = None  # 保存检测后的图片
        self.showing_original = True  # 标记当前显示的是否为原图
        self.last_detection_time = None  # 记录最后一次检测时间

        # 检查是否有可用的 CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("使用 GPU 进行推理")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            print("使用 CPU 进行推理")

    def initUI(self):
        self.setWindowTitle('水果目标检测系统')
        self.setGeometry(100, 100, 1400, 800)

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建主布局
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # 创建左侧控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)

        # 创建右侧显示区域
        display_panel = self.create_display_panel()
        layout.addWidget(display_panel, 4)

    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        panel.setObjectName("control_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 添加标题
        title = QLabel("控制面板")
        title.setObjectName("title_label")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 添加设备信息显示
        device_info = QLabel(f"当前设备: {self.device}")
        device_info.setObjectName("device_info")
        device_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(device_info)

        # 创建按钮组面板
        buttons_panel = QWidget()
        buttons_layout = QVBoxLayout(buttons_panel)
        buttons_layout.setSpacing(8)

        # 添加模型路径选择按钮
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.clicked.connect(self.select_model)
        buttons_layout.addWidget(self.select_model_btn)

        # 添加模型信息按钮
        self.model_info_btn = QPushButton("显示模型信息")
        self.model_info_btn.clicked.connect(self.show_model_info)
        self.model_info_btn.setEnabled(False)
        buttons_layout.addWidget(self.model_info_btn)

        # 置信度阈值设置
        conf_widget = QWidget()
        conf_layout = QHBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_label = QLabel("置信度阈值:")
        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setRange(1, 100)
        self.conf_spinbox.setValue(50)
        self.conf_spinbox.setSuffix("%")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_spinbox)
        buttons_layout.addWidget(conf_widget)

        # 添加其他按钮
        self.init_model_btn = QPushButton("初始化模型")
        self.init_model_btn.clicked.connect(self.initialize_model)
        self.init_model_btn.setEnabled(False)
        buttons_layout.addWidget(self.init_model_btn)

        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        buttons_layout.addWidget(self.select_image_btn)

        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.detect_fruits)
        self.detect_btn.setEnabled(False)
        buttons_layout.addWidget(self.detect_btn)

        self.toggle_view_btn = QPushButton("切换原图/结果")
        self.toggle_view_btn.clicked.connect(self.toggle_view)
        self.toggle_view_btn.setEnabled(False)
        buttons_layout.addWidget(self.toggle_view_btn)

        layout.addWidget(buttons_panel)

        # 状态显示
        self.status_label = QLabel("状态: 请选择模型文件")
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # 检测结果显示
        self.result_label = QLabel("检测结果:")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        # 添加弹性空间
        layout.addStretch()

        return panel

    def create_display_panel(self):
        """创建右侧显示区域"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 创建上部图片显示区域
        image_widget = QWidget()
        image_widget.setObjectName("image_panel")
        image_layout = QVBoxLayout(image_widget)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 创建容器widget
        container = QWidget()
        container_layout = QVBoxLayout(container)

        # 图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 4px;
            }
        """)

        # 将图片标签添加到容器
        container_layout.addWidget(self.image_label)
        container_layout.addStretch()

        # 设置容器为滚动区域的widget
        scroll_area.setWidget(container)
        image_layout.addWidget(scroll_area)

        # 创建下部信息显示区域
        info_widget = QWidget()
        info_widget.setObjectName("result_panel")
        info_layout = QVBoxLayout(info_widget)

        # 文件信息
        self.file_info_label = QLabel("文件信息: ")
        info_layout.addWidget(self.file_info_label)

        # 检测时间
        self.time_label = QLabel("检测时间: ")
        info_layout.addWidget(self.time_label)

        # 检测结果
        self.detection_info_label = QLabel("检测结果: ")
        self.detection_info_label.setWordWrap(True)
        info_layout.addWidget(self.detection_info_label)

        # 将上部和下部添加到主布局
        layout.addWidget(image_widget, 7)
        layout.addWidget(info_widget, 3)

        return panel

    def select_model(self):
        """选择模型文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "模型文件 (*.pt *.pth)"
        )

        if file_name:
            self.model_path = file_name
            self.status_label.setText(f"状态: 已选择模型文件: {os.path.basename(file_name)}")
            if self.validate_model():
                self.init_model_btn.setEnabled(True)
            else:
                self.status_label.setText("状态: 模型文件无效，请重新选择")
                self.model_path = None

    def validate_model(self):
        """验证模型文件是否有效"""
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "警告", "请先选择有效的模型文件！")
            return False

        try:
            model = YOLO(self.model_path)
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型文件无效: {str(e)}")
            return False

    def show_model_info(self):
        """显示模型信息"""
        if self.model:
            info = f"模型信息:\n"
            info += f"模型文件: {os.path.basename(self.model_path)}\n"
            info += f"类别数量: {len(self.model.names)}\n"
            info += f"类别列表: {', '.join(self.model.names)}\n"
            info += f"当前置信度阈值: {self.conf_spinbox.value()}%\n"
            info += f"推理设备: {self.device}\n"
            if torch.cuda.is_available():
                info += f"GPU: {torch.cuda.get_device_name(0)}\n"
                info += f"当前显存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB"

            QMessageBox.information(self, "模型信息", info)

    def initialize_model(self):
        """初始化模型"""
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件！")
            return

        try:
            self.status_label.setText("状态: 正在加载模型...")
            QApplication.processEvents()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            self.status_label.setText("状态: 模型加载成功")
            self.detect_btn.setEnabled(True)
            self.model_info_btn.setEnabled(True)

        except Exception as e:
            error_msg = str(e)
            QMessageBox.critical(self, "错误", f"模型加载失败: {error_msg}")
            self.status_label.setText("状态: 模型加载失败")
            print(f"详细错误信息: {error_msg}")

    def select_image(self):
        """选择图片"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            try:
                self.current_image = file_name
                # 读取并保存原始图片
                self.original_image = cv2.imread(file_name)
                if self.original_image is None:
                    raise Exception("无法读取图片")

                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(self.original_image)

                # 更新文件信息
                file_info = f"文件位置: {file_name}\n"
                file_info += f"图片尺寸: {self.original_image.shape[1]}x{self.original_image.shape[0]}"
                self.file_info_label.setText(file_info)

                if self.model:
                    self.detect_btn.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
                print(f"加载图片错误: {str(e)}")

    def display_image(self, image):
        """显示图片"""
        if image is None:
            print("Error: 图片为空")
            return

        try:
            h, w, ch = image.shape
            print(f"图片尺寸: {w}x{h}, 通道数: {ch}")  # 调试信息

            # 计算缩放比例，保持宽高比
            display_width = self.image_label.width()
            display_height = self.image_label.height()
            scale = min(display_width / w, display_height / h)

            # 缩放图片
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 转换为QImage并显示
            bytes_per_line = new_w * 3
            qt_image = QImage(scaled_image.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.image_label.setPixmap(pixmap)
            print(f"图片已显示，大小: {new_w}x{new_h}")  # 调试信息

        except Exception as e:
            print(f"显示图片时出错: {str(e)}")  # 调试信息

    def toggle_view(self):
        """切换显示原图/检测结果"""
        if not hasattr(self, 'showing_original') or self.showing_original:
            if self.detected_image is not None:
                self.display_image(self.detected_image)
                self.showing_original = False
                print("切换到检测结果视图")  # 调试信息
        else:
            if self.original_image is not None:
                self.display_image(self.original_image)
                self.showing_original = True
                print("切换到原始图片视图")  # 调试信息

    def detect_fruits(self):
        """执行水果检测"""
        if not self.model or not self.current_image:
            return

        try:
            self.status_label.setText("状态: 正在检测...")
            QApplication.processEvents()

            # 记录开始时间
            start_time = time.time()

            # 执行检测
            results = self.model.predict(
                source=self.current_image,
                conf=self.conf_spinbox.value() / 100,
                device=self.device
            )

            # 计算检测时间
            detection_time = time.time() - start_time
            self.last_detection_time = detection_time

            # 更新检测时间显示
            self.time_label.setText(f"检测时间: {detection_time:.3f}秒")

            # 获取检测结果图片
            for r in results:
                self.detected_image = r.plot()
                self.display_image(self.detected_image)

                # 更新检测结果信息
                detection_info = "检测结果:\n"
                for cls in r.boxes.cls.unique():
                    class_name = self.model.names[int(cls)]
                    count = (r.boxes.cls == cls).sum()
                    conf = r.boxes.conf[r.boxes.cls == cls].mean()
                    detection_info += f"{class_name}: {int(count)}个 (置信度: {conf:.2f})\n"

                self.detection_info_label.setText(detection_info)

            # 更新状态
            self.status_label.setText("状态: 检测完成")
            self.toggle_view_btn.setEnabled(True)
            self.showing_original = False

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")
            self.status_label.setText("状态: 检测失败")
            print(f"检测错误: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FruitDetectionSystem()
    window.show()
    sys.exit(app.exec())