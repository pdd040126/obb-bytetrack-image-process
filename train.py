# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    model = YOLO(model='yolo11n-obb.pt')
                
    model.train(
        # =========================
        # 基础训练参数
        # =========================
        data="./data.yaml",
        epochs=300,
        patience=100,
        batch=16,
        device=0,
        imgsz=960,
        workers=4,
        cache=False,
        amp=True,

        # =========================
        # 优化器
        # =========================
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        
        warmup_epochs=3.0,
        cos_lr=True,
        
        # =========================
        # 图像训练方式
        # =========================
        rect=False,
        
        # =========================
        # 几何增强
        # =========================
        degrees=30.0,
        translate=0.05,
        scale=0.1,
    
        fliplr=0.5,
        flipud=0.5,

        # =========================
        # 颜色增强
        # =========================
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.2,
        
        # =========================
        # 组合增强
        # =========================
        mosaic=0.0,
        close_mosaic=30,
        mixup=0.0,
        cutmix=0.0,

        # =========================
        # 实验复现
        # =========================
        seed=24,
        deterministic=True,
    
        project="",
        name="8.27.0-n"
    )
    