#!/usr/bin/python3
# -*- coding:utf-8 -*-

import RPi.GPIO as GPIO
import time

# 设置使用BCM编码
GPIO.setmode(GPIO.BCM)

# 设置第17号引脚为输出模式
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)

try:
    while True:
        GPIO.output(4, GPIO.HIGH)  # 点亮LED
        GPIO.output(17, GPIO.HIGH)  # 点亮LED
        GPIO.output(27, GPIO.HIGH)  # 点亮LED
        GPIO.output(18, GPIO.HIGH)  # 点亮LED
        GPIO.output(22, GPIO.HIGH)  # 点亮LED
except KeyboardInterrupt:
    GPIO.cleanup()                 # 捕获CTRL+C清理GPIO设置
