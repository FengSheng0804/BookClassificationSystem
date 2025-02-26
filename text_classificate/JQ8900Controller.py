import serial
import time

class JQ8900Controller:
    def __init__(self, port, baudrate=9600):
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        time.sleep(2)

    # 通用指令发送方法
    def send_command(self, cmd):
        self.ser.write(cmd)
        time.sleep(0.1)

    # 两线串口控制示例（推荐）
    def uart2_play(self, track_num):
        # AA 07 02 [曲目高8位] [曲目低8位] [校验和]
        track_high = (track_num >> 8) & 0xFF
        track_low = track_num & 0xFF
        checksum = (0xAA + 0x07 + 0x02 + track_high + track_low) & 0xFF
        cmd = bytes([0xAA, 0x07, 0x02, track_high, track_low, checksum])
        self.send_command(cmd)

    def set_volume(self, level):
        # 音量范围0-30
        level = max(0, min(30, level))
        cmd = bytes([0xAA, 0x13, 0x01, level, (0xAA + 0x13 + 0x01 + level) & 0xFF])
        self.send_command(cmd)

    def stop(self):
        cmd = bytes([0xAA, 0x04, 0x00, 0xAE])  # 两线停止
        self.send_command(cmd)

    def close(self):
        self.ser.close()

# 使用示例（推荐两线串口模式）
if __name__ == "__main__":
    # 注意：必须使用独立供电，不能通过USB连接电脑控制
    controller = JQ8900Controller(port='COM4', baudrate=9600)
    
    # 设置音量（20级）
    controller.set_volume(20)

    # 播放第1首（确保存在00001.mp3）
    controller.uart2_play(1)
    
    time.sleep(2)
    controller.stop()
    controller.close()
