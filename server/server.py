import time
import os
from text_classificate.JQ8900Controller import JQ8900Controller
from flask import Flask, render_template, request

# 文件保存配置
UPLOAD_FOLDER = '/home/pi/dc/content/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='/home/pi/dc/server/templates', static_folder='/home/pi/dc/server/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 限制上传文件大小为10M

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '没有选择文件'
    
    file = request.files['file']
    if file.filename == '':
        return '无效的文件名'
    
    if file and allowed_file(file.filename):
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pic.jpg')
        
        try:
            file.save(save_path)
            # 播放提示音
            # 控制语音播报
            controller = JQ8900Controller(port='/dev/ttyUSB0', baudrate=9600)
            # 设置音量（20级）
            controller.set_volume(20) 
            controller.uart2_play(13)

            # 写入日志
            with open('/home/pi/dc/content/log.txt', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t手机上传文件：pic.jpg\n")
            return f'文件 pic.jpg 上传成功'
        except Exception as e:
            return f'保存失败：{str(e)}'
    
    return '不支持的文件类型'

def start_server(host='0.0.0.0', port=8080):
    app.run(host=host, port=port)  # 必须绑定到所有接口
    # 写入日志
    with open('/home/pi/dc/content/log.txt', 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t启动服务器成功\n")
