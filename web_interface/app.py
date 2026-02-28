# -*- coding: utf-8 -*-
"""
ImgStegGAN Web Interface
GAN-Driven Image Steganography with Qwen Enhancement

This project is a modified and extended version of SteganoGAN
by MIT Data To AI Lab (https://github.com/DAI-Lab/SteganoGAN)

Original paper:
    Zhang, Kevin Alex and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan.
    SteganoGAN: High Capacity Image Steganography with GANs.
    MIT EECS, January 2019. (arXiv:1901.03892)

功能:
1. 信息嵌入 - 将秘密消息嵌入图像
2. 信息提取 - 从图像中提取隐藏消息
3. 保持原始图像尺寸不变
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import uuid
import base64
import threading
import time
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steganogan.qwen_integration import QwenSteganoGAN

VERSION = 'V1.0.0'
VERSION_DATE = '2026-02-28'

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'qwen-steganogan-secret-key'

_model = None
_model_lock = threading.Lock()

_operation = {
    'is_running': False,
    'type': None,
    'start_time': None,
    'interrupt': False,
    'lock': threading.Lock()
}

_executor = ThreadPoolExecutor(max_workers=2)
_task_history = []


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'output_qwen', 'qwen_steganogan.steg'
            )
            
            if os.path.exists(model_path):
                _model = QwenSteganoGAN.load(model_path, cuda=torch.cuda.is_available(), verbose=True)
            else:
                _model = QwenSteganoGAN(
                    data_depth=1,
                    hidden_size=64,
                    cuda=torch.cuda.is_available(),
                    verbose=True
                )
        return _model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_filename(original):
    if '.' in original:
        ext = original.rsplit('.', 1)[1].lower()[:10]
    else:
        ext = 'png'
    return f"{uuid.uuid4().hex}.{ext}"


def check_interrupt():
    with _operation['lock']:
        return _operation['interrupt']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/version', methods=['GET'])
def api_version():
    return jsonify({
        'success': True,
        'version': VERSION,
        'date': VERSION_DATE,
        'engine': 'ImgStegGAN',
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })


@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({
        'success': True,
        'models': [
            {'id': 'imgsteggan', 'name': 'ImgStegGAN', 'description': 'GAN-Driven Image Steganography with Qwen Enhancement'}
        ]
    })


@app.route('/api/tasks', methods=['GET'])
def api_tasks():
    limit = request.args.get('limit', 20, type=int)
    
    tasks = []
    for task in _task_history[-limit:][::-1]:
        task_type = task.get('type', 'encode')
        type_config = {
            'encode': {'name': '嵌入消息', 'icon': 'fa-lock'},
            'decode': {'name': '提取消息', 'icon': 'fa-unlock'},
            'batch': {'name': '批量处理', 'icon': 'fa-layer-group'}
        }.get(task_type, {'name': '未知操作', 'icon': 'fa-question'})
        
        tasks.append({
            'id': task.get('id', str(len(tasks) + 1)),
            'type': task_type,
            'status': task.get('status', 'completed'),
            'filename': task.get('filename', ''),
            'message_length': task.get('message_length', 0),
            'encode_time': task.get('encode_time', 0),
            'decode_time': task.get('decode_time', 0),
            'created_at': task.get('timestamp', task.get('created_at', '')),
            'type_name': type_config['name'],
            'icon': type_config['icon'],
            'progress': task.get('progress')
        })
    
    return jsonify({
        'success': True,
        'tasks': tasks
    })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件被上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': '不支持的文件类型'}), 400
        
        filename = gen_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with Image.open(filepath) as img:
            width, height = img.size
            img.thumbnail((800, 800))
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        max_capacity = (width * height * 3) // 8
        
        size_warning = None
        is_valid_for_steganography = True
        
        min_dimension = min(width, height)
        if min_dimension < 256:
            size_warning = f'图片尺寸过小 ({width}x{height})，最小要求 256x256 像素'
            is_valid_for_steganography = False
        elif min_dimension < 512:
            size_warning = f'图片尺寸较小 ({width}x{height})，建议使用 512x512 或更大的图片以获得更好的效果'
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'preview': f'data:image/png;base64,{img_base64}',
            'info': {
                'width': width,
                'height': height,
                'max_capacity': max_capacity,
                'max_capacity_text': f'约 {max_capacity} 字符',
                'size_warning': size_warning,
                'is_valid_for_steganography': is_valid_for_steganography
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'上传失败: {str(e)}'}), 500


@app.route('/api/encode', methods=['POST'])
def api_encode():
    start_time = time.time()
    
    with _operation['lock']:
        _operation['is_running'] = True
        _operation['type'] = 'encode'
        _operation['start_time'] = datetime.now().isoformat()
        _operation['interrupt'] = False
    
    try:
        data = request.get_json()
        
        if not data:
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
        
        filename = data.get('filename')
        message = data.get('message')
        
        if not filename or not message:
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '缺少文件名或消息'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        if check_interrupt():
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '操作已被中断', 'interrupted': True}), 499
        
        model = get_model()
        
        output_filename = f"encoded_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        model.encode(filepath, output_path, message)
        
        with Image.open(output_path) as img:
            out_width, out_height = img.size
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        file_size = os.path.getsize(output_path)
        encode_time = time.time() - start_time
        
        _task_history.append({
            'type': 'encode',
            'filename': filename,
            'message_length': len(message),
            'encode_time': encode_time,
            'timestamp': datetime.now().isoformat()
        })
        
        with _operation['lock']:
            _operation['is_running'] = False
        
        return jsonify({
            'success': True,
            'message': '消息嵌入成功',
            'filename': output_filename,
            'image': f'data:image/png;base64,{img_base64}',
            'file_size': file_size,
            'file_size_text': f'{file_size / 1024:.2f} KB',
            'encode_time': f'{encode_time:.3f}s',
            'message_length': len(message),
            'dimensions': {'width': out_width, 'height': out_height},
            'download_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        with _operation['lock']:
            _operation['is_running'] = False
        return jsonify({'success': False, 'error': f'编码失败: {str(e)}'}), 500


@app.route('/api/decode', methods=['POST'])
def api_decode():
    start_time = time.time()
    
    with _operation['lock']:
        _operation['is_running'] = True
        _operation['type'] = 'decode'
        _operation['start_time'] = datetime.now().isoformat()
        _operation['interrupt'] = False
    
    try:
        data = request.get_json()
        
        if not data:
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
        
        filename = data.get('filename')
        
        if not filename:
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '缺少文件名'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        if check_interrupt():
            with _operation['lock']:
                _operation['is_running'] = False
            return jsonify({'success': False, 'error': '操作已被中断', 'interrupted': True}), 499
        
        model = get_model()
        decoded_message = model.decode(filepath)
        
        decode_time = time.time() - start_time
        
        _task_history.append({
            'type': 'decode',
            'filename': filename,
            'message_length': len(decoded_message) if decoded_message else 0,
            'decode_time': decode_time,
            'timestamp': datetime.now().isoformat()
        })
        
        with _operation['lock']:
            _operation['is_running'] = False
        
        return jsonify({
            'success': True,
            'message': '消息提取成功',
            'decoded_message': decoded_message,
            'message_length': len(decoded_message) if decoded_message else 0,
            'decode_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'decode_time_seconds': round(decode_time, 3),
            'filename': filename
        })
        
    except Exception as e:
        with _operation['lock']:
            _operation['is_running'] = False
        return jsonify({'success': False, 'error': f'解码失败: {str(e)}'}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def api_download(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename, mimetype='image/png')
    except Exception as e:
        return jsonify({'success': False, 'error': f'下载失败: {str(e)}'}), 500


@app.route('/api/operation/status', methods=['GET'])
def api_operation_status():
    with _operation['lock']:
        return jsonify({
            'success': True,
            'is_running': _operation['is_running'],
            'type': _operation['type'],
            'start_time': _operation['start_time'],
            'interrupt_requested': _operation['interrupt']
        })


@app.route('/api/operation/interrupt', methods=['POST'])
def api_operation_interrupt():
    with _operation['lock']:
        if not _operation['is_running']:
            return jsonify({'success': False, 'error': '当前没有正在运行的操作'}), 400
        
        _operation['interrupt'] = True
        return jsonify({
            'success': True,
            'message': '中断请求已发送'
        })


@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify({
        'success': True,
        'history': _task_history[-20:]
    })


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print("  ImgStegGAN Web Interface")
    print("  GAN-Driven Image Steganography with Qwen Enhancement")
    print("=" * 60)
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"访问地址: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
