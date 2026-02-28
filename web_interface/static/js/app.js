/**
 * SteganoGAN Web Interface - JavaScript 应用逻辑
 * 支持文件拖拽和隐写信息嵌入的可视化Web界面
 * 版本: V1.3.0
 */

// ============================================================================
// 版本信息
// ============================================================================
const VERSION = 'V1.0.0';
const VERSION_DATE = '2026-02-28';

// ============================================================================
// 全局状态管理
// ============================================================================
const AppState = {
    currentTab: 'encode',
    uploadedFiles: [],
    batchFiles: [],
    batchDownloadLinks: [], // 批量下载链接
    history: [],
    maxCapacity: 0,
    isProcessing: false,
    version: VERSION,
    
    // 任务管理相关
    currentTask: null,          // 当前任务
    taskHistory: [],            // 任务历史
    interruptRequested: false,  // 是否请求中断
    
    // 历史记录筛选状态
    historyFilter: {
        type: 'all',            // 类型筛选: all, encode, decode, batch_encode
        status: 'all',          // 状态筛选: all, completed, interrupted, running
        search: '',             // 搜索关键词
        sort: 'newest'          // 排序: newest, oldest
    }
};

// ============================================================================
// DOM 元素引用 - 延迟初始化模式，确保DOM完全加载后再获取
// ============================================================================
let elements = null;

async function initElements() {
    if (document.readyState !== 'complete' && document.readyState !== 'interactive') {
        await new Promise((resolve) => {
            document.addEventListener('DOMContentLoaded', resolve, { once: true });
        });
    }
    
    elements = {
        // 导航按钮
        navButtons: document.querySelectorAll('.nav-btn'),
        
        // 标签内容
        tabContents: {
            encode: document.getElementById('tab-encode'),
            decode: document.getElementById('tab-decode'),
            batch: document.getElementById('tab-batch')
        },
        
        // 嵌入相关
        dropZoneEncode: document.getElementById('drop-zone-encode'),
        fileInputEncode: document.getElementById('file-input-encode'),
        fileListEncode: document.getElementById('file-list-encode'),
        messageInput: document.getElementById('message-input'),
        capacityBadge: document.getElementById('capacity-badge'),
        btnEncode: document.getElementById('btn-encode'),
        btnClearEncode: document.getElementById('btn-clear-encode'),
        charCount: document.getElementById('char-count'),
        capacityWarning: document.getElementById('capacity-warning'),
        resultPanelEncode: document.getElementById('result-panel-encode'),
        originalPreview: document.getElementById('original-preview'),
        encodedPreview: document.getElementById('encoded-preview'),
        resultFileSize: document.getElementById('result-file-size'),
        resultEncodeTime: document.getElementById('result-encode-time'),
        resultMessageLength: document.getElementById('result-message-length'),
        btnDownload: document.getElementById('btn-download'),
        btnNewEncode: document.getElementById('btn-new-encode'),
        
        // 解码相关
        dropZoneDecode: document.getElementById('drop-zone-decode'),
        fileInputDecode: document.getElementById('file-input-decode'),
        fileInfoDecode: document.getElementById('file-info-decode'),
        decodePreview: document.getElementById('decode-preview'),
        decodeFilename: document.getElementById('decode-filename'),
        decodeDimensions: document.getElementById('decode-dimensions'),
        decodeFilesize: document.getElementById('decode-filesize'),
        btnDecode: document.getElementById('btn-decode'),
        btnClearDecode: document.getElementById('btn-clear-decode'),
        resultPanelDecode: document.getElementById('result-panel-decode'),
        decodedMessage: document.getElementById('decoded-message'),
        decodedMessageLength: document.getElementById('decoded-message-length'),
        decodedTime: document.getElementById('decoded-time'),
        btnCopyMessage: document.getElementById('btn-copy-message'),
        btnNewDecode: document.getElementById('btn-new-decode'),
        
        // 批量嵌入相关
        dropZoneBatch: document.getElementById('drop-zone-batch'),
        fileInputBatch: document.getElementById('file-input-batch'),
        batchFileList: document.getElementById('batch-file-list'),
        batchMessageInput: document.getElementById('batch-message-input'),
        batchProgress: document.getElementById('batch-progress'),
        batchProgressText: document.getElementById('batch-progress-text'),
        batchProgressFill: document.getElementById('batch-progress-fill'),
        btnBatchEncode: document.getElementById('btn-batch-encode'),
        btnClearBatch: document.getElementById('btn-clear-batch'),
        batchResultPanel: document.getElementById('batch-result-panel'),
        batchSuccessCount: document.getElementById('batch-success-count'),
        batchFailCount: document.getElementById('batch-fail-count'),
        batchTime: document.getElementById('batch-time'),
        batchFilesResult: document.getElementById('batch-files-result'),
        btnDownloadAllBatch: document.getElementById('btn-download-all-batch'),
        btnPackageDownload: document.getElementById('btn-package-download'),
        downloadCountBadge: document.getElementById('download-count-badge'),
        downloadHintText: document.getElementById('download-hint-text'),
        btnNewBatch: document.getElementById('btn-new-batch'),
        
        // 全局元素
        historyList: document.getElementById('history-list'),
        clearHistory: document.getElementById('clear-history'),
        viewAllHistory: document.getElementById('view-all-history'),
        themeToggle: document.getElementById('theme-toggle'),
        helpBtn: document.getElementById('help-btn'),
        helpModal: document.getElementById('help-modal'),
        modalClose: document.getElementById('modal-close'),
        statusText: document.getElementById('status-text'),
        connectionStatus: document.getElementById('connection-status'),
        globalProgress: document.getElementById('global-progress'),
        globalProgressFill: document.getElementById('global-progress-fill'),
        notificationContainer: document.getElementById('notification-container'),
        
        // 历史记录模态框元素
        historyModal: document.getElementById('history-modal'),
        historyModalClose: document.getElementById('history-modal-close'),
        historyModalBackBtn: document.getElementById('history-modal-back-btn'),
        historyModalList: document.getElementById('history-modal-list'),
        historyTotalCount: document.getElementById('history-total-count'),
        historySearchInput: document.getElementById('history-search-input'),
        historySortSelect: document.getElementById('history-sort-select'),
        clearAllHistory: document.getElementById('clear-all-history')
    };
    
    const criticalElements = ['messageInput', 'btnEncode', 'dropZoneEncode', 'fileInputEncode'];
    for (const key of criticalElements) {
        if (!elements[key]) {
            console.error(`关键元素缺失: ${key}`);
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// API 客户端 - 增强版，带错误处理和重试机制
// ============================================================================
const API = {
    baseUrl: '/api',
    maxRetries: 3,
    retryDelay: 1000,
    timeout: 30000, // 30秒超时
    
    // 带超时的 fetch 包装器
    async fetchWithTimeout(url, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('请求超时，请稍后重试');
            }
            throw error;
        }
    },
    
    // 带重试的请求
    async requestWithRetry(url, options = {}, retryCount = 0) {
        try {
            const response = await this.fetchWithTimeout(url, options);
            
            // 检查响应状态
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const error = new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
                
                // 传递中断标志和额外数据
                if (errorData.interrupted) {
                    error.interrupted = true;
                }
                if (errorData.results) {
                    error.results = errorData.results;
                }
                if (errorData.total_time) {
                    error.total_time = errorData.total_time;
                }
                
                throw error;
            }
            
            return await response.json();
        } catch (error) {
            if (retryCount < this.maxRetries && this.isRetryableError(error)) {
                console.warn(`请求失败，${retryCount + 1}/${this.maxRetries} 次重试...`, error.message);
                await this.delay(this.retryDelay * (retryCount + 1)); // 指数退避
                return this.requestWithRetry(url, options, retryCount + 1);
            }
            throw error;
        }
    },
    
    // 判断错误是否可重试
    isRetryableError(error) {
        const retryableErrors = [
            'NetworkError',
            'TypeError', // 网络错误通常表现为 TypeError
            'Failed to fetch',
            '请求超时',
            'timeout'
        ];
        return retryableErrors.some(e => error.message?.includes(e));
    },
    
    // 延迟函数
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },
    
    // 检查网络连接
    async checkConnection() {
        try {
            await this.fetchWithTimeout(`${this.baseUrl}/models`, { method: 'GET' });
            return true;
        } catch {
            return false;
        }
    },
    
    async uploadFile(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);
        
        // 使用 XMLHttpRequest 来支持进度监控
        if (onProgress && typeof onProgress === 'function') {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const progress = (e.loaded / e.total) * 100;
                        onProgress(progress);
                    }
                });
                
                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            resolve(JSON.parse(xhr.responseText));
                        } catch (e) {
                            reject(new Error('解析响应失败'));
                        }
                    } else {
                        reject(new Error(`上传失败: ${xhr.statusText}`));
                    }
                });
                
                xhr.addEventListener('error', () => reject(new Error('网络错误')));
                xhr.addEventListener('timeout', () => reject(new Error('上传超时')));
                xhr.addEventListener('abort', () => reject(new Error('上传被取消')));
                
                xhr.open('POST', `${this.baseUrl}/upload`);
                xhr.timeout = this.timeout;
                xhr.send(formData);
            });
        }
        
        // 普通 fetch 方式
        return this.requestWithRetry(`${this.baseUrl}/upload`, {
            method: 'POST',
            body: formData
        });
    },
    
    async encodeMessage(filename, message) {
        return this.requestWithRetry(`${this.baseUrl}/encode`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename,
                message
            })
        });
    },
    
    async decodeMessage(filename) {
        return this.requestWithRetry(`${this.baseUrl}/decode`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename
            })
        });
    },
    
    getDownloadUrl(filename) {
        return `${this.baseUrl}/download/${filename}`;
    },
    
    async cleanupFiles(filenames) {
        return this.requestWithRetry(`${this.baseUrl}/cleanup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filenames })
        });
    },
    
    async batchEncode(filenames, message) {
        return this.requestWithRetry(`${this.baseUrl}/batch_encode`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filenames,
                message
            })
        });
    },
    
    // 任务管理 API
    async getTasks(limit = 50) {
        return this.requestWithRetry(`${this.baseUrl}/tasks?limit=${limit}`, {
            method: 'GET'
        });
    },
    
    async getTask(taskId) {
        return this.requestWithRetry(`${this.baseUrl}/tasks/${taskId}`, {
            method: 'GET'
        });
    },
    
    async interruptTask(taskId) {
        return this.requestWithRetry(`${this.baseUrl}/tasks/${taskId}/interrupt`, {
            method: 'POST'
        });
    },
    
    async resumeTask(taskId) {
        return this.requestWithRetry(`${this.baseUrl}/tasks/${taskId}/resume`, {
            method: 'POST'
        });
    },
    
    async deleteTask(taskId) {
        return this.requestWithRetry(`${this.baseUrl}/tasks/${taskId}`, {
            method: 'DELETE'
        });
    },
    
    async getInterruptedTasks() {
        return this.requestWithRetry(`${this.baseUrl}/tasks/interrupted`, {
            method: 'GET'
        });
    },
    
    // 操作中断 API
    async getOperationStatus() {
        return this.requestWithRetry(`${this.baseUrl}/operation/status`, {
            method: 'GET'
        });
    },
    
    async interruptCurrentOperation() {
        return this.requestWithRetry(`${this.baseUrl}/operation/interrupt`, {
            method: 'POST'
        });
    }
};

// ============================================================================
// 通知系统 - 增强版
// ============================================================================
const NotificationManager = {
    container: null,
    notifications: [],
    maxNotifications: 5,
    
    init() {
        this.container = document.getElementById('notification-container');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'notification-container';
            this.container.className = 'notification-container';
            document.body.appendChild(this.container);
        }
    },
    
    show(message, type = 'info', duration = 5000, actions = []) {
        this.init();
        
        // 限制通知数量
        if (this.notifications.length >= this.maxNotifications) {
            this.notifications[0].remove();
            this.notifications.shift();
        }
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle',
            loading: 'fa-spinner fa-spin'
        };
        
        const titles = {
            success: '成功',
            error: '错误',
            warning: '警告',
            info: '提示',
            loading: '处理中'
        };
        
        const actionButtons = actions.map(action => 
            `<button class="btn btn-sm btn-${action.type || 'secondary'}" onclick="${action.onClick}">${action.text}</button>`
        ).join('');
        
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas ${icons[type]}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${titles[type]}</div>
                <div class="notification-message">${message}</div>
                ${actionButtons ? `<div class="notification-actions" style="margin-top: 8px; display: flex; gap: 8px;">${actionButtons}</div>` : ''}
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // 关闭按钮事件
        notification.querySelector('.notification-close').addEventListener('click', () => {
            this.remove(notification);
        });
        
        this.container.appendChild(notification);
        this.notifications.push(notification);
        
        // 自动移除
        if (duration > 0) {
            setTimeout(() => {
                this.remove(notification);
            }, duration);
        }
        
        return notification;
    },
    
    remove(notification) {
        if (!notification || notification.classList.contains('removing')) return;
        
        notification.classList.add('removing');
        notification.style.animation = 'slideOut 0.3s ease forwards';
        
        setTimeout(() => {
            notification.remove();
            const index = this.notifications.indexOf(notification);
            if (index > -1) {
                this.notifications.splice(index, 1);
            }
        }, 300);
    },
    
    success(message, duration) {
        return this.show(message, 'success', duration);
    },
    
    error(message, duration) {
        return this.show(message, 'error', duration || 8000);
    },
    
    warning(message, duration) {
        return this.show(message, 'warning', duration);
    },
    
    info(message, duration) {
        return this.show(message, 'info', duration);
    },
    
    loading(message) {
        return this.show(message, 'loading', 0);
    },
    
    updateLoading(notification, message, type = 'success') {
        if (!notification) return;
        
        const icon = notification.querySelector('.notification-icon i');
        const title = notification.querySelector('.notification-title');
        const content = notification.querySelector('.notification-message');
        
        icon.className = `fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}`;
        title.textContent = type === 'success' ? '成功' : '错误';
        content.textContent = message;
        notification.className = `notification ${type}`;
        
        // 3秒后自动移除
        setTimeout(() => this.remove(notification), 3000);
    },
    
    clearAll() {
        this.notifications.forEach(n => this.remove(n));
    }
};

// 向后兼容的简写函数
function showNotification(message, type = 'info') {
    return NotificationManager.show(message, type);
}

// ============================================================================
// 拖拽上传功能 - 增强版（支持文件夹）
// ============================================================================
function setupDragAndDrop(dropZone, fileInput, onFileSelect) {
    let dragCounter = 0;
    let isProcessingFolder = false;

    // 递归遍历文件夹获取所有文件
    async function traverseFileTree(item, path = '') {
        const files = [];

        if (item.isFile) {
            // 是文件，直接返回
            return new Promise((resolve) => {
                item.file((file) => {
                    // 检查是否是图像文件
                    if (file.type.startsWith('image/')) {
                        files.push(file);
                    }
                    resolve(files);
                });
            });
        } else if (item.isDirectory) {
            // 是文件夹，递归读取
            const dirReader = item.createReader();
            const entries = await new Promise((resolve) => {
                dirReader.readEntries(resolve);
            });

            for (const entry of entries) {
                const subFiles = await traverseFileTree(entry, path + item.name + '/');
                files.push(...subFiles);
            }
        }

        return files;
    }

    // 从 DataTransferItemList 获取所有文件（包括文件夹内的文件）
    async function getFilesFromDataTransfer(dataTransfer) {
        const files = [];
        const items = dataTransfer.items;

        if (!items) {
            // 不支持 items API，回退到 files
            return Array.from(dataTransfer.files);
        }

        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === 'file') {
                const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
                if (entry) {
                    const entryFiles = await traverseFileTree(entry);
                    files.push(...entryFiles);
                } else {
                    // 回退：直接获取文件
                    const file = item.getAsFile();
                    if (file) files.push(file);
                }
            }
        }

        return files;
    }

    // 点击上传
    dropZone.addEventListener('click', (e) => {
        // 如果点击的是移除按钮，不触发文件选择
        if (e.target.closest('.file-item-remove')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            // 过滤只保留图像文件
            const imageFiles = files.filter(file => file.type.startsWith('image/'));
            if (imageFiles.length > 0) {
                onFileSelect(imageFiles);
            } else {
                NotificationManager.warning('未找到图像文件，请上传 PNG、JPG、JPEG 或 BMP 格式的文件');
            }
        }
        fileInput.value = ''; // 重置input
    });

    // 拖拽事件
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // 拖拽进入
    dropZone.addEventListener('dragenter', (e) => {
        dragCounter++;
        dropZone.classList.add('dragover');

        // 显示文件数量提示
        const items = e.dataTransfer.items;
        if (items) {
            const count = items.length;
            const overlay = dropZone.querySelector('.drop-zone-overlay span');
            if (overlay) {
                overlay.textContent = `释放以上传 ${count} 个项目（支持文件夹）`;
            }
        }
    }, false);

    // 拖拽悬停
    dropZone.addEventListener('dragover', (e) => {
        dropZone.classList.add('dragover');
        e.dataTransfer.dropEffect = 'copy';
    }, false);

    // 拖拽离开
    dropZone.addEventListener('dragleave', () => {
        dragCounter--;
        if (dragCounter === 0) {
            dropZone.classList.remove('dragover');
        }
    }, false);

    // 放置文件 - 支持文件夹
    dropZone.addEventListener('drop', async (e) => {
        if (isProcessingFolder) return;

        dragCounter = 0;
        dropZone.classList.remove('dragover');

        // 添加放置动画效果
        dropZone.style.transform = 'scale(0.98)';
        setTimeout(() => {
            dropZone.style.transform = '';
        }, 150);

        isProcessingFolder = true;

        try {
            // 显示加载提示
            const loadingNotification = NotificationManager.loading('正在扫描文件和文件夹...');

            // 获取所有文件（包括文件夹内的）
            const files = await getFilesFromDataTransfer(e.dataTransfer);

            NotificationManager.remove(loadingNotification);

            if (files.length === 0) {
                NotificationManager.warning('未找到可上传的文件');
                isProcessingFolder = false;
                return;
            }

            // 过滤只保留图像文件
            const imageFiles = files.filter(file => file.type.startsWith('image/'));

            if (imageFiles.length === 0) {
                NotificationManager.warning('未找到图像文件，请上传 PNG、JPG、JPEG 或 BMP 格式的文件');
                isProcessingFolder = false;
                return;
            }

            // 统计信息
            const folderCount = e.dataTransfer.items ?
                Array.from(e.dataTransfer.items).filter(item => {
                    const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
                    return entry && entry.isDirectory;
                }).length : 0;

            if (folderCount > 0) {
                NotificationManager.info(`成功从 ${folderCount} 个文件夹中扫描到 ${imageFiles.length} 个图像文件`);
            }

            // 调用回调处理文件
            onFileSelect(imageFiles);
        } catch (error) {
            console.error('处理拖拽文件时出错:', error);
            NotificationManager.error('处理文件时出错: ' + error.message);
        } finally {
            isProcessingFolder = false;
        }
    }, false);

    // 粘贴上传支持
    document.addEventListener('paste', (e) => {
        if (!dropZone.closest('.tab-content.active')) return;

        const items = e.clipboardData.items;
        const files = [];

        for (let i = 0; i < items.length; i++) {
            if (items[i].kind === 'file') {
                const file = items[i].getAsFile();
                if (file && file.type.startsWith('image/')) {
                    files.push(file);
                }
            }
        }

        if (files.length > 0) {
            onFileSelect(files);
            NotificationManager.info(`从剪贴板粘贴了 ${files.length} 个图像`);
        }
    });
}

// ============================================================================
// 文件处理
// ============================================================================
function validateFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        showNotification('不支持的文件类型。请上传 PNG、JPG、JPEG 或 BMP 格式的图像。', 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('文件过大。最大允许大小为 16MB。', 'error');
        return false;
    }
    
    return true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// ============================================================================
// 文件列表渲染 - 增强版
// ============================================================================
function renderFileItem(file, options = {}) {
    const item = document.createElement('div');
    item.className = 'file-item card-hover';
    item.dataset.filename = file.name;
    
    // 添加加载状态
    const isLoading = options.loading || false;
    
    item.innerHTML = `
        <div class="file-item-preview">
            ${isLoading 
                ? '<div class="skeleton" style="width: 100%; height: 100%;"></div>'
                : `<img src="${file.preview}" alt="${file.name}" loading="lazy">`
            }
        </div>
        <div class="file-item-info">
            <div class="file-item-name" title="${file.name}">${file.name}</div>
            <div class="file-item-size">${formatFileSize(file.size)}</div>
            ${options.showDimensions ? `<div class="file-item-dimensions">${options.dimensions}</div>` : ''}
        </div>
        <button class="file-item-remove magnetic-btn" title="移除文件">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // 添加进入动画
    item.style.animationDelay = `${options.index * 0.05}s`;
    
    // 移除按钮事件
    const removeBtn = item.querySelector('.file-item-remove');
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        
        // 添加移除动画
        item.style.transform = 'translateX(100%)';
        item.style.opacity = '0';
        
        setTimeout(() => {
            item.remove();
            AppState.uploadedFiles = AppState.uploadedFiles.filter(f => f.name !== file.name);
            updateEncodeUI();
            
            // 如果没有文件了，重置容量
            if (AppState.uploadedFiles.length === 0) {
                AppState.maxCapacity = 0;
            }
        }, 300);
    });
    
    // 点击预览
    item.addEventListener('click', () => {
        showImagePreview(file.preview, file.name);
    });
    
    return item;
}

function renderBatchFileItem(file, index) {
    const item = document.createElement('div');
    item.className = 'file-item batch-file-item';
    item.dataset.index = index;
    
    item.innerHTML = `
        <div class="file-item-icon">
            <i class="fas fa-image"></i>
        </div>
        <div class="file-item-info">
            <div class="file-item-name" title="${file.name}">${file.name}</div>
            <div class="file-item-size">${formatFileSize(file.size)}</div>
        </div>
        <div class="batch-file-status">
            <i class="fas fa-circle" style="color: var(--text-muted); font-size: 8px;"></i>
        </div>
        <button class="file-item-remove magnetic-btn" title="移除文件">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // 移除按钮事件
    item.querySelector('.file-item-remove').addEventListener('click', (e) => {
        e.stopPropagation();
        item.style.transform = 'translateX(100%)';
        item.style.opacity = '0';
        
        setTimeout(() => {
            item.remove();
            AppState.batchFiles = AppState.batchFiles.filter((_, i) => i !== index);
            updateBatchUI();
        }, 300);
    });
    
    return item;
}

// 图片预览模态框
function showImagePreview(src, title) {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 90vw; max-height: 90vh; background: transparent; box-shadow: none;">
            <div style="position: relative; display: inline-block;">
                <img src="${src}" alt="${title}" style="max-width: 100%; max-height: 80vh; border-radius: var(--radius-lg); box-shadow: var(--shadow-xl);">
                <button class="modal-close" style="position: absolute; top: -40px; right: 0; background: rgba(0,0,0,0.5); color: white;">
                    <i class="fas fa-times"></i>
                </button>
                <div style="position: absolute; bottom: -30px; left: 0; right: 0; text-align: center; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">
                    ${title}
                </div>
            </div>
        </div>
    `;
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    modal.querySelector('.modal-close').addEventListener('click', () => {
        modal.remove();
    });
    
    document.body.appendChild(modal);
}

// ============================================================================
// 嵌入功能 - 增强版（仅支持单张图片）
// ============================================================================

// 单张图片替换确认弹窗相关元素
let replaceImageModal = null;
let pendingReplaceFiles = [];

function initReplaceImageModal() {
    replaceImageModal = document.getElementById('replace-image-modal');
    const btnReplace = document.getElementById('btn-replace-image');
    const btnCancel = document.getElementById('btn-cancel-replace');
    
    if (btnReplace) {
        btnReplace.addEventListener('click', () => {
            closeReplaceImageModal();
            // 使用第一张图片替换
            if (pendingReplaceFiles.length > 0) {
                performReplaceImage(pendingReplaceFiles[0]);
            }
        });
    }
    
    if (btnCancel) {
        btnCancel.addEventListener('click', () => {
            closeReplaceImageModal();
            pendingReplaceFiles = [];
        });
    }
    
    // 点击背景关闭
    if (replaceImageModal) {
        replaceImageModal.addEventListener('click', (e) => {
            if (e.target === replaceImageModal) {
                closeReplaceImageModal();
                pendingReplaceFiles = [];
            }
        });
    }
}

function openReplaceImageModal(fileCount) {
    if (!replaceImageModal) {
        initReplaceImageModal();
    }
    
    const countElement = document.getElementById('selected-file-count');
    if (countElement) {
        countElement.textContent = fileCount;
    }
    
    replaceImageModal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeReplaceImageModal() {
    if (replaceImageModal) {
        replaceImageModal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

async function performReplaceImage(file) {
    // 清除现有文件
    AppState.uploadedFiles = [];
    elements.fileListEncode.innerHTML = '';
    AppState.maxCapacity = 0;
    
    // 上传新文件
    const loadingNotification = NotificationManager.loading('正在上传文件...');
    
    try {
        const result = await API.uploadFile(file);
        
        if (result.success) {
            AppState.uploadedFiles.push({
                name: result.filename,
                originalName: file.name,
                size: file.size,
                preview: result.preview,
                info: result.info
            });
            
            elements.fileListEncode.appendChild(renderFileItem({
                name: result.filename,
                size: file.size,
                preview: result.preview
            }));
            
            AppState.maxCapacity = result.info.max_capacity;
            
            NotificationManager.remove(loadingNotification);
            
            // 检查尺寸警告
            if (result.info.size_warning) {
                if (!result.info.is_valid_for_steganography) {
                    // 尺寸过小，无法使用
                    NotificationManager.error(result.info.size_warning, 8000);
                    elements.capacityBadge.textContent = '容量: 不足';
                    elements.capacityWarning.textContent = '图片尺寸过小，无法嵌入消息';
                    elements.capacityWarning.style.color = 'var(--error-color)';
                } else {
                    // 尺寸较小但有警告
                    NotificationManager.warning(result.info.size_warning, 5000);
                    NotificationManager.success('文件上传成功');
                }
            } else {
                NotificationManager.success('文件上传成功');
            }
            
            updateEncodeUI();
        }
    } catch (error) {
        console.error('上传失败:', error);
        NotificationManager.remove(loadingNotification);
        NotificationManager.error(`上传失败: ${error.message}`, 5000);
    }
}

async function handleEncodeFiles(files) {
    const validFiles = files.filter(validateFile);
    
    if (validFiles.length === 0) return;
    
    // 单张图片限制：如果已存在文件，显示替换确认弹窗
    if (AppState.uploadedFiles.length > 0) {
        pendingReplaceFiles = validFiles;
        openReplaceImageModal(validFiles.length);
        return;
    }
    
    // 单张图片限制：如果选择了多张，只使用第一张
    if (validFiles.length > 1) {
        pendingReplaceFiles = validFiles;
        openReplaceImageModal(validFiles.length);
        return;
    }
    
    // 正常上传单张图片
    await performReplaceImage(validFiles[0]);
}

function updateEncodeUI() {
    // 安全检查：确保关键元素存在
    if (!elements || !elements.messageInput || !elements.btnEncode) {
        console.warn('updateEncodeUI: 关键元素未初始化');
        return;
    }
    
    const hasFiles = AppState.uploadedFiles.length > 0;
    const hasValidSize = AppState.maxCapacity > 0;
    const messageValue = elements.messageInput.value || '';
    const hasMessage = messageValue.trim().length > 0;
    
    elements.btnEncode.disabled = !hasFiles || !hasValidSize || !hasMessage;
    
    // 更新容量信息
    if (AppState.maxCapacity > 0) {
        const messageLength = messageValue.length;
        const capacityUsed = messageLength / AppState.maxCapacity * 100;
        
        elements.capacityBadge.textContent = `容量: ${formatFileSize(AppState.maxCapacity)}`;
        
        if (capacityUsed > 100) {
            elements.capacityWarning.textContent = '消息超出容量限制';
            elements.capacityWarning.style.color = 'var(--error-color)';
        } else if (capacityUsed > 80) {
            elements.capacityWarning.textContent = `已使用 ${capacityUsed.toFixed(1)}%`;
            elements.capacityWarning.style.color = 'var(--warning-color)';
        } else {
            elements.capacityWarning.textContent = '';
        }
    } else if (hasFiles) {
        // 有文件但容量为0，说明尺寸过小
        elements.capacityBadge.textContent = '容量: 不足';
        elements.capacityWarning.textContent = '图片尺寸过小，无法嵌入消息';
        elements.capacityWarning.style.color = 'var(--error-color)';
    } else {
        elements.capacityBadge.textContent = '容量: --';
        elements.capacityWarning.textContent = '';
    }
    
    // 更新字符计数
    if (elements.charCount) {
        elements.charCount.textContent = `${messageValue.length} 字符`;
    }
}

async function encodeMessage() {
    if (AppState.uploadedFiles.length === 0) {
        NotificationManager.warning('请先上传图像文件');
        elements.dropZoneEncode.classList.add('error-shake');
        setTimeout(() => elements.dropZoneEncode.classList.remove('error-shake'), 500);
        return;
    }
    
    if (!elements.messageInput) {
        NotificationManager.error('系统错误：消息输入框未找到，请刷新页面重试');
        return;
    }
    
    const message = elements.messageInput.value.trim();
    
    if (!message) {
        NotificationManager.warning('请输入要嵌入的消息');
        elements.messageInput.focus();
        elements.messageInput.classList.add('error-shake');
        setTimeout(() => elements.messageInput.classList.remove('error-shake'), 500);
        return;
    }
    
    if (message.length > AppState.maxCapacity) {
        NotificationManager.error(
            `消息超出图像容量限制！当前: ${message.length} 字符, 最大: ${AppState.maxCapacity} 字符`
        );
        return;
    }
    
    // 检查网络连接
    const isConnected = await API.checkConnection();
    if (!isConnected) {
        NotificationManager.error('无法连接到服务器，请检查网络连接');
        return;
    }
    
    AppState.isProcessing = true;
    setProcessingState(true);
    
    const loadingNotification = NotificationManager.loading('正在嵌入消息，请稍候...');
    
    try {
        const file = AppState.uploadedFiles[0];
        const result = await API.encodeMessage(
            file.name,
            message
        );
        
        if (result.success) {
            // 显示结果
            elements.originalPreview.innerHTML = `<img src="${file.preview}" alt="原始图像">`;
            elements.encodedPreview.innerHTML = `<img src="${result.image}" alt="隐写图像">`;
            elements.resultFileSize.textContent = result.file_size_text;
            elements.resultEncodeTime.textContent = result.encode_time;
            elements.resultMessageLength.textContent = `${result.message_length} 字符`;
            elements.btnDownload.href = result.download_url;
            
            // 滚动到结果区域
            elements.resultPanelEncode.style.display = 'block';
            elements.resultPanelEncode.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            // 添加到历史记录
            addToHistory('encode', {
                filename: file.originalName,
                message: message.substring(0, 50) + (message.length > 50 ? '...' : ''),
                time: new Date().toLocaleString()
            });
            
            NotificationManager.updateLoading(
                loadingNotification,
                `消息嵌入成功！耗时 ${result.encode_time}`,
                'success'
            );
        } else {
            throw new Error(result.error || '嵌入失败');
        }
    } catch (error) {
        console.error('编码错误:', error);
        NotificationManager.remove(loadingNotification);
        
        // 检查是否为中断错误
        if (error.interrupted || error.message?.includes('中断')) {
            NotificationManager.warning('操作已被中断');
            // 刷新历史记录以显示中断的任务
            await loadTaskHistory();
            return;
        }
        
        // 提供更详细的错误信息
        let errorMsg = error.message;
        if (error.message.includes('模型加载失败')) {
            errorMsg = '模型加载失败，请检查模型文件是否完整';
        } else if (error.message.includes('timeout')) {
            errorMsg = '处理超时，请尝试使用较小的图像或消息';
        } else if (error.message.includes('CUDA')) {
            errorMsg = 'GPU加速出错，请关闭GPU选项后重试';
        }
        
        NotificationManager.error(`嵌入失败: ${errorMsg}`, 8000, [
            {
                text: '重试',
                type: 'primary',
                onClick: 'encodeMessage()'
            }
        ]);
    } finally {
        AppState.isProcessing = false;
        setProcessingState(false);
        AppState.interruptRequested = false;
        
        // 操作完成后刷新历史记录，确保状态正确显示
        loadTaskHistory();
    }
}

// ============================================================================
// 解码功能
// ============================================================================
async function handleDecodeFile(files) {
    // 解码功能只处理单个文件，取第一个
    const file = files[0];
    if (!file) {
        NotificationManager.warning('未找到有效的图像文件');
        return;
    }
    if (!validateFile(file)) return;
    
    const loadingNotification = NotificationManager.loading('正在上传文件...');
    
    try {
        const result = await API.uploadFile(file);
        
        if (result.success) {
            AppState.uploadedFiles = [{
                name: result.filename,
                originalName: file.name,
                size: file.size,
                preview: result.preview,
                info: result.info
            }];
            
            // 显示文件信息
            elements.fileInfoDecode.style.display = 'flex';
            elements.decodePreview.innerHTML = `<img src="${result.preview}" alt="${file.name}">`;
            elements.decodeFilename.textContent = file.name;
            elements.decodeDimensions.textContent = `${result.info.width} x ${result.info.height}`;
            elements.decodeFilesize.textContent = formatFileSize(file.size);
            
            NotificationManager.remove(loadingNotification);
            
            // 检查尺寸警告
            if (result.info.size_warning) {
                if (!result.info.is_valid_for_steganography) {
                    // 尺寸过小，无法解码
                    NotificationManager.error(result.info.size_warning, 8000);
                    elements.btnDecode.disabled = true;
                    
                    // 显示尺寸不足提示
                    const sizeHint = document.createElement('div');
                    sizeHint.className = 'size-warning-hint';
                    sizeHint.innerHTML = `
                        <i class="fas fa-exclamation-triangle" style="color: var(--error-color);"></i>
                        <span>图片尺寸不足，无法包含隐藏信息</span>
                    `;
                    const existingHint = elements.fileInfoDecode.querySelector('.size-warning-hint');
                    if (existingHint) existingHint.remove();
                    elements.fileInfoDecode.appendChild(sizeHint);
                } else {
                    // 尺寸较小但有警告
                    NotificationManager.warning(result.info.size_warning, 5000);
                    NotificationManager.success('文件上传成功');
                    elements.btnDecode.disabled = false;
                }
            } else {
                NotificationManager.success('文件上传成功');
                elements.btnDecode.disabled = false;
            }
        }
    } catch (error) {
        NotificationManager.remove(loadingNotification);
        NotificationManager.error(`上传失败: ${error.message}`);
    }
}

async function decodeMessage() {
    if (AppState.uploadedFiles.length === 0) {
        NotificationManager.warning('请先上传包含隐藏消息的图像');
        elements.dropZoneDecode.classList.add('error-shake');
        setTimeout(() => elements.dropZoneDecode.classList.remove('error-shake'), 500);
        return;
    }
    
    // 检查网络连接
    const isConnected = await API.checkConnection();
    if (!isConnected) {
        NotificationManager.error('无法连接到服务器，请检查网络连接');
        return;
    }
    
    AppState.isProcessing = true;
    setProcessingState(true);
    
    const file = AppState.uploadedFiles[0];
    
    const loadingNotification = NotificationManager.loading('正在提取隐藏消息...');
    
    try {
        const result = await API.decodeMessage(
            file.name
        );
        
        if (result.success) {
            elements.decodedMessage.value = result.decoded_message;
            elements.decodedMessageLength.textContent = `${result.message_length} 字符`;
            elements.decodedTime.textContent = result.decode_time + (result.decode_time_seconds ? ` (${result.decode_time_seconds}s)` : '');
            
            elements.resultPanelDecode.style.display = 'block';
            elements.resultPanelDecode.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            // 添加到历史记录
            addToHistory('decode', {
                filename: file.originalName,
                message: result.decoded_message.substring(0, 50) + (result.decoded_message.length > 50 ? '...' : ''),
                time: new Date().toLocaleString()
            });
            
            // 构建成功消息
            let successMsg = `成功提取 ${result.message_length} 个字符的消息`;
            if (result.decode_time_seconds) {
                successMsg += `，耗时 ${result.decode_time_seconds}秒`;
            }
            
            NotificationManager.updateLoading(loadingNotification, successMsg, 'success');
            
            // 如果消息为空，显示警告
            if (!result.decoded_message || result.decoded_message.trim() === '') {
                NotificationManager.warning('未检测到隐藏消息，请确认图像是否包含隐藏信息');
            }
        } else {
            throw result;
        }
    } catch (error) {
        console.error('解码错误:', error);
        NotificationManager.remove(loadingNotification);
        
        // 检查是否为中断错误
        if (error.interrupted || error.message?.includes('中断')) {
            NotificationManager.warning('操作已被中断');
            await loadTaskHistory();
            return;
        }
        
        // 检查是否为超时错误
        if (error.message?.includes('超时') || error.message?.includes('timeout') || error.error?.includes('超时')) {
            NotificationManager.error('处理超时，建议：\n1. 使用更小的图像\n2. 启用自动模型检测');
            elements.decodedMessage.value = '解码超时：处理时间过长已自动终止。\n\n建议：\n1. 使用尺寸较小的图像\n2. 确保图像包含有效的隐藏信息';
            AppState.isProcessing = false;
            setProcessingState(false);
            loadTaskHistory();
            return;
        }
        
        // 处理带有尝试记录的详细错误
        let errorMsg = error.error || error.message || '未知错误';
        let actions = [];
        
        // 如果有多个模型的尝试记录
        if (error.attempts && error.attempts.length > 0) {
            const failedModels = error.attempts.filter(a => !a.success);
            const attemptedModels = error.attempts.map(a => a.architecture).join(', ');
            
            errorMsg = `所有模型(${attemptedModels})均未能解码隐藏消息`;
            
            // 构建详细的错误报告
            let detailHtml = '<div class="decode-error-detail" style="margin-top: 10px; text-align: left; font-size: 12px;">';
            detailHtml += '<strong>尝试记录：</strong><br>';
            error.attempts.forEach(attempt => {
                const icon = attempt.success ? '✓' : '✗';
                const color = attempt.success ? '#10b981' : '#ef4444';
                detailHtml += `<span style="color: ${color}">${icon} ${attempt.architecture}</span> - `;
                if (attempt.success) {
                    detailHtml += `成功 (${attempt.time.toFixed(1)}s)`;
                } else {
                    detailHtml += `失败 (${attempt.time.toFixed(1)}s)`;
                }
                detailHtml += '<br>';
            });
            detailHtml += '</div>';
            
            elements.decodedMessage.value = `解码失败：${errorMsg}\n\n建议：\n${(error.suggestions || []).join('\n')}`;
        } else if (error.message?.includes('模型加载失败')) {
            errorMsg = '模型加载失败，请检查模型文件是否完整';
            actions = [
                { text: '刷新页面', type: 'primary', onClick: 'location.reload()' }
            ];
        } else if (error.message?.includes('timeout') || error.message?.includes('超时')) {
            errorMsg = '处理超时，请稍后重试或尝试使用较小的图像';
        } else if (error.message?.includes('架构') || error.message?.includes('model')) {
            errorMsg = '可能使用了错误的模型架构，建议启用自动模型检测';
        }
        
        // 显示建议
        if (error.suggestions && error.suggestions.length > 0) {
            actions.push({
                text: '查看解决方案',
                type: 'secondary',
                onClick: `showDecodeSuggestions(${JSON.stringify(error.suggestions).replace(/"/g, '&quot;')})`
            });
        }
        
        NotificationManager.error(`提取失败: ${errorMsg}`, 10000, actions);
        
        // 如果有尝试记录，显示更详细的信息
        if (!error.attempts) {
            elements.decodedMessage.value = '提取失败：' + errorMsg;
        }
    } finally {
        AppState.isProcessing = false;
        setProcessingState(false);
        AppState.interruptRequested = false;
        
        // 操作完成后刷新历史记录，确保状态正确显示
        loadTaskHistory();
    }
}

// 显示解码建议模态框
function showDecodeSuggestions(suggestions) {
    const suggestionHtml = suggestions.map((s, i) => `<li>${s}</li>`).join('');
    
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <div class="modal-header">
                <h3><i class="fas fa-lightbulb"></i> 解码失败 - 解决方案建议</h3>
            </div>
            <div class="modal-body" style="text-align: left;">
                <p>无法从图像中提取隐藏消息，可能的原因和解决方案：</p>
                <ol style="padding-left: 20px; line-height: 2;">
                    ${suggestionHtml}
                </ol>
                <div style="margin-top: 20px; padding: 15px; background: var(--bg-secondary); border-radius: 8px;">
                    <strong><i class="fas fa-info-circle"></i> 提示：</strong>
                    <p style="margin: 10px 0 0 0; font-size: 14px;">
                        如果您确定图像包含隐藏信息，请确认编码时使用的模型架构与当前选择的模型一致。
                        系统已启用自动模型检测功能，会依次尝试所有可用模型。
                    </p>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-primary" onclick="this.closest('.modal').remove()">我知道了</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

// ============================================================================
// 批量处理功能
// ============================================================================

// 请求中断当前任务
async function requestInterrupt() {
    // 首先检查本地状态
    if (!AppState.isProcessing) {
        NotificationManager.warning('当前没有正在运行的任务');
        return;
    }
    
    // 显示中断中提示
    const loadingNotification = NotificationManager.loading('正在发送中断请求...');
    
    try {
        // 调用新的中断当前操作 API
        const result = await API.interruptCurrentOperation();
        
        if (result.success) {
            AppState.interruptRequested = true;
            
            NotificationManager.updateLoading(
                loadingNotification,
                '中断请求已发送，正在等待操作停止...',
                'warning'
            );
            
            // 显示中断成功提示
            const operationType = result.operation?.type || '未知';
            const typeNames = {
                'encode': '消息嵌入',
                'decode': '消息提取',
                'batch': '批量嵌入'
            };
            
            setTimeout(() => {
                NotificationManager.show(
                    `<div style="text-align: left;">
                        <strong style="font-size: 14px; color: var(--warning-color);">
                            <i class="fas fa-exclamation-triangle"></i> 任务已标记中断
                        </strong><br>
                        <div style="margin: 8px 0; font-size: 13px;">
                            操作类型: ${typeNames[operationType] || operationType}<br>
                            开始时间: ${result.operation?.start_time || '未知'}<br>
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary);">
                            操作将在当前处理步骤完成后停止
                        </div>
                    </div>`,
                    'warning',
                    5000
                );
            }, 500);
            
        } else {
            NotificationManager.updateLoading(
                loadingNotification,
                result.error || '中断失败',
                'error'
            );
        }
    } catch (error) {
        console.error('中断任务失败:', error);
        NotificationManager.updateLoading(
            loadingNotification,
            `中断失败: ${error.message}`,
            'error'
        );
    }
}

// 显示中断通知（简化版本）
function showInterruptNotification(task) {
    const interruptTime = task.interrupted_at || new Date().toLocaleString();
    const progressCurrent = task.progress?.current || 0;
    const progressTotal = task.progress?.total || 0;
    const progressText = progressTotal > 0 ? `已处理: ${progressCurrent}/${progressTotal}` : '处理中中断';
    
    NotificationManager.show(
        `<div style="text-align: left;">
            <strong style="font-size: 14px;">任务已中断</strong><br>
            <div style="margin: 8px 0; font-size: 13px;">
                任务已在 ${interruptTime} 中断<br>
                <span style="color: var(--text-muted);">${progressText}</span>
            </div>
        </div>`,
        'warning',
        5000, // 5秒后自动关闭
        [
            {
                text: '查看历史',
                type: 'primary',
                onClick: `openHistoryModal()`
            }
        ]
    );
}

// 重新执行中断的任务（简化版本）
async function resumeInterruptedTask(taskId) {
    // 直接删除旧任务并提示用户重新操作
    try {
        await API.deleteTask(taskId);
        NotificationManager.info('已清除中断任务记录，请重新操作');
        loadTaskHistory();
    } catch (error) {
        console.error('清除中断任务失败:', error);
    }
}

// 删除任务
async function deleteTask(taskId) {
    if (!confirm('确定要删除这个任务吗？此操作不可恢复。')) {
        return;
    }
    
    try {
        const result = await API.deleteTask(taskId);
        
        if (result.success) {
            NotificationManager.success('任务已删除');
            loadTaskHistory();
        } else {
            NotificationManager.error(`删除失败: ${result.error}`);
        }
    } catch (error) {
        console.error('删除任务失败:', error);
        NotificationManager.error(`删除失败: ${error.message}`);
    }
}

// 加载任务历史
async function loadTaskHistory() {
    try {
        const result = await API.getTasks(20);
        
        if (result.success) {
            AppState.taskHistory = result.tasks || [];
            console.log('[DEBUG] 加载任务历史:', AppState.taskHistory.length, '条记录');
            renderTaskHistory();
        } else {
            console.warn('[DEBUG] 加载任务历史失败:', result.error);
            AppState.taskHistory = [];
            renderTaskHistory();
        }
    } catch (error) {
        console.error('加载任务历史失败:', error);
        AppState.taskHistory = [];
        renderTaskHistory();
    }
}

// 渲染任务历史（侧边栏）
function renderTaskHistory() {
    const historyList = elements.historyList;
    if (!historyList) return;
    
    if (AppState.taskHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>暂无历史记录</p>
            </div>
        `;
        return;
    }
    
    const recentTasks = AppState.taskHistory.slice(0, 5);
    
    historyList.innerHTML = recentTasks.map(task => {
        const typeConfig = getTaskTypeConfig(task.type);
        const time = formatTaskTime(task.created_at);
        const filename = task.filename || '未知文件';
        const statusClass = task.status === 'interrupted' ? 'interrupted' : '';
        
        return `
            <div class="history-item ${task.type} ${statusClass}" 
                 data-task-id="${task.id}" 
                 data-task-type="${task.type}"
                 onclick="handleHistoryItemClick('${task.id}', '${task.type}')"
                 title="点击查看详情">
                <div class="history-item-icon">
                    <i class="fas ${typeConfig.icon}"></i>
                </div>
                <div class="history-item-info">
                    <div class="history-item-title">${filename}</div>
                    <div class="history-item-meta">
                        <span class="history-item-badge ${task.type}">${typeConfig.name}</span>
                        <span class="history-item-time">${time}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// 处理历史记录点击
function handleHistoryItemClick(taskId, taskType) {
    const task = AppState.taskHistory.find(t => t.id === taskId);
    if (!task) return;
    
    if (taskType === 'encode') {
        switchTab('encode');
        if (task.filename) {
            showNotification(`正在加载: ${task.filename}`, 'info');
        }
    } else if (taskType === 'decode') {
        switchTab('decode');
        if (task.filename) {
            showNotification(`正在加载: ${task.filename}`, 'info');
        }
    } else if (taskType === 'batch') {
        switchTab('batch');
    }
    
    openHistoryModal();
}

// 获取任务类型配置
function getTaskTypeConfig(type) {
    const configs = {
        'encode': { name: '嵌入消息', icon: 'fa-lock' },
        'decode': { name: '提取消息', icon: 'fa-unlock' },
        'batch': { name: '批量处理', icon: 'fa-layer-group' }
    };
    return configs[type] || { name: '未知操作', icon: 'fa-question' };
}

// 获取任务类型配置
function getTaskTypeConfig(type) {
    const configs = {
        'encode': { name: '嵌入消息', icon: 'fa-lock' },
        'decode': { name: '提取消息', icon: 'fa-unlock' },
        'batch_encode': { name: '批量嵌入', icon: 'fa-layer-group' }
    };
    return configs[type] || { name: '未知任务', icon: 'fa-question' };
}

// 格式化任务时间
function formatTaskTime(timeString) {
    try {
        const date = new Date(timeString);
        return date.toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    } catch {
        return timeString;
    }
}

// ============================================================================
// 完整历史记录模态框功能
// ============================================================================

function openHistoryModal() {
    if (elements.historyModal) {
        elements.historyModal.classList.add('active');
        document.body.style.overflow = 'hidden'; // 防止背景滚动
        
        // 加载最新历史记录
        loadTaskHistory().then(() => {
            renderFullHistory();
        });
        
        // 添加键盘事件监听
        document.addEventListener('keydown', handleHistoryModalKeydown);
    }
}

function closeHistoryModal() {
    if (elements.historyModal) {
        elements.historyModal.classList.remove('active');
        document.body.style.overflow = ''; // 恢复背景滚动
        
        // 移除键盘事件监听
        document.removeEventListener('keydown', handleHistoryModalKeydown);
    }
}

function handleHistoryModalKeydown(e) {
    if (e.key === 'Escape') {
        closeHistoryModal();
    }
}

function renderFullHistory() {
    if (!elements.historyModalList) return;
    
    // 确保 taskHistory 是数组
    if (!Array.isArray(AppState.taskHistory)) {
        AppState.taskHistory = [];
    }
    
    let filteredTasks = [...AppState.taskHistory];
    
    // 过滤掉无效的任务数据
    filteredTasks = filteredTasks.filter(task => task && typeof task === 'object');
    
    // 类型筛选
    if (AppState.historyFilter.type !== 'all') {
        filteredTasks = filteredTasks.filter(task => task.type === AppState.historyFilter.type);
    }
    
    // 状态筛选
    if (AppState.historyFilter.status !== 'all') {
        filteredTasks = filteredTasks.filter(task => task.status === AppState.historyFilter.status);
    }
    
    // 搜索筛选
    if (AppState.historyFilter.search) {
        const searchTerm = AppState.historyFilter.search.toLowerCase();
        filteredTasks = filteredTasks.filter(task => {
            const typeName = getTaskTypeConfig(task.type).name.toLowerCase();
            const statusText = getTaskStatusConfig(task.status).text.toLowerCase();
            const fileInfo = task.progress?.current_file?.toLowerCase() || '';
            return typeName.includes(searchTerm) || 
                   statusText.includes(searchTerm) || 
                   fileInfo.includes(searchTerm);
        });
    }
    
    // 排序
    filteredTasks.sort((a, b) => {
        const timeA = new Date(a.created_at || 0).getTime();
        const timeB = new Date(b.created_at || 0).getTime();
        return AppState.historyFilter.sort === 'newest' ? timeB - timeA : timeA - timeB;
    });
    
    // 更新统计
    if (elements.historyTotalCount) {
        elements.historyTotalCount.textContent = filteredTasks.length;
    }
    
    // 渲染列表
    if (filteredTasks.length === 0) {
        elements.historyModalList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>暂无符合条件的历史记录</p>
                <small>尝试调整筛选条件或搜索关键词</small>
            </div>
        `;
        return;
    }
    
    elements.historyModalList.innerHTML = filteredTasks.map(task => {
        const type = task.type || 'encode';
        const taskId = task.id || 'unknown';
        const filename = task.filename || '未知文件';
        const createdAt = task.created_at ? formatTaskTime(task.created_at) : '未知时间';
        
        const typeConfig = getTaskTypeConfig(type);
        
        return `
            <div class="history-item ${type}" 
                 data-task-id="${taskId}"
                 onclick="handleHistoryItemClick('${taskId}', '${type}')"
                 title="点击查看详情">
                <div class="history-item-icon">
                    <i class="fas ${typeConfig.icon}"></i>
                </div>
                <div class="history-item-info">
                    <div class="history-item-title">${filename}</div>
                    <div class="history-item-meta">
                        <span class="history-item-badge ${type}">${typeConfig.name}</span>
                        <span class="history-item-time">${createdAt}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

async function clearAllHistory() {
    if (!confirm('确定要清空所有历史记录吗？此操作不可恢复。')) {
        return;
    }
    
    try {
        // 删除所有任务
        const deletePromises = AppState.taskHistory.map(task => 
            API.deleteTask(task.id).catch(() => null)
        );
        await Promise.all(deletePromises);
        
        AppState.taskHistory = [];
        renderFullHistory();
        renderTaskHistory();
        
        NotificationManager.success('所有历史记录已清空');
    } catch (error) {
        NotificationManager.error('清空历史记录失败: ' + error.message);
    }
}

async function handleBatchFiles(files) {
    const validFiles = files.filter(validateFile);
    
    if (validFiles.length === 0) return;
    
    const loadingNotification = NotificationManager.loading(`正在处理 ${validFiles.length} 个文件...`);
    
    AppState.batchFiles = [];
    elements.batchFileList.innerHTML = '';
    
    let validCount = 0;
    let warningCount = 0;
    let errorCount = 0;
    
    for (let i = 0; i < validFiles.length; i++) {
        const file = validFiles[i];
        
        try {
            const result = await API.uploadFile(file);
            
            if (result.success) {
                const fileInfo = {
                    name: result.filename,
                    originalName: file.name,
                    size: file.size,
                    preview: result.preview,
                    processed: false,
                    error: null,
                    isValid: result.info.is_valid_for_steganography !== false,
                    sizeWarning: result.info.size_warning
                };
                
                AppState.batchFiles.push(fileInfo);
                
                // 创建文件项并添加状态指示
                const fileItem = renderBatchFileItem({
                    name: result.originalName,
                    size: file.size
                }, i);
                
                // 如果尺寸不足，添加警告标记
                if (!fileInfo.isValid) {
                    fileItem.classList.add('size-invalid');
                    const statusSpan = fileItem.querySelector('.batch-file-status');
                    if (statusSpan) {
                        statusSpan.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: var(--error-color);" title="尺寸过小"></i>';
                    }
                    errorCount++;
                } else if (fileInfo.sizeWarning) {
                    fileItem.classList.add('size-warning');
                    warningCount++;
                    validCount++;
                } else {
                    validCount++;
                }
                
                elements.batchFileList.appendChild(fileItem);
            }
        } catch (error) {
            AppState.batchFiles.push({
                originalName: file.name,
                processed: false,
                error: error.message,
                isValid: false
            });
            errorCount++;
        }
    }
    
    NotificationManager.remove(loadingNotification);
    
    // 显示处理结果摘要
    if (errorCount > 0 || warningCount > 0) {
        let summaryMsg = `处理完成: ${validCount} 个可用`;
        if (warningCount > 0) {
            summaryMsg += `, ${warningCount} 个有尺寸警告`;
        }
        if (errorCount > 0) {
            summaryMsg += `, ${errorCount} 个尺寸不足`;
        }
        NotificationManager.warning(summaryMsg, 5000);
    } else {
        NotificationManager.success(`已添加 ${validCount} 个文件`);
    }
    
    updateBatchUI();
}

function updateBatchUI() {
    const hasFiles = AppState.batchFiles.length > 0;
    const hasMessage = elements.batchMessageInput.value.trim();
    
    // 检查是否有足够尺寸的有效文件
    const hasValidFiles = AppState.batchFiles.some(f => f.isValid !== false);
    
    // 只有当有有效文件且有消息时才启用按钮
    elements.btnBatchEncode.disabled = !hasFiles || !hasMessage || !hasValidFiles;
}

async function batchEncode() {
    const message = elements.batchMessageInput.value.trim();
    if (!message) {
        NotificationManager.warning('请输入要嵌入的消息');
        elements.batchMessageInput.focus();
        return;
    }
    
    if (AppState.batchFiles.length === 0) {
        NotificationManager.warning('请先上传要处理的图像文件');
        return;
    }
    
    // 检查网络连接
    const isConnected = await API.checkConnection();
    if (!isConnected) {
        NotificationManager.error('无法连接到服务器，请检查网络连接');
        return;
    }
    
    AppState.isProcessing = true;
    setProcessingState(true);
    
    elements.batchProgress.style.display = 'block';
    
    const total = AppState.batchFiles.length;
    const loadingNotification = NotificationManager.loading(
        `批量嵌入中... 0/${total} 完成`
    );
    
    // 准备文件列表
    const filenames = AppState.batchFiles.map(f => f.name);
    
    // 存储下载链接
    AppState.batchDownloadLinks = [];
    
    try {
        // 使用新的批量处理 API
        const result = await API.batchEncode(
            filenames,
            message
        );
        
        if (result.success) {
            // 更新每个文件的状态
            let success = 0;
            let failed = 0;
            
            result.results.forEach((res, i) => {
                const file = AppState.batchFiles[i];
                const fileElement = document.querySelector(`[data-index="${i}"]`);
                const statusElement = fileElement?.querySelector('.batch-file-status');
                
                if (res.success) {
                    file.processed = true;
                    file.result = res;
                    success++;
                    
                    // 保存下载链接
                    AppState.batchDownloadLinks.push({
                        name: res.output || `encoded_${file.originalName}`,
                        url: res.download_url,
                        originalName: file.originalName,
                        size: res.file_size
                    });
                    
                    if (statusElement) {
                        statusElement.innerHTML = '<i class="fas fa-check-circle" style="color: var(--success-color);"></i>';
                    }
                    if (fileElement) {
                        fileElement.classList.add('success');
                    }
                } else {
                    file.error = res.error || '处理失败';
                    failed++;
                    
                    if (statusElement) {
                        statusElement.innerHTML = '<i class="fas fa-times-circle" style="color: var(--error-color);"></i>';
                    }
                    if (fileElement) {
                        fileElement.classList.add('error');
                    }
                }
            });
            
            elements.batchProgressText.textContent = `${success}/${total}`;
            elements.batchProgressFill.style.width = `${(success / total) * 100}%`;
            
            NotificationManager.remove(loadingNotification);
            
            // 显示批量结果面板
            showBatchResult(success, failed, result.total_time, result.results);
            
            // 显示结果通知
            if (success === total) {
                NotificationManager.success(
                    `批量嵌入完成！所有 ${total} 个文件处理成功 (耗时: ${result.total_time})`,
                    3000
                );
                addToHistory('batch', `批量嵌入 ${total} 个文件`, result.total_time);
            } else if (success > 0) {
                NotificationManager.warning(
                    `批量嵌入完成: ${success} 个成功, ${failed} 个失败 (耗时: ${result.total_time})`,
                    5000
                );
            } else {
                NotificationManager.error('所有文件处理失败，请检查文件和网络连接');
            }
        } else {
            throw new Error(result.error || '批量嵌入失败');
        }
    } catch (error) {
        NotificationManager.remove(loadingNotification);
        
        // 检查是否为中断错误
        if (error.interrupted || error.message?.includes('中断')) {
            NotificationManager.warning('批量嵌入已被中断');
            // 如果有部分结果，显示已完成的部分
            if (error.results && error.results.length > 0) {
                const successCount = error.results.filter(r => r.success).length;
                if (successCount > 0) {
                    showBatchResult(successCount, error.results.length - successCount, error.total_time || '0s', error.results);
                }
            }
            // 刷新历史记录以显示中断的任务
            await loadTaskHistory();
            return;
        }
        
        NotificationManager.error(`批量嵌入失败: ${error.message}`);
        console.error('批量嵌入错误:', error);
    } finally {
        AppState.isProcessing = false;
        setProcessingState(false);
        AppState.interruptRequested = false;
        
        // 操作完成后刷新历史记录，确保状态正确显示
        loadTaskHistory();
    }
}

// 显示批量处理结果
function showBatchResult(successCount, failCount, totalTime, results) {
    // 隐藏进度条
    elements.batchProgress.style.display = 'none';
    
    // 显示结果面板
    elements.batchResultPanel.style.display = 'block';
    elements.batchResultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // 更新统计数据
    elements.batchSuccessCount.textContent = successCount;
    elements.batchFailCount.textContent = failCount;
    elements.batchTime.textContent = totalTime;
    
    // 更新下载按钮
    const downloadCount = successCount;
    elements.downloadCountBadge.textContent = downloadCount > 0 ? `(${downloadCount} 个文件)` : '';
    
    // 生成文件列表
    let filesHtml = '';
    AppState.batchFiles.forEach((file, index) => {
        const result = results[index];
        const isSuccess = result && result.success;
        
        filesHtml += `
            <div class="batch-file-result ${isSuccess ? 'success' : 'error'}">
                <div class="file-result-icon">
                    <i class="fas ${isSuccess ? 'fa-check-circle' : 'fa-times-circle'}"></i>
                </div>
                <div class="file-result-info">
                    <div class="file-result-name" title="${file.originalName}">${file.originalName}</div>
                    <div class="file-result-status">
                        ${isSuccess 
                            ? `<span class="success-text">嵌入成功</span> - ${formatFileSize(result.file_size)}`
                            : `<span class="error-text">${file.error || '处理失败'}</span>`
                        }
                    </div>
                </div>
                ${isSuccess ? `
                    <button class="btn btn-sm btn-secondary file-download-btn" data-index="${index}">
                        <i class="fas fa-download"></i>
                        <span>下载</span>
                    </button>
                ` : ''}
            </div>
        `;
    });
    
    elements.batchFilesResult.innerHTML = filesHtml;
    
    // 为每个下载按钮添加事件
    document.querySelectorAll('.file-download-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.currentTarget.dataset.index);
            const link = AppState.batchDownloadLinks[index];
            if (link) {
                downloadSingleFile(link);
            }
        });
    });
    
    // 更新下载提示
    if (successCount > 0) {
        elements.downloadHintText.textContent = `共 ${successCount} 个文件可以下载，点击上方按钮一键下载全部`;
    } else {
        elements.downloadHintText.textContent = '没有可下载的文件';
        elements.btnDownloadAllBatch.disabled = true;
        elements.btnPackageDownload.disabled = true;
    }
}

// 下载单个文件
async function downloadSingleFile(link) {
    try {
        const fullUrl = link.url.startsWith('http') ? link.url : `${window.location.origin}${link.url}`;
        const response = await fetch(fullUrl);
        
        if (!response.ok) {
            throw new Error(`下载失败: ${response.statusText}`);
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = link.name.startsWith('encoded_') ? link.name : `encoded_${link.name}`;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);
        
        NotificationManager.success(`已下载: ${link.originalName}`, 2000);
    } catch (error) {
        console.error(`下载失败:`, error);
        NotificationManager.error(`下载失败: ${error.message}`);
    }
}

// 批量下载功能 - 增强版，支持完整URL和进度提示
async function downloadAllFiles(links) {
    if (!links || links.length === 0) {
        NotificationManager.warning('没有可下载的文件');
        return;
    }
    
    const totalFiles = links.length;
    const downloadNotification = NotificationManager.loading(`正在准备下载 ${totalFiles} 个文件...`);
    
    let successCount = 0;
    let failCount = 0;
    
    for (let i = 0; i < links.length; i++) {
        const link = links[i];
        const progress = `下载进度: ${i + 1}/${totalFiles}`;
        
        // 更新下载进度提示
        const content = downloadNotification.querySelector('.notification-message');
        if (content) {
            content.textContent = `${progress} - ${link.name}`;
        }
        
        try {
            // 使用完整的API URL
            const fullUrl = link.url.startsWith('http') ? link.url : `${window.location.origin}${link.url}`;
            
            // 使用fetch下载文件，避免浏览器阻止
            const response = await fetch(fullUrl);
            
            if (!response.ok) {
                throw new Error(`下载失败: ${response.statusText}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            
            // 创建下载链接
            const a = document.createElement('a');
            a.href = url;
            a.download = link.name.startsWith('encoded_') ? link.name : `encoded_${link.name}`;
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();
            
            // 清理
            setTimeout(() => {
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }, 100);
            
            successCount++;
            
            // 增加延迟避免浏览器阻止连续下载
            if (i < links.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 800));
            }
            
        } catch (error) {
            console.error(`下载 ${link.name} 失败:`, error);
            failCount++;
        }
    }
    
    NotificationManager.remove(downloadNotification);
    
    // 显示下载结果
    if (successCount === totalFiles) {
        NotificationManager.success(`成功下载全部 ${totalFiles} 个文件`, 5000);
    } else if (successCount > 0) {
        NotificationManager.warning(
            `下载完成: ${successCount} 个成功, ${failCount} 个失败`,
            6000,
            failCount > 0 ? [{
                text: '重试失败的文件',
                type: 'primary',
                onClick: `downloadAllFiles(${JSON.stringify(links.filter((link, index) => {
                    // 这里简化处理，实际应该跟踪失败的文件
                    return index >= successCount;
                }))})`
            }] : []
        );
    } else {
        NotificationManager.error('所有文件下载失败，请检查网络连接后重试', 8000, [
            {
                text: '重试',
                type: 'primary',
                onClick: `downloadAllFiles(${JSON.stringify(links)})`
            }
        ]);
    }
}

// ============================================================================
// 历史记录
// ============================================================================
function addToHistory(type, data) {
    AppState.history.unshift({
        type,
        ...data,
        timestamp: Date.now()
    });
    
    // 只保留最近10条记录
    if (AppState.history.length > 10) {
        AppState.history = AppState.history.slice(0, 10);
    }
    
    renderHistory();
}

function renderHistory() {
    if (!elements || !elements.historyList) {
        console.warn('renderHistory: elements 或 historyList 未初始化');
        return;
    }
    
    if (AppState.history.length === 0) {
        elements.historyList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>暂无历史记录</p>
            </div>
        `;
        return;
    }
    
    elements.historyList.innerHTML = AppState.history.map(item => {
        const icon = item.type === 'encode' ? 'fa-lock' : (item.type === 'decode' ? 'fa-unlock' : 'fa-layer-group');
        const time = item.timestamp ? new Date(item.timestamp).toLocaleString() : (item.time || '');
        
        return `
            <div class="history-item">
                <div class="history-item-icon">
                    <i class="fas ${icon}"></i>
                </div>
                <div class="history-item-info">
                    <div class="history-item-title">${item.filename || item.message || ''}</div>
                    <div class="history-item-time">${time}</div>
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================================
// 主题切换
// ============================================================================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    elements.themeToggle.innerHTML = `<i class="fas fa-${newTheme === 'dark' ? 'sun' : 'moon'}"></i>`;
    
    // 保存到本地存储
    localStorage.setItem('theme', newTheme);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    elements.themeToggle.innerHTML = `<i class="fas fa-${savedTheme === 'dark' ? 'sun' : 'moon'}"></i>`;
}

// ============================================================================
// 处理状态 - 增强版
// ============================================================================
function setProcessingState(processing) {
    const buttons = [
        elements.btnEncode,
        elements.btnDecode,
        elements.btnBatchEncode
    ];
    
    const interruptButtons = [
        document.getElementById('btn-interrupt-encode'),
        document.getElementById('btn-interrupt-decode'),
        document.getElementById('btn-interrupt-batch')
    ];
    
    buttons.forEach(btn => {
        if (!btn) return;
        
        if (processing) {
            btn.classList.add('loading');
            btn.disabled = true;
            // 添加微妙的脉冲效果
            btn.style.animation = 'pulse 2s ease-in-out infinite';
        } else {
            btn.classList.remove('loading');
            btn.disabled = false;
            btn.style.animation = '';
        }
    });
    
    // 显示/隐藏中断按钮
    interruptButtons.forEach(btn => {
        if (!btn) return;
        
        if (processing) {
            btn.style.display = 'inline-flex';
            btn.classList.add('pulse-animation');
        } else {
            btn.style.display = 'none';
            btn.classList.remove('pulse-animation');
        }
    });
    
    // 更新状态栏
    if (elements.statusText) {
        elements.statusText.textContent = processing ? '处理中...' : '就绪';
        elements.statusText.classList.toggle('status-pulse', processing);
    }
    
    // 显示/隐藏全局进度条
    if (elements.globalProgress) {
        elements.globalProgress.style.display = processing ? 'block' : 'none';
        if (processing) {
            // 进度条动画
            let progress = 0;
            const interval = setInterval(() => {
                progress += 2;
                if (progress > 90 || !AppState.isProcessing) {
                    clearInterval(interval);
                }
                if (elements.globalProgressFill) {
                    elements.globalProgressFill.style.width = `${progress}%`;
                }
            }, 200);
        } else {
            if (elements.globalProgressFill) {
                elements.globalProgressFill.style.width = '100%';
                setTimeout(() => {
                    elements.globalProgressFill.style.width = '0%';
                }, 300);
            }
        }
    }
    
    // 禁用/启用拖拽区域
    const dropZones = [
        elements.dropZoneEncode,
        elements.dropZoneDecode,
        elements.dropZoneBatch
    ];
    
    dropZones.forEach(zone => {
        if (zone) {
            zone.style.pointerEvents = processing ? 'none' : 'auto';
            zone.style.opacity = processing ? '0.6' : '1';
        }
    });
}

// ============================================================================
// 主题切换 - 增强版
// ============================================================================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    // 添加过渡效果
    html.style.transition = 'background-color 0.3s ease, color 0.3s ease';
    
    html.setAttribute('data-theme', newTheme);
    
    if (elements.themeToggle) {
        elements.themeToggle.innerHTML = `<i class="fas fa-${newTheme === 'dark' ? 'sun' : 'moon'}"></i>`;
        elements.themeToggle.title = newTheme === 'dark' ? '切换到浅色主题' : '切换到深色主题';
    }
    
    // 保存到本地存储
    localStorage.setItem('theme', newTheme);
    
    // 触发主题切换事件
    window.dispatchEvent(new CustomEvent('themechange', { 
        detail: { theme: newTheme } 
    }));
    
    NotificationManager.info(`已切换到${newTheme === 'dark' ? '深色' : '浅色'}主题`);
    
    // 移除过渡效果
    setTimeout(() => {
        html.style.transition = '';
    }, 300);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    if (elements.themeToggle) {
        elements.themeToggle.innerHTML = `<i class="fas fa-${savedTheme === 'dark' ? 'sun' : 'moon'}"></i>`;
        elements.themeToggle.title = savedTheme === 'dark' ? '切换到浅色主题' : '切换到深色主题';
    }
    
    // 检测系统主题偏好
    if (!localStorage.getItem('theme')) {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
            document.documentElement.setAttribute('data-theme', 'dark');
            if (elements.themeToggle) {
                elements.themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            }
        }
    }
}

// ============================================================================
// 帮助模态框
// ============================================================================
function toggleHelpModal(show) {
    if (show) {
        elements.helpModal.classList.add('active');
    } else {
        elements.helpModal.classList.remove('active');
    }
}

// ============================================================================
// 事件监听器设置
// ============================================================================
function setupEventListeners() {
    if (!elements) {
        console.error('setupEventListeners: elements对象未初始化');
        return;
    }
    
    if (elements.navButtons) {
        elements.navButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                switchTab(tab);
            });
        });
    }
    
    // 嵌入相关
    if (elements.dropZoneEncode && elements.fileInputEncode) {
        setupDragAndDrop(elements.dropZoneEncode, elements.fileInputEncode, handleEncodeFiles);
    }
    if (elements.btnEncode) {
        elements.btnEncode.addEventListener('click', encodeMessage);
    }
    if (elements.btnClearEncode) {
        elements.btnClearEncode.addEventListener('click', clearEncode);
    }
    if (elements.btnNewEncode) {
        elements.btnNewEncode.addEventListener('click', clearEncode);
    }
    
    if (elements.dropZoneDecode && elements.fileInputDecode) {
        setupDragAndDrop(elements.dropZoneDecode, elements.fileInputDecode, handleDecodeFile);
    }
    if (elements.btnDecode) {
        elements.btnDecode.addEventListener('click', decodeMessage);
    }
    if (elements.btnClearDecode) {
        elements.btnClearDecode.addEventListener('click', clearDecode);
    }
    if (elements.btnNewDecode) {
        elements.btnNewDecode.addEventListener('click', clearDecode);
    }
    
    if (elements.dropZoneBatch && elements.fileInputBatch) {
        setupDragAndDrop(elements.dropZoneBatch, elements.fileInputBatch, handleBatchFiles);
    }
    if (elements.btnBatchEncode) {
        elements.btnBatchEncode.addEventListener('click', batchEncode);
    }
    if (elements.btnClearBatch) {
        elements.btnClearBatch.addEventListener('click', clearBatch);
    }
    if (elements.btnNewBatch) {
        elements.btnNewBatch.addEventListener('click', clearBatch);
    }
    
    const btnInterruptEncode = document.getElementById('btn-interrupt-encode');
    const btnInterruptDecode = document.getElementById('btn-interrupt-decode');
    const btnInterruptBatch = document.getElementById('btn-interrupt-batch');
    
    if (btnInterruptEncode) {
        btnInterruptEncode.addEventListener('click', requestInterrupt);
    }
    if (btnInterruptDecode) {
        btnInterruptDecode.addEventListener('click', requestInterrupt);
    }
    if (btnInterruptBatch) {
        btnInterruptBatch.addEventListener('click', requestInterrupt);
    }
    
    if (elements.btnDownloadAllBatch) {
        elements.btnDownloadAllBatch.addEventListener('click', () => {
            if (AppState.batchDownloadLinks && AppState.batchDownloadLinks.length > 0) {
                downloadAllFiles(AppState.batchDownloadLinks);
            }
        });
    }
    
    if (elements.btnPackageDownload) {
        elements.btnPackageDownload.addEventListener('click', () => {
            NotificationManager.info('打包下载功能开发中，请使用"下载全部文件"按钮');
        });
    }
    
    if (elements.messageInput) {
        elements.messageInput.addEventListener('input', updateEncodeUI);
        elements.messageInput.addEventListener('compositionend', updateEncodeUI);
        elements.messageInput.addEventListener('paste', () => {
            setTimeout(updateEncodeUI, 0);
        });
    }
    
    if (elements.batchMessageInput) {
        elements.batchMessageInput.addEventListener('input', updateBatchUI);
        elements.batchMessageInput.addEventListener('compositionend', updateBatchUI);
        elements.batchMessageInput.addEventListener('paste', () => {
            setTimeout(updateBatchUI, 0);
        });
    }
    
    if (elements.clearHistory) {
        elements.clearHistory.addEventListener('click', () => {
            AppState.history = [];
            renderHistory();
            showNotification('历史记录已清空', 'info');
        });
    }
    
    if (elements.viewAllHistory) {
        elements.viewAllHistory.addEventListener('click', openHistoryModal);
    }
    
    if (elements.historyModalClose) {
        elements.historyModalClose.addEventListener('click', closeHistoryModal);
    }
    if (elements.historyModalBackBtn) {
        elements.historyModalBackBtn.addEventListener('click', closeHistoryModal);
    }
    if (elements.historyModal) {
        elements.historyModal.addEventListener('click', (e) => {
            if (e.target === elements.historyModal) {
                closeHistoryModal();
            }
        });
    }
    
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            AppState.historyFilter.type = btn.dataset.filter;
            renderFullHistory();
        });
    });
    
    document.querySelectorAll('.filter-btn-status').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn-status').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            AppState.historyFilter.status = btn.dataset.status;
            renderFullHistory();
        });
    });
    
    if (elements.historySearchInput) {
        elements.historySearchInput.addEventListener('input', (e) => {
            AppState.historyFilter.search = e.target.value.toLowerCase();
            renderFullHistory();
        });
    }
    
    if (elements.historySortSelect) {
        elements.historySortSelect.addEventListener('change', (e) => {
            AppState.historyFilter.sort = e.target.value;
            renderFullHistory();
        });
    }
    
    if (elements.clearAllHistory) {
        elements.clearAllHistory.addEventListener('click', clearAllHistory);
    }
    
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', toggleTheme);
    }
    
    if (elements.helpBtn) {
        elements.helpBtn.addEventListener('click', () => toggleHelpModal(true));
    }
    if (elements.modalClose) {
        elements.modalClose.addEventListener('click', () => toggleHelpModal(false));
    }
    if (elements.helpModal) {
        elements.helpModal.addEventListener('click', (e) => {
            if (e.target === elements.helpModal) {
                toggleHelpModal(false);
            }
        });
    }
    
    if (elements.btnCopyMessage) {
        elements.btnCopyMessage.addEventListener('click', () => {
            navigator.clipboard.writeText(elements.decodedMessage.value);
            showNotification('消息已复制到剪贴板', 'success');
        });
    }
}

// ============================================================================
// 标签切换
// ============================================================================
function switchTab(tab) {
    AppState.currentTab = tab;
    
    // 更新导航按钮状态
    elements.navButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    
    // 更新标签内容显示
    Object.keys(elements.tabContents).forEach(key => {
        elements.tabContents[key].classList.toggle('active', key === tab);
    });
}

// ============================================================================
// 清除功能
// ============================================================================
function clearEncode() {
    AppState.uploadedFiles = [];
    elements.fileListEncode.innerHTML = '';
    elements.messageInput.value = '';
    elements.resultPanelEncode.style.display = 'none';
    updateEncodeUI();
    
    showNotification('已清除', 'info');
}

function clearDecode() {
    AppState.uploadedFiles = [];
    elements.fileInfoDecode.style.display = 'none';
    elements.resultPanelDecode.style.display = 'none';
    elements.btnDecode.disabled = true;
    
    showNotification('已清除', 'info');
}

function clearBatch() {
    AppState.batchFiles = [];
    AppState.batchDownloadLinks = [];
    elements.batchFileList.innerHTML = '';
    elements.batchMessageInput.value = '';
    elements.batchProgress.style.display = 'none';
    elements.batchResultPanel.style.display = 'none';
    elements.batchFilesResult.innerHTML = '';
    
    // 重置下载按钮状态
    elements.btnDownloadAllBatch.disabled = false;
    elements.btnPackageDownload.disabled = false;
    
    updateBatchUI();
    
    showNotification('已清除', 'info');
}

// ============================================================================
// 初始化
// ============================================================================
async function init() {
    console.log('开始初始化应用...');
    
    // 首先初始化 DOM 元素引用
    const elementsInitialized = await initElements();
    if (!elementsInitialized) {
        console.error('DOM 元素初始化失败，无法继续');
        alert('页面初始化失败，请刷新页面重试。');
        return;
    }
    
    loadTheme();
    setupEventListeners();
    renderHistory();
    
    // 初始化替换确认弹窗（嵌入消息功能）
    initReplaceImageModal();
    
    // 检查连接状态
    checkConnection();
    
    // 更新版本号显示
    updateVersionDisplay();
    
    // 加载任务历史
    loadTaskHistory();
    
    // 键盘快捷键
    setupKeyboardShortcuts();
    
    // 设置页面关闭/刷新时的中断处理
    setupPageUnloadHandler();
    
    console.log(`ImgStegGAN Web Interface V${VERSION} 初始化完成`);
}

// ============================================================================
// 页面关闭/刷新中断处理
// ============================================================================

/**
 * 设置页面关闭/刷新时的中断处理
 * 当用户关闭页面或刷新时，如果有正在运行的任务，自动发送中断请求
 */
function setupPageUnloadHandler() {
    // 页面即将卸载时（关闭、刷新、导航离开）
    window.addEventListener('beforeunload', handlePageUnload);
    
    // 页面隐藏时（切换标签页、最小化浏览器）
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // 页面卸载完成（iOS Safari 支持）
    window.addEventListener('pagehide', handlePageUnload);
}

/**
 * 处理页面卸载事件
 * @param {BeforeUnloadEvent} event 
 */
function handlePageUnload(event) {
    if (AppState.isProcessing) {
        // 发送中断请求（使用 sendBeacon 确保请求能发送出去）
        interruptOperationSilent();
        
        // 显示提示信息（仅在某些浏览器中有效）
        const message = '当前有正在运行的任务，离开页面将中断操作。确定要离开吗？';
        event.returnValue = message;
        return message;
    }
}

/**
 * 处理页面可见性变化
 */
function handleVisibilityChange() {
    if (document.hidden && AppState.isProcessing) {
        // 页面被隐藏时，记录时间戳
        AppState.hiddenStartTime = Date.now();
        console.log('[页面状态] 页面隐藏，任务继续运行...');
    } else if (!document.hidden && AppState.hiddenStartTime) {
        // 页面重新可见时，检查是否需要刷新状态
        const hiddenDuration = Date.now() - AppState.hiddenStartTime;
        console.log(`[页面状态] 页面恢复可见，隐藏时长: ${hiddenDuration}ms`);
        
        // 如果隐藏时间超过30秒，刷新历史记录
        if (hiddenDuration > 30000) {
            loadTaskHistory();
        }
        AppState.hiddenStartTime = null;
    }
}

/**
 * 静默发送中断请求（不显示通知）
 * 使用 sendBeacon 或 XMLHttpRequest 同步请求确保请求能发送
 */
function interruptOperationSilent() {
    if (!AppState.isProcessing) return;
    
    try {
        // 使用 sendBeacon 发送中断请求（更可靠）
        const url = `${API.baseUrl}/operation/interrupt`;
        const data = new Blob([JSON.stringify({})], { type: 'application/json' });
        
        if (navigator.sendBeacon) {
            navigator.sendBeacon(url, data);
            console.log('[中断] 已通过 sendBeacon 发送中断请求');
        } else {
            // 回退方案：使用同步 XMLHttpRequest
            const xhr = new XMLHttpRequest();
            xhr.open('POST', url, false); // 同步请求
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.send(JSON.stringify({}));
            console.log('[中断] 已通过同步 XHR 发送中断请求');
        }
    } catch (error) {
        console.error('[中断] 发送中断请求失败:', error);
    }
}

// 更新版本号显示
async function updateVersionDisplay() {
    const versionElement = document.getElementById('version');
    if (versionElement) {
        versionElement.textContent = VERSION;
        versionElement.title = `版本: ${VERSION} | 发布日期: ${VERSION_DATE}`;
    }
    
    // 尝试从后端获取版本信息
    try {
        const response = await fetch('/api/version');
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.version) {
                if (versionElement) {
                    versionElement.textContent = data.version;
                    versionElement.title = `版本: ${data.version} | 发布日期: ${data.date}`;
                }
                console.log(`Server version: ${data.version} (${data.date})`);
            }
        }
    } catch (error) {
        // 使用本地版本信息
        console.log('Using local version info:', VERSION);
    }
}

function checkConnection() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                elements.connectionStatus.innerHTML = '<i class="fas fa-circle"></i> 已连接';
                elements.connectionStatus.classList.remove('offline');
                elements.connectionStatus.classList.add('online');
            }
        })
        .catch(() => {
            elements.connectionStatus.innerHTML = '<i class="fas fa-circle"></i> 未连接';
            elements.connectionStatus.classList.remove('online');
            elements.connectionStatus.classList.add('offline');
        });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + 1: 切换到嵌入标签
        if ((e.ctrlKey || e.metaKey) && e.key === '1') {
            e.preventDefault();
            switchTab('encode');
        }
        
        // Ctrl/Cmd + 2: �switch到解码标签
        if ((e.ctrlKey || e.metaKey) && e.key === '2') {
            e.preventDefault();
            switchTab('decode');
        }
        
        // Ctrl/Cmd + 3: 切换到批量处理标签
        if ((e.ctrlKey || e.metaKey) && e.key === '3') {
            e.preventDefault();
            switchTab('batch');
        }
        
        // Escape: 关闭模态框
        if (e.key === 'Escape') {
            e.preventDefault();
            toggleHelpModal(false);
        }
    });
}

// ============================================================================
// 页面加载完成后初始化
// ============================================================================
document.addEventListener('DOMContentLoaded', init);