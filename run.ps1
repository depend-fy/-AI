# 智鉴 AI（VeriMind AI）一键启动脚本
#
# 用法（在项目根目录）：
#     powershell -ExecutionPolicy Bypass -File .\run.ps1
#     powershell -ExecutionPolicy Bypass -File .\run.ps1 -Port 8600
#
# 它做了三件事：
#   1. 把当前终端切到 UTF-8，避免中文乱码
#   2. 检查虚拟环境和依赖是否存在，缺了就给明确提示
#   3. 启动 Streamlit 界面

param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'

# --- 1. 中文显示：切到 UTF-8 代码页 ---
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 | Out-Null
} catch {
    Write-Host "提示：切换 UTF-8 代码页失败，若中文显示乱码可手动执行 chcp 65001" -ForegroundColor Yellow
}
$env:PYTHONIOENCODING = 'utf-8'

# --- 2. 定位项目根目录与虚拟环境 ---
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'

Write-Host ""
Write-Host "智鉴 AI（VeriMind AI）" -ForegroundColor Cyan
Write-Host "项目目录：$Root"
Write-Host ""

if (-not (Test-Path $Python)) {
    Write-Host "找不到虚拟环境：$Python" -ForegroundColor Red
    Write-Host ""
    Write-Host "请先创建虚拟环境并安装依赖（在项目根目录执行）：" -ForegroundColor Yellow
    Write-Host "    py -3.13 -m venv .venv"
    Write-Host "    .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-Host ""
    Write-Host "更详细的环境问题请运行：" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\python.exe scripts\check_env.py"
    exit 1
}

# --- 3. 启动界面 ---
Write-Host "正在启动界面，浏览器会自动打开：http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "停止服务：在本窗口按 Ctrl + C" -ForegroundColor DarkGray
Write-Host ""

& $Python -m streamlit run (Join-Path $Root 'app.py') --server.port $Port
