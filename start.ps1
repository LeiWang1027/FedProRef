# FedProRef 启动脚本
# 在导入任何 Python 库之前设置环境变量

# 禁用 MKL 线程冲突
$env:MKL_THREADING_LAYER = "GNU"
$env:OMP_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:MKL_SERVICE_FORCE_INTEL = "1"

# 禁用 Python 的线程竞争
$env:PYTHONPATH = "."

Write-Host "Environment variables set:"
Write-Host "  MKL_THREADING_LAYER = $env:MKL_THREADING_LAYER"
Write-Host "  OMP_NUM_THREADS = $env:OMP_NUM_THREADS"
Write-Host "  KMP_DUPLICATE_LIB_OK = $env:KMP_DUPLICATE_LIB_OK"
Write-Host ""

# 运行项目：默认在 seed=42,43,44 上各运行一次
python run.py --seed 42 --repeats 3 @args
