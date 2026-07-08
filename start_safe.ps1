# FedProRef 安全启动脚本（解决 forrtl error 200）

# 方法 1: 禁用 MKL 加速（最稳定但较慢）
# $env:BLAS = "openblas"
# $env:LAPACK = "openblas"

# 方法 2: 限制 MKL 线程（推荐）
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# 允许重复的 OpenMP 运行时
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:KMP_INIT_AT_FORK = "FALSE"

# 禁用 PyTorch 的并行
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "=== Environment Setup ===" -ForegroundColor Green
Get-ChildItem Env: | Where-Object {$_.Name -match "THREAD|MKL|OMP|KMP|BLAS"} | Format-Table Name, Value -AutoSize
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

# 清理缓存
Remove-Item -Recurse -Force .\__pycache__ -ErrorAction SilentlyContinue

# 运行：默认在 seed=42,43,44 上各运行一次
python run.py --seed 42 --repeats 3 @args
