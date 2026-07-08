$env:MKL_THREADING_LAYER = "GNU"
$env:OMP_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:MKL_SERVICE_FORCE_INTEL = "1"
$env:PYTHONPATH = "."

$python = "D:\SoftInstall\Miniconda\envs\fedfm\python.exe"

& $python "federated_loop.py" `
  --method "fedproref" `
  --dataset "cifar10" `
  --alpha "0.1" `
  --num_clients "10" `
  --backbone "ViT-B-16" `
  --pretrained "pretrain_path\old_open_clip_model.safetensors" `
  --head_type "mlp" `
  --device "cuda" `
  --exp_name "full_vitb16_old_mlp" `
  --seed "42"
