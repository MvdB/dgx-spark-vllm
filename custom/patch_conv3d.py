"""
Patch vLLM conv.py for GB10 (sm_120) compatibility.

forward_cuda() normally only uses the safe linear-matmul path (_forward_mulmat)
for PyTorch 2.9.x (where F.conv3d had a regression). On PyTorch 2.11.0 (NVIDIA
Spark image), F.conv3d hits a cuDNN "no algorithm found" error on GB10 for the
Qwen3.5-VL patch embedding shape (kernel=(2,16,16), stride=(2,16,16)).

Fix: remove the version check → always use _forward_mulmat when enable_linear=True.
"""
import sys

CONV_PATH = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/conv.py'

OLD = '        if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):'
NEW = '        if self.enable_linear:  # GB10 fix: cuDNN conv3d fails on sm_120 for ViT patch sizes'

content = open(CONV_PATH).read()
if OLD not in content:
    print(f'ERROR: Pattern not found in {CONV_PATH}', file=sys.stderr)
    print('First 10 lines of forward_cuda:', file=sys.stderr)
    for i, line in enumerate(content.split('\n')):
        if 'forward_cuda' in line or 'enable_linear' in line:
            print(f'  {i}: {line}', file=sys.stderr)
    sys.exit(1)

open(CONV_PATH, 'w').write(content.replace(OLD, NEW, 1))
print('OK: conv3d patched')
