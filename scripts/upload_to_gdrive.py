"""
Google Drive上传辅助脚本。

功能：
1. 打印手动上传步骤说明
2. 生成Colab下载测试代码
3. 生成/更新 02-DATASET-INFO.md 数据集信息文档

用法:
    python scripts/upload_to_gdrive.py [--data-dir data/]
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import h5py


# ============================================================
# 手动上传步骤
# ============================================================

UPLOAD_INSTRUCTIONS = """
============================================================
Google Drive 手动上传步骤
============================================================

1. 访问 https://drive.google.com

2. 创建文件夹 "ViT-Ctr-Dataset"

3. 上传以下HDF5文件到该文件夹:
{file_list}

4. 对每个文件：右键 -> 共享 -> 常规访问权限改为"知道链接的任何人"

5. 复制每个文件的共享链接，提取文件ID：
   链接格式: https://drive.google.com/file/d/{{FILE_ID}}/view?usp=sharing
   文件ID是 /d/ 和 /view 之间的部分

6. 将文件ID填入 .planning/phases/02-large-scale-dataset-generation/02-DATASET-INFO.md

============================================================
"""


# ============================================================
# Colab测试代码生成
# ============================================================

COLAB_TEST_CODE = '''# ============================================================
# Colab 下载与验证测试代码
# 在Google Colab中运行此代码，验证数据集是否可正常访问
# ============================================================

# 安装依赖
!pip install gdown h5py -q

import gdown
import h5py
import numpy as np

# --- 配置区: 填入你的Google Drive文件ID ---
FILE_IDS = {
    'dithioester': 'YOUR_FILE_ID_HERE',
    'trithiocarbonate': 'YOUR_FILE_ID_HERE',
    'xanthate': 'YOUR_FILE_ID_HERE',
    'dithiocarbamate': 'YOUR_FILE_ID_HERE',
}
# --- 配置区结束 ---

# 下载所有文件
for name, file_id in FILE_IDS.items():
    if file_id == 'YOUR_FILE_ID_HERE':
        print(f"  Skipping {name}: no file ID configured")
        continue
    url = f'https://drive.google.com/uc?id={file_id}'
    output = f'{name}.h5'
    print(f"Downloading {name}...")
    gdown.download(url, output, quiet=False)

# 验证加载
total_samples = 0
for name in FILE_IDS:
    fname = f'{name}.h5'
    try:
        with h5py.File(fname, 'r') as f:
            fp_shape = f['fingerprints'].shape
            lbl_shape = f['labels'].shape
            n = fp_shape[0]
            total_samples += n

            # 加载前100个样本做快速检查
            fp_sample = f['fingerprints'][:min(100, n)]
            lbl_sample = f['labels'][:min(100, n)]

            has_nan = np.isnan(fp_sample).any() or np.isnan(lbl_sample).any()
            has_inf = np.isinf(fp_sample).any() or np.isinf(lbl_sample).any()

            status = "OK" if not (has_nan or has_inf) else "WARNING: NaN/Inf detected"
            print(f"  {name}: fingerprints={fp_shape}, labels={lbl_shape} [{status}]")
    except FileNotFoundError:
        print(f"  {name}: file not found (skipped or download failed)")

print(f"\\nTotal samples loaded: {total_samples}")
print("Dataset access test complete!")
'''


def generate_colab_test_notebook():
    """生成Colab测试代码并打印到终端。"""
    print("\n" + "=" * 60)
    print("Colab Test Code (copy-paste into a Colab notebook cell)")
    print("=" * 60)
    print(COLAB_TEST_CODE)
    return COLAB_TEST_CODE


# ============================================================
# 数据集信息文档生成
# ============================================================

def generate_dataset_info(h5_files, output_path):
    """
    生成02-DATASET-INFO.md文档，包含数据集元信息和Colab访问代码。

    Parameters
    ----------
    h5_files : list of str
        HDF5文件路径列表
    output_path : str
        输出的markdown文件路径
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # 收集文件信息
    file_rows = []
    total_samples = 0
    for fpath in h5_files:
        fname = os.path.basename(fpath)
        size_bytes = os.path.getsize(fpath)

        # 选择合适的大小单位
        if size_bytes >= 1024 * 1024 * 1024:
            size_str = f"{size_bytes / (1024**3):.1f} GB"
        elif size_bytes >= 1024 * 1024:
            size_str = f"{size_bytes / (1024**2):.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.1f} KB"

        with h5py.File(fpath, 'r') as f:
            n = f['fingerprints'].shape[0]
            total_samples += n

        file_rows.append((fname, n, size_str))

    # 生成Markdown内容
    content = f"""# Dataset Information

**Generated:** {today}
**Total samples:** {total_samples}
**Storage:** Google Drive (ViT-Ctr-Dataset folder)
**Format:** HDF5, chunked (chunk_size=1000)

## Fingerprint Specification

- **Shape:** (n, 64, 64, 2) per file
- **Channel 0:** Mn channel (normalized number-average molecular weight)
- **Channel 1:** Dispersity channel (clipped at 4.0)
- **Coordinate axes:** x = [CTA]/[M] ratio, y = conversion
- **Labels shape:** (n, 3) — [log10(Ctr), inhibition_period, retardation_factor]

## Files

| File | Samples | Size | Google Drive ID |
|------|---------|------|-----------------|
"""
    for fname, n, size_str in file_rows:
        content += f"| {fname} | {n} | {size_str} | [TBD] |\n"

    content += f"""
**Total:** {total_samples} samples

## Colab Access

```python
# 在Colab中运行
!pip install gdown h5py -q
import gdown
import h5py

# 替换为实际的Google Drive文件ID
FILE_ID = 'YOUR_FILE_ID_HERE'
gdown.download(f'https://drive.google.com/uc?id={{FILE_ID}}', 'dithioester.h5', quiet=False)

# 验证加载
with h5py.File('dithioester.h5', 'r') as f:
    print('Fingerprints shape:', f['fingerprints'].shape)
    print('Labels shape:', f['labels'].shape)
    fp_sample = f['fingerprints'][:100]
    print(f'First 100 samples loaded successfully')
```

## Notes

- Google Drive文件ID需在手动上传后填入上表
- 运行 `python scripts/upload_to_gdrive.py` 查看完整上传步骤和Colab测试代码
- 大规模数据集（~1M样本）需使用 `python src/dataset_generator.py` 生成

---

*Document auto-generated by scripts/upload_to_gdrive.py*
"""

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n  Dataset info document created: {output_path}")
    return content


# ============================================================
# 主流程
# ============================================================

def main():
    """主流程：打印上传说明、生成Colab代码、创建数据集信息文档。"""
    parser = argparse.ArgumentParser(description='ViT-Ctr Google Drive上传辅助')
    parser.add_argument('--data-dir', default='data/', help='HDF5文件目录（默认: data/）')
    args = parser.parse_args()

    data_dir = args.data_dir

    # 扫描HDF5文件
    h5_pattern = os.path.join(data_dir, '*.h5')
    h5_files = sorted(glob.glob(h5_pattern))

    if not h5_files:
        print(f"\n  WARNING: No HDF5 files found in {data_dir}")
        print("  Run 'python src/dataset_generator.py' to generate the dataset first.")
        return 1

    # 1. 打印手动上传说明
    file_list_str = ""
    for fpath in h5_files:
        size_bytes = os.path.getsize(fpath)
        if size_bytes >= 1024 * 1024 * 1024:
            size_str = f"{size_bytes / (1024**3):.1f} GB"
        elif size_bytes >= 1024 * 1024:
            size_str = f"{size_bytes / (1024**2):.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.1f} KB"
        file_list_str += f"     - {fpath} ({size_str})\n"

    print(UPLOAD_INSTRUCTIONS.format(file_list=file_list_str.rstrip()))

    # 2. 生成Colab测试代码
    generate_colab_test_notebook()

    # 3. 生成数据集信息文档
    info_path = os.path.join(
        '.planning', 'phases', '02-large-scale-dataset-generation', '02-DATASET-INFO.md'
    )
    generate_dataset_info(h5_files, info_path)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Follow the upload instructions above")
    print("  2. Fill in Google Drive file IDs in 02-DATASET-INFO.md")
    print("  3. Test Colab access using the generated test code")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
