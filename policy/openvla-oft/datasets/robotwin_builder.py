"""
RoboTwin 通用 RLDS 数据集 Builder

用法 (从项目根目录运行):
    python policy/openvla-oft/datasets/robotwin_builder.py \
        --task_name beat_block_hammer \
        --data_dir data/beat_block_hammer/processed_openvla \
        --save_path /path/to/output

或 (从 policy/openvla-oft/ 目录运行):
    python -m datasets.robotwin_builder \
        --task_name beat_block_hammer \
        --data_dir ../../data/beat_block_hammer/processed_openvla \
        --save_path /path/to/output

会自动:
1. 动态创建名为 aloha_{task_name} 的 builder 类
2. 从 data_dir/train/*.hdf5 和 data_dir/val/*.hdf5 读取数据
3. 生成 TFRecord 到 save_path/aloha_{task_name}/（默认 ~/tensorflow_datasets/aloha_{task_name}/）

注意: 运行前需手动在 configs.py / transforms.py / mixtures.py 中注册数据集
"""

import argparse
import glob
import os
import sys
from typing import Iterator, Tuple, Any

# 确保 policy/openvla-oft/ 和 policy/openvla-oft/datasets/ 在搜索路径中，
# 这样无论从哪个目录运行都能正确导入。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENVLA_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_OPENVLA_DIR, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 禁止 TensorFlow / TFDS 联网查询（GCS / GCE 元数据），纯本地构建。
# 必须在 import tensorflow 之前设置。
os.environ["TFDS_OFFLINE"] = "1"
os.environ["NO_GCE_CHECK"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GCS_READ_CACHE_DISABLED"] = "1"
os.environ["GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import h5py
import numpy as np
import tensorflow as tf
# 禁用 TF 的 GCS 文件系统，防止 C++ 层面发起网络请求
tf.config.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds
# 禁用 tfds 的 GCS 数据集信息查询
try:
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils.gcs_dataset_info_files = lambda *a, **kw: None
    gcs_utils.is_dataset_on_gcs = lambda *a, **kw: False
except (ImportError, AttributeError):
    pass

from datasets.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """通用的 HDF5 → RLDS example 解析函数（与原 move_can_pot_builder 完全一致）"""
    print(f"[INFO] Generating examples from {len(paths)} paths")
    for path in paths:
        print(f"[INFO] Parsing file: {path}")
        with h5py.File(path, "r") as f:
            required_keys = [
                "/relative_action",
                "/head_camera_image",
                "/left_wrist_image",
                "/right_wrist_image",
                "/low_cam_image",
                "/action",
                "/seen",
            ]
            if not all(k in f for k in required_keys):
                for key in required_keys:
                    if key not in f:
                        print(f"[ERROR] Missing key: {key} in {path}")
                print(f"[WARNING] Missing expected keys in {path}, skipping")
                continue

            T = f["/action"].shape[0]
            actions = f["/action"][1:].astype(np.float32)
            head = f["/head_camera_image"][:T - 1].astype(np.uint8)
            left = f["/left_wrist_image"][:T - 1].astype(np.uint8)
            right = f["/right_wrist_image"][:T - 1].astype(np.uint8)
            low = f["/low_cam_image"][:T - 1].astype(np.uint8)
            states = f["/action"][:T - 1].astype(np.float32)
            seen = [
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in f["/seen"][()]
            ]
            T -= 1

            if not seen:
                print(f"[ERROR] No 'seen' instructions found in {path}")
                continue

            if not (
                head.shape[0]
                == left.shape[0]
                == right.shape[0]
                == low.shape[0]
                == T
                == states.shape[0]
            ):
                print(f"[ERROR] Data length mismatch in {path}")
                continue

            instruction = seen

            steps = []
            for i in range(T):
                step = {
                    "observation": {
                        "image": head[i],
                        "left_wrist_image": left[i],
                        "right_wrist_image": right[i],
                        "low_cam_image": low[i],
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == T - 1 else 0.0),
                    "is_first": np.bool_(i == 0),
                    "is_last": np.bool_(i == T - 1),
                    "is_terminal": np.bool_(i == T - 1),
                    "language_instruction": instruction,
                }
                steps.append(step)

            print(f"[INFO] Yielding {len(steps)} steps from {path}")
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}


ROBOTWIN_FEATURE_DICT = tfds.features.FeaturesDict(
    {
        "steps": tfds.features.Dataset(
            {
                "observation": tfds.features.FeaturesDict(
                    {
                        "image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                        ),
                        "left_wrist_image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                        ),
                        "right_wrist_image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                        ),
                        "low_cam_image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                        ),
                        "state": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                    }
                ),
                "action": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                "discount": tfds.features.Scalar(dtype=np.float32),
                "reward": tfds.features.Scalar(dtype=np.float32),
                "is_first": tfds.features.Scalar(dtype=np.bool_),
                "is_last": tfds.features.Scalar(dtype=np.bool_),
                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                "language_instruction": tfds.features.Sequence(tfds.features.Text()),
            }
        ),
        "episode_metadata": tfds.features.FeaturesDict(
            {
                "file_path": tfds.features.Text(),
            }
        ),
    }
)


def make_builder_class(dataset_name: str, data_dir: str):
    """动态创建一个以 dataset_name 为类名的 builder 类。

    tensorflow_datasets 要求类名 = 数据集名，
    所以这里用 type() 动态创建。
    """
    train_pattern = os.path.join(data_dir, "train", "*.hdf5")
    val_pattern = os.path.join(data_dir, "val", "*.hdf5")

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(features=ROBOTWIN_FEATURE_DICT)

    def _split_paths(self):
        train_files = sorted(glob.glob(train_pattern))
        val_files = sorted(glob.glob(val_pattern))
        print(f"[INFO] Found {len(train_files)} training files")
        print(f"[INFO] Found {len(val_files)} validation files")
        if not train_files:
            print(f"[ERROR] No training files found at: {train_pattern}")
            sys.exit(1)
        return {"train": train_files, "val": val_files}

    cls = type(
        dataset_name,
        (MultiThreadedDatasetBuilder,),
        {
            "VERSION": tfds.core.Version("1.0.0"),
            "RELEASE_NOTES": {"1.0.0": f"RoboTwin dataset: {dataset_name}"},
            "N_WORKERS": 1,
            "MAX_PATHS_IN_MEMORY": 100,
            "PARSE_FCN": staticmethod(_generate_examples),
            "_info": _info,
            "_split_paths": _split_paths,
        },
    )
    return cls


def main():
    parser = argparse.ArgumentParser(description="RoboTwin 通用 RLDS Builder")
    parser.add_argument(
        "--task_name", type=str, required=True,
        help="任务名称，如 beat_block_hammer（数据集将命名为 aloha_{task_name}）",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="预处理后的数据目录（包含 train/ 和 val/ 子目录）",
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="RLDS 数据集输出目录（默认 ~/tensorflow_datasets/）",
    )
    args = parser.parse_args()

    dataset_name = f"aloha_{args.task_name}"
    save_path = args.save_path
    if save_path is not None:
        save_path = os.path.abspath(os.path.expanduser(save_path))
        os.makedirs(save_path, exist_ok=True)

    print(f"数据集名称: {dataset_name}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {save_path or '~/tensorflow_datasets/'}")

    print(f"\n构建 RLDS 数据集...")
    BuilderClass = make_builder_class(dataset_name, args.data_dir)
    builder = BuilderClass(data_dir=save_path)
    builder.download_and_prepare()
    output_dir = save_path or os.path.expanduser("~/tensorflow_datasets")
    print(f"\n完成！TFRecord 已生成到: {output_dir}/{dataset_name}/")


if __name__ == "__main__":
    main()
