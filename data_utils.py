"""
FedProRef: Data utilities
- CIFAR-10/100 loading with Dirichlet non-IID partition
- TinyImageNet loading with Dirichlet non-IID partition
- PACS/OfficeHome multi-domain loading following the GGEUR one-domain-one-client setup
- CLIP feature extraction (cached to disk)
"""
import json
import os
import time
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from backbone_utils import (
    open_clip_load_kwargs,
    resolve_feature_dim,
    resolve_pretrained_identity,
)


MULTI_DOMAIN_SPECS = {
    "pacs": {
        "root": "PACS",
        "domains": ["photo", "art_painting", "cartoon", "sketch"],
        "num_classes": 7,
        "split": "pacs",
    },
    "officehome": {
        "root": "Office-Home",
        "domains": ["Art", "Clipart", "Product", "Real_World"],
        "num_classes": 65,
        "split": "officehome",
    },
}


def is_multi_domain_dataset(dataset_name: str) -> bool:
    return dataset_name.lower() in MULTI_DOMAIN_SPECS


# ── Dataset wrapper ──────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """Simple dataset holding pre-extracted features and labels."""
    def __init__(self, features, labels, device=None):
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
        if device is not None:
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── TinyImageNet loader ───────────────────────────────────────────────

def _load_tinyimagenet(data_dir: str):
    """
    Load TinyImageNet (200 classes, 64x64).
    Downloads and extracts automatically if not present.
    Returns (train_images, train_labels, val_images, val_labels) as uint8 numpy arrays.
    """
    from PIL import Image as PILImage

    root = os.path.join(data_dir, "tiny-imagenet-200")
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    # Download if needed
    if not os.path.isdir(root):
        if not os.path.exists(zip_path):
            print(f"Downloading TinyImageNet from {url} ...")
            urllib.request.urlretrieve(url, zip_path)
        print("Extracting TinyImageNet ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)

    # Build class → integer label mapping from wnids.txt
    wnids_file = os.path.join(root, "wnids.txt")
    with open(wnids_file, 'r') as f:
        wnids = [line.strip() for line in f if line.strip()]
    wnid2label = {w: i for i, w in enumerate(sorted(wnids))}

    # Load training set
    train_dir = os.path.join(root, "train")
    train_images, train_labels = [], []
    for wnid in sorted(os.listdir(train_dir)):
        if wnid not in wnid2label:
            continue
        label = wnid2label[wnid]
        img_dir = os.path.join(train_dir, wnid, "images")
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            img = PILImage.open(os.path.join(img_dir, fname)).convert("RGB")
            train_images.append(np.array(img, dtype=np.uint8))
            train_labels.append(label)

    # Load validation set using val_annotations.txt
    val_ann = os.path.join(root, "val", "val_annotations.txt")
    val_img_dir = os.path.join(root, "val", "images")
    fname2wnid = {}
    with open(val_ann, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                fname2wnid[parts[0]] = parts[1]

    val_images, val_labels = [], []
    for fname in sorted(fname2wnid.keys()):
        wnid = fname2wnid[fname]
        if wnid not in wnid2label:
            continue
        img_path = os.path.join(val_img_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = PILImage.open(img_path).convert("RGB")
        val_images.append(np.array(img, dtype=np.uint8))
        val_labels.append(wnid2label[wnid])

    train_images = np.stack(train_images)   # (100000, 64, 64, 3)
    val_images   = np.stack(val_images)     # (10000,  64, 64, 3)
    train_labels = np.array(train_labels, dtype=np.int64)
    val_labels   = np.array(val_labels,   dtype=np.int64)

    return train_images, train_labels, val_images, val_labels


def _domain_imagefolder_samples(domain_root: str, class_to_idx=None):
    ds = datasets.ImageFolder(domain_root)
    if class_to_idx is None:
        class_to_idx = ds.class_to_idx
    elif set(ds.class_to_idx.keys()) != set(class_to_idx.keys()):
        missing = sorted(set(class_to_idx.keys()) - set(ds.class_to_idx.keys()))
        extra = sorted(set(ds.class_to_idx.keys()) - set(class_to_idx.keys()))
        raise ValueError(
            f"Class mismatch in {domain_root}: missing={missing}, extra={extra}")

    samples = []
    for path, local_label in ds.samples:
        class_name = ds.classes[local_label]
        samples.append((path, class_to_idx[class_name]))
    return samples, class_to_idx


def _split_train_test_indices(n_samples: int, train_ratio: float, rng: np.random.RandomState):
    indices = rng.permutation(n_samples)
    n_train = int(train_ratio * n_samples)
    return indices[:n_train].tolist(), indices[n_train:].tolist()


def _group_indices_by_label(samples, indices, num_classes: int):
    grouped = {c: [] for c in range(num_classes)}
    for idx in indices:
        _, label = samples[idx]
        grouped[label].append(idx)
    return grouped


def _load_multidomain_dataset(dataset_name: str, data_dir: str, alpha: float, seed: int):
    spec = MULTI_DOMAIN_SPECS[dataset_name.lower()]
    dataset_root = os.path.join(data_dir, spec["root"])
    if not os.path.isdir(dataset_root):
        domains = ", ".join(spec["domains"])
        raise FileNotFoundError(
            f"Expected {dataset_name} data under {dataset_root}. "
            f"It should contain domain folders: {domains}")

    rng = np.random.RandomState(seed)
    class_to_idx = None
    all_client_paths = []
    all_client_labels = []
    client_indices = []
    test_paths = []
    test_labels = []
    test_domain_ids = []
    offset = 0

    if spec["split"] == "officehome":
        dirichlet_matrix = rng.dirichlet(
            np.repeat(alpha, len(spec["domains"])),
            size=spec["num_classes"],
        ).T
    else:
        dirichlet_matrix = None

    for domain_idx, domain in enumerate(spec["domains"]):
        domain_root = os.path.join(dataset_root, domain)
        samples, class_to_idx = _domain_imagefolder_samples(domain_root, class_to_idx)
        train_indices, domain_test_indices = _split_train_test_indices(
            len(samples), 0.7, rng)

        if spec["split"] == "pacs":
            n_client = min(int(len(samples) * 0.3), len(train_indices))
            selected = rng.choice(train_indices, n_client, replace=False).tolist()
        else:
            grouped = _group_indices_by_label(samples, train_indices, spec["num_classes"])
            selected = []
            for c in range(spec["num_classes"]):
                cls_indices = grouped[c]
                if len(cls_indices) == 0:
                    continue
                n_take = int(dirichlet_matrix[domain_idx, c] * len(cls_indices))
                selected.extend(cls_indices[:n_take])

        selected = sorted(selected)
        client_paths = [samples[i][0] for i in selected]
        client_labels = [samples[i][1] for i in selected]
        all_client_paths.extend(client_paths)
        all_client_labels.extend(client_labels)
        client_indices.append(np.arange(offset, offset + len(client_paths), dtype=np.int64))
        offset += len(client_paths)

        test_paths.extend(samples[i][0] for i in domain_test_indices)
        test_labels.extend(samples[i][1] for i in domain_test_indices)
        test_domain_ids.extend([domain_idx] * len(domain_test_indices))

        counts = np.bincount(np.asarray(client_labels, dtype=np.int64),
                             minlength=spec["num_classes"])
        print(f"  Domain client {domain_idx} ({domain}): "
              f"train={len(client_paths)} test={len(domain_test_indices)} per_class={counts}")

    return (list(all_client_paths),
            np.asarray(all_client_labels, dtype=np.int64),
            list(test_paths),
            np.asarray(test_labels, dtype=np.int64),
            client_indices,
            np.asarray(test_domain_ids, dtype=np.int64),
            list(spec["domains"]))


# ── Data loading ─────────────────────────────────────────────────────

def load_raw_dataset(dataset_name: str, data_dir: str, alpha: float = 0.1, seed: int = 42):
    """Returns raw train/test data. Multi-domain datasets also return client indices."""
    os.makedirs(data_dir, exist_ok=True)

    if is_multi_domain_dataset(dataset_name):
        return _load_multidomain_dataset(dataset_name, data_dir, alpha, seed)

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(data_dir, train=True, download=True)
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True)
        train_images = train_ds.data  # (N,32,32,3) uint8
        train_labels = np.array(train_ds.targets)
        test_images = test_ds.data
        test_labels = np.array(test_ds.targets)

    elif dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(data_dir, train=True, download=True)
        test_ds = datasets.CIFAR100(data_dir, train=False, download=True)
        train_images = train_ds.data
        train_labels = np.array(train_ds.targets)
        test_images = test_ds.data
        test_labels = np.array(test_ds.targets)

    elif dataset_name == "tinyimagenet":
        train_images, train_labels, test_images, test_labels = _load_tinyimagenet(data_dir)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_images, train_labels, test_images, test_labels


# ── Dirichlet partition ──────────────────────────────────────────────

def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        num_classes: int, min_require_size: int = 0,
                        seed: int = 42,
                        max_attempts: int = 50):
    """
    Partition dataset indices into `num_clients` using Dirichlet distribution.
    For floor-zero partitions, empty clients are repaired deterministically by
    moving one sample from the largest donor client after sampling attempts are
    exhausted. This keeps client-class cells free to remain empty.
    Returns: list of np.ndarray (one per client) with indices into `labels`.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_require_size < 0:
        raise ValueError("min_require_size must be nonnegative")
    if len(labels) < num_clients:
        raise ValueError(
            "dataset must contain at least one sample per client")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    rng = np.random.RandomState(seed)
    indices_by_class = [np.where(labels == c)[0] for c in range(num_classes)]

    if min_require_size > 0:
        required_per_class = num_clients * min_require_size
        client_indices = [[] for _ in range(num_clients)]
        for class_id, idx_c in enumerate(indices_by_class):
            if len(idx_c) < required_per_class:
                raise ValueError(
                    f"class {class_id} has {len(idx_c)} samples, but "
                    f"min_require_size={min_require_size} with {num_clients} "
                    f"clients requires {required_per_class}")

            shuffled = rng.permutation(idx_c)
            seeded = shuffled[:required_per_class]
            for client_id in range(num_clients):
                start = client_id * min_require_size
                stop = start + min_require_size
                client_indices[client_id].extend(seeded[start:stop].tolist())

            remaining = shuffled[required_per_class:]
            if len(remaining) > 0:
                proportions = rng.dirichlet(np.repeat(alpha, num_clients))
                counts = rng.multinomial(len(remaining), proportions)
                splits = np.split(remaining, np.cumsum(counts[:-1]))
                for client_id, client_split in enumerate(splits):
                    client_indices[client_id].extend(client_split.tolist())

        finalized = []
        for indices in client_indices:
            shuffled_client = np.asarray(indices, dtype=np.int64)
            rng.shuffle(shuffled_client)
            finalized.append(shuffled_client)
        return finalized

    for _ in range(max_attempts):
        client_indices = [[] for _ in range(num_clients)]

        for idx_c in indices_by_class:
            if len(idx_c) == 0:
                continue
            shuffled = rng.permutation(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            counts = rng.multinomial(len(shuffled), proportions)
            split = np.split(shuffled, np.cumsum(counts[:-1]))
            for client_id, client_split in enumerate(split):
                client_indices[client_id].extend(client_split.tolist())

        if all(client_indices):
            return [np.asarray(ci, dtype=np.int64) for ci in client_indices]

    empty_clients = [
        client_id for client_id, indices in enumerate(client_indices)
        if not indices
    ]
    for empty_client in empty_clients:
        donor = max(
            (client_id for client_id, indices in enumerate(client_indices)
             if len(indices) > 1),
            key=lambda client_id: (len(client_indices[client_id]), -client_id),
        )
        moved_offset = int(rng.randint(len(client_indices[donor])))
        client_indices[empty_client].append(
            client_indices[donor].pop(moved_offset))

    return [np.asarray(ci, dtype=np.int64) for ci in client_indices]


# ── CLIP feature extraction ─────────────────────────────────────────

def extract_clip_features(images: np.ndarray, backbone: str, pretrained: str,
                          device: str, batch_size: int = 256,
                          identity: dict | None = None):
    """
    Extract L2-normalized CLIP features from uint8 images.
    Returns: np.ndarray of shape (N, feat_dim).
    """
    import open_clip
    from PIL import Image as PILImage

    identity = identity or resolve_pretrained_identity(backbone, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained, **open_clip_load_kwargs(identity))
    model = model.eval().to(device)

    all_features = []
    n = len(images)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for i in range(start, end):
            item = images[i]
            if isinstance(item, (str, os.PathLike)):
                img = PILImage.open(item).convert("RGB")
            elif isinstance(item, PILImage.Image):
                img = item.convert("RGB")
            else:
                img = PILImage.fromarray(item).convert("RGB")
            batch_imgs.append(preprocess(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        # L2 normalize
        feats = feats / feats.norm(dim=1, keepdim=True)
        all_features.append(feats.cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"  Extracting features: {end}/{n}")

    return np.concatenate(all_features, axis=0)


def _partition_scheme(args) -> str:
    if is_multi_domain_dataset(args.dataset):
        return "multidomain-v1"
    return "partition-nonempty-v4"


def _expected_cache_metadata(args, identity) -> dict:
    return {
        "dataset": args.dataset,
        "backbone": identity["backbone"],
        "checkpoint_hash": identity["checkpoint_hash"],
        "alpha": float(args.alpha),
        "num_clients": int(args.num_clients),
        "partition_seed": int(getattr(args, "partition_seed", args.seed)),
        "min_require_size": int(getattr(args, "min_require_size", 0)),
        "partition_scheme": _partition_scheme(args),
    }


def _load_feature_cache(cache_file, args, identity=None):
    print(f"Loading cached features from {cache_file}")
    identity = identity or resolve_pretrained_identity(args.backbone, args.pretrained)
    with np.load(cache_file, allow_pickle=True) as data:
        if "metadata" not in data:
            raise ValueError(f"feature cache has no metadata: {cache_file}")
        metadata = json.loads(str(data["metadata"].item()))
        expected = _expected_cache_metadata(args, identity)
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"feature cache metadata mismatch for {key}: "
                    f"stored={metadata.get(key)!r}, expected={expected_value!r}")

        train_features = np.asarray(data["train_features"])
        train_labels = np.asarray(data["train_labels"])
        test_features_np = np.asarray(data["test_features"])
        test_labels_np = np.asarray(data["test_labels"])
        client_indices = data["client_indices"].copy()
        resolved_dim = resolve_feature_dim(None, train_features, test_features_np)
        if metadata.get("feature_dim") != resolved_dim:
            raise ValueError(
                f"feature cache metadata mismatch for feature_dim: "
                f"stored={metadata.get('feature_dim')!r}, resolved={resolved_dim}")

        if "test_domain_ids" in data and "test_domain_names" in data:
            args.test_domain_ids = torch.from_numpy(
                np.asarray(data["test_domain_ids"])).long()
            args.test_domain_names = [
                str(x) for x in data["test_domain_names"].tolist()]
        else:
            args.test_domain_ids = None
            args.test_domain_names = None

    args.feature_cache_file = os.path.realpath(cache_file)
    args.backbone_identity = metadata
    test_features = torch.from_numpy(test_features_np).float()
    test_labels = torch.from_numpy(test_labels_np).long()
    client_data = []
    for ci in client_indices:
        ci = np.asarray(ci, dtype=np.int64)
        client_data.append((train_features[ci], train_labels[ci]))
    return client_data, test_features, test_labels


def _acquire_cache_lock(cache_file):
    lock_file = f"{cache_file}.lock"
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return lock_file, fd
        except FileExistsError:
            time.sleep(5)


def _release_cache_lock(lock_file, fd):
    os.close(fd)
    try:
        os.unlink(lock_file)
    except FileNotFoundError:
        pass

def _feature_cache_file(args, identity=None):
    identity = identity or resolve_pretrained_identity(args.backbone, args.pretrained)
    cache_dir = os.path.join(args.data_dir, f"{args.dataset}_clip_cache")
    pretrained_tag = os.path.basename(identity["pretrained"]).replace(" ", "_")
    partition_seed = getattr(args, "partition_seed", args.seed)
    backbone_tag = args.backbone.replace("/", "_")
    min_require_size = int(getattr(args, "min_require_size", 0))
    checkpoint_tag = identity["checkpoint_hash"][:12]
    partition_tag = _partition_scheme(args)
    return os.path.join(
        cache_dir,
        f"{backbone_tag}_{pretrained_tag}_{checkpoint_tag}_minpc{min_require_size}_"
        f"{partition_tag}_"
        f"alpha{args.alpha}_c{args.num_clients}_ps{partition_seed}.npz",
    )


def get_or_extract_features(args):
    """
    Load cached features or extract them. Returns:
        client_data: list of (features_np, labels_np) per client
        test_features: torch.Tensor
        test_labels: torch.Tensor
    """
    identity = resolve_pretrained_identity(args.backbone, args.pretrained)
    cache_file = _feature_cache_file(args, identity)
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    partition_seed = getattr(args, "partition_seed", args.seed)

    if os.path.exists(cache_file):
        return _load_feature_cache(cache_file, args, identity)

    lock_file, lock_fd = _acquire_cache_lock(cache_file)
    try:
        if os.path.exists(cache_file):
            return _load_feature_cache(cache_file, args, identity)

        print("Extracting CLIP features (this may take a while)...")
        raw_data = load_raw_dataset(args.dataset, args.data_dir, args.alpha, partition_seed)
        test_domain_ids = None
        test_domain_names = None
        if len(raw_data) == 7:
            (train_images, train_labels, test_images, test_labels_np,
             client_indices, test_domain_ids, test_domain_names) = raw_data
        elif len(raw_data) == 5:
            train_images, train_labels, test_images, test_labels_np, client_indices = raw_data
        else:
            train_images, train_labels, test_images, test_labels_np = raw_data
            client_indices = None
        args.test_domain_ids = (torch.from_numpy(test_domain_ids).long()
                                if test_domain_ids is not None else None)
        args.test_domain_names = test_domain_names

        train_features = extract_clip_features(
            train_images, args.backbone, args.pretrained, args.device,
            identity=identity)
        test_features_np = extract_clip_features(
            test_images, args.backbone, args.pretrained, args.device,
            identity=identity)
        feature_dim = resolve_feature_dim(None, train_features, test_features_np)

        # Partition
        if client_indices is None:
            client_indices = dirichlet_partition(
                train_labels, args.num_clients, args.alpha,
                args.num_classes,
                min_require_size=int(getattr(args, "min_require_size", 0)),
                seed=partition_seed)

        metadata = _expected_cache_metadata(args, identity)
        metadata["feature_dim"] = feature_dim
        metadata["pretrained"] = identity["pretrained"]
        metadata["source"] = identity["source"]

        # Save cache atomically so parallel runs sharing a partition do not corrupt it.
        tmp_cache_file = f"{cache_file}.tmp.{os.getpid()}.npz"
        np.savez(tmp_cache_file,
                 metadata=np.array(json.dumps(metadata, sort_keys=True)),
                 train_features=train_features,
                 train_labels=train_labels,
                 test_features=test_features_np,
                 test_labels=test_labels_np,
                 client_indices=np.array(client_indices, dtype=object),
                 test_domain_ids=(test_domain_ids if test_domain_ids is not None else np.array([], dtype=np.int64)),
                 test_domain_names=np.array(test_domain_names or [], dtype=object))
        os.replace(tmp_cache_file, cache_file)
        args.feature_cache_file = os.path.realpath(cache_file)
        args.backbone_identity = metadata

        # Print distribution
        print("\nClient data distribution:")
        for k, ci in enumerate(client_indices):
            ci = np.asarray(ci, dtype=np.int64)
            counts = np.bincount(train_labels[ci], minlength=args.num_classes)
            print(f"  Client {k}: total={len(ci)}, per_class={counts}")

        test_features = torch.from_numpy(test_features_np).float()
        test_labels_t = torch.from_numpy(test_labels_np).long()
        client_data = []
        for ci in client_indices:
            ci = np.asarray(ci, dtype=np.int64)
            client_data.append((train_features[ci], train_labels[ci]))

    finally:
        _release_cache_lock(lock_file, lock_fd)

    return client_data, test_features, test_labels_t
