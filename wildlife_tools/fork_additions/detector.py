import os
from typing import List, Tuple, Union, Optional
import onnxruntime as ort
import pkg_resources
import re
import torch.nn.functional as F
from torchvision.ops import nms
from addict import Dict
import torchvision.transforms.functional as TF
import onnx
import torch
from onnxconverter_common import float16
import numpy as np


def cuda_available():
    return torch.cuda.is_available() and torch.version.cuda is not None


def get_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def parse_device_id(device: str) -> Optional[int]:
    if device == "auto":
        device = get_device()
    if device == "cpu":
        return -1
    if "cuda" in device:
        return parse_cuda_device_id(device)
    return None


def parse_cuda_device_id(device: str) -> int:
    match_result = re.match("([^:]+)(:[0-9]+)?$", device)
    assert match_result is not None, f"Can not parse device {device}."
    assert match_result.group(1).lower() == "cuda", "Not cuda device."

    device_id = 0 if match_result.lastindex == 1 else int(match_result.group(2)[1:])

    return device_id


def onnx_is_fp16(nn):
    for tensor in nn.graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT16:
            return False
    return True


def onnx_to_fp16(checkpoint):
    assert os.path.isfile(checkpoint), f"The checkpoint path: {checkpoint}, does not lead to a valid file."
    nn = onnx.load(checkpoint)
    if not onnx_is_fp16(nn):
        nn = float16.convert_float_to_float16(nn)
        onnx.save(nn, checkpoint)


def xyxy_xywh(bboxes):
    bboxes_out = bboxes.clone()
    bboxes_out[:, 2] = bboxes_out[:, 2] - bboxes_out[:, 0]
    bboxes_out[:, 3] = bboxes_out[:, 3] - bboxes_out[:, 1]
    return bboxes_out


class ImagePreprocessor:
    def __init__(self, mean, std, pad_value=114.0, input_size=(640, 640)):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.pad_value = pad_value
        self.pad_size_divisor = 1
        self.input_size = input_size

    def __call__(self, frame):
        # Get original frame dimensions (assuming frame is H, W, C)
        orig_h, orig_w, _ = frame.shape

        input_ = torch.tensor(frame).permute(2, 0, 1)
        _, img_h, img_w = input_.shape
        target_w, target_h = self.input_size
        ratio = img_w / img_h

        rsz_w = min(target_w, target_h * ratio)
        rsz_h = min(target_h, target_w / ratio)
        pad_w = target_w
        pad_h = target_h

        rsz_w, rsz_h = int(round(rsz_w)), int(round(rsz_h))
        input_ = TF.resize(input_, size=[rsz_h, rsz_w], antialias=True)

        right = pad_w - rsz_w
        bottom = pad_h - rsz_h
        input_ = F.pad(input_, (0, right, 0, bottom), value=self.pad_value)

        assert input_.dim() == 3 and input_.shape[0] == 3, f"Expected (3, H, W) tensor, got {input_.shape}"
        input_ = input_.float()

        input_ = (input_ - self.mean) / self.std

        final_h, final_w = input_.shape[1:]

        data_sample = dict(
            input_size=(rsz_w, rsz_h), input_center=(orig_w / 2.0, orig_h / 2.0), input_scale=(orig_w, orig_h)
        )

        return input_.view(1, 3, final_h, final_w), data_sample


class NMSPostProcessor:

    def __init__(
        self,
        score_thr: Optional[float] = 0.1,
        nms_thr: Optional[float] = 0.5,
        nms_pre: Optional[int] = 1000,
    ):
        super().__init__()
        assert 0 <= score_thr <= 1
        assert 0 <= nms_thr <= 1
        assert 0 <= nms_pre and isinstance(nms_pre, int)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.nms_pre = nms_pre

    def __call__(self, bboxes, scores, keypoints, kpt_vis, labels, features, priors, keep_idxs):
        valid_mask = scores > self.score_thr
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask)
        num_topk = min(self.nms_pre, valid_idxs.size(0))

        scores, idxs = scores.sort(descending=True)
        scores = scores[:num_topk]
        keep_idxs, _ = valid_idxs[idxs[:num_topk]].unbind(dim=1)

        bboxes = bboxes[keep_idxs]
        features = features[keep_idxs]
        features = F.normalize(features, p=2, dim=-1, eps=1e-12)

        if bboxes.numel() > 0:
            features_nms = features
            scores_nms = scores
            if self.nms_thr < 1.0:
                keep_idxs_nms = nms(bboxes, scores, self.nms_thr)
                if keep_idxs_nms.numel() == 0:
                    return bboxes, scores_nms, keypoints, kpt_vis, labels, features
                keep_idxs = keep_idxs[keep_idxs_nms]
                bboxes = bboxes[keep_idxs_nms]
                scores_nms = scores[keep_idxs_nms]
                features_nms = features[keep_idxs_nms]

            features = features_nms
            scores = scores_nms

        labels = labels[keep_idxs]
        kpt_vis = kpt_vis[keep_idxs]
        keypoints = keypoints[keep_idxs]
        priors = priors[keep_idxs]

        return bboxes, scores, keypoints, kpt_vis, labels, features, priors, keep_idxs


def postprocess_one_stage_detections(
    post_processor,
    scores: torch.Tensor,
    objectness: torch.Tensor,
    bboxes: torch.Tensor,
    kpts: torch.Tensor,
    kpt_vis: torch.Tensor,
    features: torch.Tensor,
    priors: torch.Tensor,
    strides: torch.Tensor,
    data_samples: List[dict],
    kpt_score_thr: Optional[int] = 0,
):
    assert bboxes.shape[0] == len(data_samples)

    scores = scores.sigmoid()
    objectness = objectness.sigmoid()
    scores *= objectness
    scores, labels = scores.max(2, keepdim=True)

    formatted_outputs = []
    for i, data_sample in enumerate(data_samples):
        i_bboxes = bboxes[i]
        i_scores = scores[i]
        i_kpts = kpts[i]
        i_kpt_vis = kpt_vis[i]
        i_labels = labels[i]
        i_features = features[i]

        i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, i_priors, i_kept_idxs = post_processor(
            i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, priors[0], torch.tensor(0)
        )

        i_scores = i_scores.flatten()
        i_kpt_vis = i_kpt_vis.sigmoid()
        i_labels = i_labels.flatten()

        input_size = data_sample["input_size"]
        input_center = data_sample["input_center"]
        input_scale = data_sample["input_scale"]

        scale = torch.tensor(input_scale, dtype=torch.float32, device=i_bboxes.device)
        rescale = scale / torch.tensor(input_size, dtype=torch.float32, device=i_bboxes.device)
        translation = torch.tensor(input_center, dtype=torch.float32, device=i_bboxes.device) - 0.5 * scale

        i_kpts = i_kpts * rescale.view(1, 1, 2) + translation.view(1, 1, 2)
        i_kpts[i_kpt_vis < kpt_score_thr] = 0.0

        i_bboxes = i_bboxes * torch.tile(rescale, (i_bboxes.shape[0], 2)) + torch.tile(
            translation, (i_bboxes.shape[0], 2)
        )
        i_bboxes = xyxy_xywh(i_bboxes)

        pred_instances = Dict()
        pred_instances.bboxes = i_bboxes
        pred_instances.scores = i_scores
        pred_instances.keypoints = i_kpts
        pred_instances.keypoint_scores = i_kpt_vis
        pred_instances.labels = i_labels
        pred_instances.features = F.normalize(i_features, p=2, dim=-1, eps=1e-12)
        pred_instances.kept_idxs = i_kept_idxs
        pred_instances.feature_maps = features[i]
        pred_instances.priors = i_priors

        formatted_pred_instances = {
            "ori_shape": getattr(data_sample, "ori_shape", None),
            "img_id": getattr(data_sample, "img_id", None),
            "seq_id": getattr(data_sample, "seq_id", None),
            "img_path": getattr(data_sample, "img_path", None),
            "id": getattr(data_sample, "id", None),
            "category_id": getattr(data_sample, "category_id", 1),
            "gt_instances": getattr(data_sample, "gt_instance_labels", None),
        }

        formatted_pred_instances["pred_instances"] = pred_instances
        formatted_outputs.append(formatted_pred_instances)
    return formatted_outputs


class ONNXDetector:
    def __init__(
        self,
        input_shapes: List[Tuple],
        checkpoint: str,
        device: str = "auto",
        mean=[0, 0, 0],
        std=[1, 1, 1],
        half_precision: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        if device is None:
            device == "auto"
        self.device = get_device() if device == "auto" else device
        self.half_precision = half_precision and self.device != "cpu"
        self.verbose = verbose
        self.checkpoint = checkpoint

        self.torch_dtype = torch.float16 if self.half_precision else torch.float32
        self.numpy_dtype = np.float16 if self.half_precision else np.float32

        self.running_batch = -1
        self.input_shapes = [(self.running_batch, 3) + torch.Size(s) for s in input_shapes]

        self.checkpoint = checkpoint
        self._live_inputs: List[ort.OrtValue] = []

        self.preprocessor = ImagePreprocessor(mean=mean, std=std)

        self.postprocessor = NMSPostProcessor()

        self._assert_runtime()

    def _assert_runtime(self) -> None:
        if self.half_precision and self.device != "cpu":
            onnx_to_fp16(self.checkpoint)

        self.device_id = parse_device_id(self.device)
        ep = list()
        if self.device == "cpu":
            ep.append("CPUExecutionProvider")
        elif "cuda" in self.device:
            ep.append(
                ("CUDAExecutionProvider", {"device_id": self.device_id}),
            )
        else:
            raise ValueError(f"The {self.device} device is not yet supported.")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(self.checkpoint, providers=ep, sess_options=so)

        self.io_binding = self.session.io_binding()
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_names = [i.name for i in self.session.get_inputs()]

        runtime = "onnxruntime"
        if "cuda" in self.device:
            runtime += "-gpu"

    def __call__(self, frame):
        input, data_sample = self.preprocessor(frame)
        outputs = self.predict(input, data_sample)
        return postprocess_one_stage_detections(self.postprocessor, *outputs, [data_sample])

    def predict(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor]],
        data_samples: List[dict],
    ) -> Tuple[torch.Tensor]:
        """Feed the tensors, run the model, return CUDA tensors."""
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        self._bind_inputs(inputs)
        self._bind_outputs()

        if "cuda" in self.device:
            torch.cuda.synchronize()

        self.session.run_with_iobinding(self.io_binding)

        outputs = tuple(torch.utils.dlpack.from_dlpack(o._ortvalue.to_dlpack()) for o in self.io_binding.get_outputs())

        self._live_inputs.clear()  # Served its purpose...
        return outputs

    def _bind_inputs(self, tensors: Tuple[torch.Tensor]):
        """Create OrtValues and bind them; resizes batch if needed."""
        for idx, (t, name, template_shape) in enumerate(zip(tensors, self.input_names, self.input_shapes)):
            B, *features_block = t.shape
            assert features_block == list(template_shape[1:]), f"Expected {template_shape[1:]}, got {features_block}"

            if B != self.running_batch:
                self.input_shapes[idx] = torch.Size([B, *features_block])
            self.running_batch = B

            ort_value = ort.OrtValue.ortvalue_from_numpy(
                t.detach().to(self.torch_dtype).cpu().contiguous().numpy(),
                self.device,
                self.device_id,
            )
            self.io_binding.bind_ortvalue_input(name, ort_value)
            self._live_inputs.append(ort_value)  # keep the pointers alive troughout gc...

    def _bind_outputs(self):
        for name in self.output_names:
            self.io_binding.bind_output(name, self.device, self.device_id)
