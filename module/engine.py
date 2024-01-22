# -*- coding: utf-8 -*-

import onnxruntime as ort
import sophon.sail as sail

__all__ = ["MODEL_PATH", "DETREngine"]

MODEL_PATH = {
    "ONNX(CPU)": {
        "FP32": {
            "800x1333": "./checkpoints/DETR_R50_800x1333.onnx",
            "1333x800": "./checkpoints/DETR_R50_1333x800.onnx",
        },
        "FP16": {
            "800x1333": "./checkpoints/DETR_R50_800x1333_FP16.onnx",
            "1333x800": "./checkpoints/DETR_R50_1333x800_FP16.onnx",
        },
    },
    "BModel(TPU)": {
        "FP32": {
            "800x1333": "./checkpoints/DETR_R50_800x1333.bmodel",
            "1333x800": "./checkpoints/DETR_R50_1333x800.bmodel",
        },
        "FP16": {
            "800x1333": "./checkpoints/DETR_R50_800x1333_FP16.bmodel",
            "1333x800": "./checkpoints/DETR_R50_1333x800_FP16.bmodel",
        },
    }
}


class DETREngine(object):
    def __init__(self, backend="ONNX(CPU)", precision="FP32", input_size="800x1333"):
        self.backend = backend
        self.precision = precision
        self.input_size = input_size
        self._check_and_init_backend()

    def _check_and_init_backend(self, backend=None, precision=None):
        if backend is not None:
            self.backend = backend
        if precision is not None:
            self.precision = precision

        model_path = MODEL_PATH[self.backend][self.precision][self.input_size]

        if self.backend == "ONNX(CPU)":
            assert model_path.endswith(".onnx"), ValueError(model_path)
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            self.net = ort.InferenceSession(
                model_path, so, providers=["CPUExecutionProvider"])
        elif self.backend == "BModel(TPU)":
            assert model_path.endswith(".bmodel"), ValueError(model_path)
            self.net = sail.Engine(model_path, 0, sail.IOMode.SYSIO)
            self.graph_name = self.net.get_graph_names()[0]
        else:
            raise NotImplementedError(self.backend)

    def __call__(self, image, mask):
        assert image.ndim == 3, ValueError(image.shape)
        assert mask.ndim == 2, ValueError(mask.shape)
        image = image[None]
        mask = mask[None]
        if self.backend == "ONNX(CPU)":
            input_dict = {
                "image": image,
                "mask": mask,
            }
            output_list = self.net.run(None, input_dict)
            pred_logits, pred_boxes = output_list
        elif self.backend == "BModel(TPU)":
            input_dict = {
                "image": image,
                "mask": mask,
            }
            output = self.net.process(self.graph_name, input_dict)
            output_list = list(output.values())
            if output_list[0].shape[-1] == 4:
                pred_boxes, pred_logits = output_list
            else:
                pred_logits, pred_boxes = output_list
        else:
            raise NotImplementedError(self.backend)

        return pred_logits, pred_boxes
