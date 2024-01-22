# -*- coding: utf-8 -*-
# -*- coding: gbk -*-

import warnings
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

import gradio as gr
import pandas as pd

from module import (
    detr_preprocess,
    detr_postprocess,
    draw_bbox,
    DETREngine,
    coco_catid2name,
)


ENGINE_LIST = [
    DETREngine(input_size="800x1333"),
    DETREngine(input_size="1333x800"),
]


def detr_infer(input, backend, precision, threshold, target_cls):
    if ENGINE_LIST[0].backend != backend or ENGINE_LIST[0].precision != precision:
        ENGINE_LIST[0]._check_and_init_backend(backend=backend, precision=precision)

    if ENGINE_LIST[1].backend != backend or ENGINE_LIST[1].precision != precision:
        ENGINE_LIST[1]._check_and_init_backend(backend=backend, precision=precision)

    image, mask, ori_shape = detr_preprocess(input, [800, 1333])
    if mask.shape[0] == 800:
        pred_logits, pred_boxes = ENGINE_LIST[0](image, mask)
        scores, labels, boxes = detr_postprocess(pred_logits, pred_boxes, ori_shape)
    else:
        pred_logits, pred_boxes = ENGINE_LIST[1](image, mask)
        scores, labels, boxes = detr_postprocess(pred_logits, pred_boxes, ori_shape)

    output, json_out = draw_bbox(input, [scores, labels, boxes], threshold, target_cls)

    return output, pd.DataFrame(json_out), json_out


def init():
    return None, "ONNX(CPU)", "FP32", 0.5, [], None, None, None


def set_example(*args):
    return args


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.")
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        type=str,
        help="Demo server name.")
    parser.add_argument(
        "--server-port",
        default=8888,
        type=int,
        help="Demo server port.")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("# <center>DETR基于Sophon BM1684X TPU的部署</center>")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    input = gr.Image(label="待推理图像输入", visible=True)
                with gr.Row():
                    backend = gr.Radio(choices=["ONNX(CPU)", "BModel(TPU)"],
                                       value="ONNX(CPU)", label="推理后端", interactive=True)
                    precision = gr.Radio(choices=["FP32", "FP16"],
                                         value="FP32", label="模型精度", interactive=True)
                with gr.Row():
                    threshold = gr.Slider(minimum=0, maximum=1, value=0.5,
                                          label="可视化阈值", interactive=True)
                with gr.Row():
                    target_label = gr.Dropdown(choices=list(coco_catid2name.values(), ),
                                               multiselect=True, label="目标类别(COCO)",
                                               interactive=True)

                with gr.Row():
                    run_btn = gr.Button(value="🚀 运行")
                    clear_btn = gr.Button(value="🧹 初始化")

            with gr.Column(scale=5):
                with gr.Tab("图像"):
                    output = gr.Image(label="结果可视化图像输出", interactive=False, visible=True)
                with gr.Tab("DataFrame"):
                    data_frame_res = gr.DataFrame(label="推理结果DataFrame格式输出", height=1000,
                                                  interactive=False)
                with gr.Tab("JSON"):
                    json_res = gr.JSON(label="推理结果JSON格式输出")

        run_btn.click(fn=detr_infer, inputs=[input, backend, precision, threshold, target_label],
                      outputs=[output, data_frame_res, json_res])
        clear_btn.click(fn=init, inputs=[],
                        outputs=[input, backend, precision, threshold, target_label, output,
                                 data_frame_res, json_res])

        gr.Examples(
            examples=[
                ["./images/000000006249.jpg", "ONNX(CPU)", "FP32", 0.5,
                 ["zebra", ], None, None, None],
                ["./images/000000007775.jpg", "BModel(TPU)", "FP32", 0.6,
                 [], None, None, None],
                ["./images/000000009502.jpg", "BModel(TPU)", "FP16", 0.7,
                 ["person", ], None, None, None],
            ],
            inputs=[input, backend, precision, threshold, target_label, output,
                    data_frame_res, json_res],
            outputs=[input, backend, precision, threshold, target_label, output,
                     data_frame_res, json_res],
            fn=set_example, run_on_click=True, label="输入样例")

    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port)
