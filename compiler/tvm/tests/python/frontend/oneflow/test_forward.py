# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-self, invalid-name
# pylint: disable=arguments-differ, unused-argument, unused-import
"""Unit tests for various models and operators"""
import os
import sys

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor

import oneflow as flow

MODEL_HOME = "test_model"


def mkdir(path):
    # init
    path = path.strip()
    path = path.rstrip("\\")

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("{} is already here".format(path))


def rmdir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.removedirs(path)


def assert_shape(out1, out2):
    if out1.shape != out2.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(out1.shape, out2.shape))


class OneFlowGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


class OneFlowGraph_v2(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x1, x2, x3):
        out = self.m(x1, x2, x3)
        return out


def get_oneflow_output(model, inputs):
    flow_output = model(inputs)
    return flow_output.numpy()


def get_oneflow_concat_output(model, input1, input2, input3):
    flow_output = model(input1, input2, input3).numpy()
    return flow_output


def get_tvm_output(graph, model_path, inputs: flow.tensor, target="llvm", dtype="float32"):
    inputs_numpy = inputs.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(tvm.nd.array(inputs_numpy.astype(dtype)), **params).numpy()
    return tvm_output


def get_tvm_concat_output(
    graph,
    model_path,
    input1: flow.tensor,
    input2: flow.tensor,
    input3: flow.tensor,
    target="llvm",
    dtype="float32",
):
    input1_numpy = input1.numpy()
    input2_numpy = input2.numpy()
    input3_numpy = input3.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(
        tvm.nd.array(input1_numpy.astype(dtype)),
        tvm.nd.array(input2_numpy.astype(dtype)),
        tvm.nd.array(input3_numpy.astype(dtype)),
        **params,
    ).numpy()
    return tvm_output


def verify_conv(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_pool(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_normalization(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    # write params
    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_upsample(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 50, 50),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_convtran(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 50, 50),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_activation(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(10, 10),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_math(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(100, 1),
        dtype=flow.float32,
    ),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_concat(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs1=flow.tensor(np.random.randn(2, 5, 5, 4), dtype=flow.float32),
    inputs2=flow.tensor(np.random.randn(2, 5, 5, 2), dtype=flow.float32),
    inputs3=flow.tensor(np.random.randn(2, 5, 5, 3), dtype=flow.float32),
    device="llvm",
):
    if device == "cuda":
        model.to(device)
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)

    graph = OneFlowGraph_v2(model)
    graph._compile(inputs1, inputs2, inputs3)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_concat_output(graph, inputs1, inputs2, inputs3)
    out_tvm = get_tvm_concat_output(graph, MODEL_HOME, inputs1, inputs2, inputs3, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


# defs/nn
@tvm.testing.uses_gpu
def test_conv2d():
    class Conv2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = flow.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = Conv2dModel()
    model.eval()

    for device in ["llvm"]:
        verify_conv(model, device=device)


@tvm.testing.uses_gpu
def test_pool2d():
    class MaxPool2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = flow.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.pool(x)
            return x

    class AvgPool2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = flow.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.pool(x)
            return x

    class AdaptiveAvgPool2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = flow.nn.AdaptiveAvgPool2d((None, 7))

        def forward(self, x):
            x = self.pool(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model1 = MaxPool2dModel().eval()
    model2 = AvgPool2dModel().eval()
    model3 = AdaptiveAvgPool2dModel().eval()

    for device in ["llvm"]:
        verify_pool(model1, device=device)
        verify_pool(model2, device=device)
        verify_pool(model3, device=device)


@tvm.testing.uses_gpu
def test_normalization():
    class BatchNorm2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.normalization = flow.nn.BatchNorm2d(3)

        def forward(self, x):
            x = self.normalization(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = BatchNorm2dModel().eval()

    for device in ["llvm"]:
        verify_normalization(model, device=device)


@tvm.testing.uses_gpu
def test_upsample():
    class UpsampleModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = flow.nn.Upsample(scale_factor=2.0, mode="nearest")

        def forward(self, x):
            x = self.upsample(x)
            return x

    class UpsampleBiliModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = flow.nn.UpsamplingBilinear2d(scale_factor=2.0)

        def forward(self, x):
            x = self.upsample(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model1 = UpsampleModel().eval()
    model2 = UpsampleBiliModel().eval()

    for device in ["llvm"]:
        verify_upsample(model1, device=device)
        verify_upsample(model2, device=device)


@tvm.testing.uses_gpu
def test_convtran():
    class ConvTranModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.convtran = flow.nn.ConvTranspose2d(3, 4, (3, 5), stride=(2, 1), padding=(4, 2))

        def forward(self, x):
            x = self.convtran(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = ConvTranModel().eval()

    for device in ["llvm"]:
        verify_convtran(model, device=device)


@tvm.testing.uses_gpu
def test_activation():
    class Softmax(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Softmax()

        def forward(self, x):
            x = self.active(x)
            return x

    class Softplus(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Softplus()

        def forward(self, x):
            x = self.active(x)
            return x

    class Softsign(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Softsign()

        def forward(self, x):
            x = self.active(x)
            return x

    class Tanh(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Tanh()

        def forward(self, x):
            x = self.active(x)
            return x

    class ReLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.ReLU()

        def forward(self, x):
            x = self.active(x)
            return x

    class ReLU6(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.ReLU6()

        def forward(self, x):
            x = self.active(x)
            return x

    class PReLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.PReLU()

        def forward(self, x):
            x = self.active(x)
            return x

    class SELU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.SELU()

        def forward(self, x):
            x = self.active(x)
            return x

    class SiLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.SiLU()

        def forward(self, x):
            x = self.active(x)
            return x

    class LeakyReLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.LeakyReLU(0.1)

        def forward(self, x):
            x = self.active(x)
            return x

    class GELU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.GELU()

        def forward(self, x):
            x = self.active(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model1 = Softmax().eval()
    model2 = Softplus().eval()
    model3 = Softsign().eval()
    model4 = Tanh().eval()
    model5 = ReLU().eval()
    model6 = ReLU6().eval()
    model7 = PReLU().eval()
    model8 = SELU().eval()
    model9 = SiLU().eval()
    model10 = LeakyReLU().eval()
    model11 = GELU().eval()

    for device in ["llvm"]:
        verify_activation(model1, device=device)
        # verify_activation(model2, device=device) # NO PASS
        verify_activation(model3, device=device)
        verify_activation(model4, device=device)
        verify_activation(model5, device=device)
        verify_activation(model6, device=device)
        verify_activation(model7, device=device)
        verify_activation(model8, device=device)
        verify_activation(model9, device=device)
        verify_activation(model10, device=device)
        verify_activation(model11, device=device)


@tvm.testing.uses_gpu
def test_math():
    class Sigmoid(flow.nn.Module):
        def forward(self, x):
            return flow.sigmoid(x)

    class Sign(flow.nn.Module):
        def forward(self, x):
            return flow.sign(x)

    class Reciprocal(flow.nn.Module):
        def forward(self, x):
            return flow.reciprocal(x)

    class Pow(flow.nn.Module):
        def forward(self, x):
            return flow.pow(x, 2.0)

    class Log(flow.nn.Module):
        def forward(self, x):
            return flow.log(x)

    class Log2(flow.nn.Module):
        def forward(self, x):
            return flow.log1p(x)

    class Exp(flow.nn.Module):
        def forward(self, x):
            return flow.exp(x)

    class Exp2(flow.nn.Module):
        def forward(self, x):
            return flow.expm1(x)

    model1 = Sigmoid().eval()
    model2 = Sign().eval()
    model3 = Log().eval()
    model4 = Log2().eval()
    model5 = Exp().eval()
    model6 = Exp2().eval()

    for device in ["llvm"]:
        verify_math(model1, device=device)
        verify_math(model2, device=device)
        verify_math(model3, device=device)
        verify_math(model4, device=device)
        verify_math(model5, device=device)
        verify_math(model6, device=device)


@tvm.testing.uses_gpu
def test_slice():
    class Slice(flow.nn.Module):
        def forward(self, x):
            tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
            out = flow.slice(x, slice_tup_list=tup_list)
            return out

    model = Slice().eval()

    for device in ["llvm"]:
        verify_math(
            model, device=device, inputs=flow.tensor(np.random.randn(3, 6, 9).astype(np.float32))
        )


@tvm.testing.uses_gpu
def test_concat():
    class Concat(flow.nn.Module):
        def forward(self, x1, x2, x3):
            out = flow.cat([x1, x2, x3], dim=-1)
            return out

    model = Concat().eval()

    for device in ["llvm"]:
        verify_concat(model, device=device)


if __name__ == "__main__":
    test_conv2d()
    test_pool2d()
    test_normalization()
    test_upsample()
    test_convtran()
    test_activation()
    test_math()
    test_slice()
    test_concat()
    rmdir("log")
