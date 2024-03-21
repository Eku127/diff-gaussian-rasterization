/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

// GPT: 在PyTorch的C++扩展中，可以通过 PYBIND11_MODULE 宏将C++代码绑定到Python模块。
// 该宏创建一个名为 _C 的PyBind11模块，并在其中注册C++函数，以便可以从Python中调用。
// 因此，当您在Python中调用C++扩展函数时，使用 _C 来引用该模块。 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}