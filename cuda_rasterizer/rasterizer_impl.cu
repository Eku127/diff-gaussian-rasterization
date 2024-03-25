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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// TODO: 进一步继续理解这部分代码
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// 计算2d高斯椭圆中心点points_xy在2d像素平面上占据的tile的tileID，
// 并将tileID|depth组合成64位的key值，value值为高斯球的编号
__global__ void duplicateWithKeys(
	int P,									// 高斯球数量
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,				// 累计的tiles数量的数组
	uint64_t* gaussian_keys_unsorted,		// 未排序的key
	uint32_t* gaussian_values_unsorted,		// 未排序的value
	int* radii,
	dim3 grid)								// block编号的上限
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		//第idx个高斯球前面已经占据的tiles总数
		// 前序和计算的结果
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// 计算像素点points_xy[idx]在半径为radii[idx]的圆所占据的网格编号的最小值和最大值
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// 通过这样的编码，可以让所有高斯球先通过tile进行sort，然后通过depth进行sort

		// 遍历该高斯球涉及的所有tiles，并存到gaussian_keys_unsorted和
		// gaussian_values_unsorted两个数组
		// 这两个数组之前已经使用每一个高斯占用的网格数量进行初始化内存分配过，
		// 所以每一个高斯球只要offset计算对了这边的填入也就会是对的
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// 计算当前block所在的tile ID
				uint64_t key = y * grid.x + x;
				// 左移32位，空出后面32位
				key <<= 32;
				// 后面32位存放该高斯的深度值
				key |= *((uint32_t*)&depths[idx]);
				// 记录高斯key，用于后面的排序
				gaussian_keys_unsorted[off] = key;
				// 记录高斯的id信息
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 看函数名字以及注释，大概就是去从所有的list里面去找每一个tile对应的value的信息
// 并行的是num_rendered，也就是所有的tile数量
// 会遍历所有的tile。uint2就相当于一个pair
// ranges：每一项存储对应tile的的id范围[0，L-1]，
// 这个id表示的是在point_list_keys中的索引，通过binningState.point_list找到对应高斯球编号

// 每个线程都会执行这段逻辑，但是对应不同的 idx 值，所以每个线程负责处理不同的元素。
// CUDA会自动地并行执行这些线程，每个线程独立地更新 ranges 数组的不同位置，
// 因此不需要显式地对 idx 进行递增操作。

// 函数最后的目的：ranges中的key对应每一个tile的ID，然后value是这个tile在point list中的范围，来检索到高斯球的情况
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	// 本质上idx就像是for循环中的i，这边就是在检索。因为每一个idx都会对应一个key，所以可以并行去做
	uint64_t key = point_list_keys[idx];
	// 读当前tile 的id
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 如果发现不等了，就说明进入到新的区域了
		// 这个时候就记录一下这个上一个tile（也就是之前）的end的id
		// 以及下一个新的tile id开始的位置
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
		// 如果相等就无事发生，
	}
	// 到最后了，就记录最后的 结束的位置就是L。不过为啥不是L-1
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	//参数：第三个in，第四个out，最后一个num。
	// 当第一个参数为NULL时, 所需的分配大小被写入第二个参数，并且不执行任何工作
    // https://github.com/dmlc/cub/blob/master/cub/device/device_scan.cuh
	// 源码中写的很清楚，先determine，后分配，这边就写到scan size中区了
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

// memory allocation
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
	// 从fov解算 fx fy
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// P就是三维高斯点的数量
	// 根据 GeometryState 的大小和点的数量 P 来计算所需的内存块大小
	size_t chunk_size = required<GeometryState>(P);
	// 调用rasterize_points.cu文件中的resizeFunctional函数里面
	// 嵌套的匿名函数lambda来调整显存块大小，并返回首地址
	char *chunkptr = geometryBuffer(chunk_size);
	// 分配各个内存位置,申请显存
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 计算当前图像的大小涵盖的tile的范围
	// 图像块数量
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// 块中线程数量 16*16
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// 模版函数required调用fromChunk函数来获取内存，返回结束地址，也即所需存储大小
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// 把指针（point_offsets + P - 1），也就是point_offsets数组的最后一个元素的值，
	// 赋给num_rendered，也就是总共覆盖的tiles数量
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 计算所需的BinningState的数量，即每个高斯球覆盖的tile都有对应的装箱状态BinningState数据
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	// 调整显存块大小，返回首地址
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	// 利用显存块首地址作为参数，调用fromchunk申请显存
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 16*16 = 256,每一个块一共有256个线程
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,					// 累计高斯球touch的tile数目
		binningState.point_list_keys_unsorted,		// 存储key [tile|depth]
		binningState.point_list_unsorted,			// 存储对应的高斯球id
		radii,
		tile_grid)									// 全图块数量
	CHECK_CUDA(, debug)

	// 找到最高位
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 32+bit表示了用于排序的位数
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,	// 辅助空间
		binningState.sorting_size,			// 辅助空间的大小
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug) // 排序操作涉及的tile数量（所有tile总数），开始bit位，结束bit位

	// 将内存区域imgState.ranges数据设置为0
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	// 每一个tile自己render自己的
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass

// backward的入口函数，主流程
// incoming gradient --- dL_dpixels
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}