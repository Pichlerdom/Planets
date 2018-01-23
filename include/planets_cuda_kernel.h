#ifndef _PLANETS_CUDA_KERNEL_H_
#define _PLANETS_CUDA_KERNEL_H_

#include "planets.h"

__global__ void move_planets_kernel(Planet* d_planets);

__global__ void hit_detection(Planet* d_planets);

__global__ void calculate_f_sum_reduction(Planet *d_planets, Vec* d_f);

__global__ void calculate_f_sum(Planet* d_planets, Vec* d_f);

__global__ void calculate_dist(Planet* d_planets);

__device__ Vec calculate_f(Planet p1, Planet p2);

__device__ float calculate_dist(Vec pos1, Vec pos2);

#endif


