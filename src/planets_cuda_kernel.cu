


#include "planets.h"

__global__ void move_planets_kernel(Planet* d_planets){
	unsigned int i = blockDim.x * blockIdx.x +threadIdx.x;
	
	if(d_planets[i].mass > 0.0){
		d_planets[i].pos.x += (d_planets[i].dir.x * __MOVE);
		d_planets[i].pos.y += (d_planets[i].dir.y * __MOVE);
	}

	__syncthreads();
}

__global__ void hit_detection(Planet* d_planets){
	unsigned int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	unsigned int y = (blockDim.y * blockIdx.y) + threadIdx.y;
	
	float r1 = d_planets[x].r;
	float r2 = d_planets[y].r;
	float mass1 = d_planets[x].mass;
	float mass2 = d_planets[y].mass;
	float dist = calculate_dist(d_planets[x].pos, d_planets[y].pos);

	if( mass2 > 0 && mass1 > 0 && r1 > r2 && 
		dist < (r2 + r1)){
	
		d_planets[x].dir.x = ((d_planets[x].dir.x * mass1 + d_planets[y].dir.x * mass2)/(mass2 + mass1) )* __MOVE;
		d_planets[x].dir.y = ((d_planets[x].dir.y * mass1 + d_planets[y].dir.y * mass2)/(mass2 + mass1) )* __MOVE;
		d_planets[x].mass += d_planets[y].mass;
		
		d_planets[x].r = log2f(d_planets[x].mass) * 0.18;
		d_planets[y].mass = -1.0;
		d_planets[y].r = 0.0;
	}
	__syncthreads();
}



__global__ void calculate_f_sum_reduction(Planet *planets, Vec* d_f){

	__shared__ Vec sdata[PLANET_BLOCK_N * PLANET_BLOCK_N];
	
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int gx = blockIdx.x * blockDim.x + tx;
	unsigned int gy = blockIdx.y * blockDim.y + ty;

	unsigned int i = (gy * gridDim.y) + blockIdx.x; 	
	Planet planet1 = planets[gx];
	Planet planet2 = planets[gy];

 
	sdata[ty * blockDim.x + tx] = calculate_f(planet1, planet2);

	__syncthreads();
	for(unsigned int swap = blockDim.x/2;swap > 0;swap >>= 1){
		if(tx < swap){
			sdata[ty*blockDim.x + tx].x += sdata[ty*blockDim.x + tx + swap].x;
			sdata[ty*blockDim.x + tx].y += sdata[ty*blockDim.x + tx + swap].y;
		}
		__syncthreads();
	}

	__syncthreads();
	if(tx == 0){
		d_f[i].x = sdata[ty*blockDim.x].x;
		d_f[i].y = sdata[ty*blockDim.x].y;

	}
}


__global__ void calculate_f_sum(Planet* d_planets,Vec *d_f){

	unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int numPlan = blockDim.x * gridDim.x;
	float sum_x = 0.0;
	float sum_y = 0.0;
	for(int i = 0; i < gridDim.x; i++){
		sum_x += d_f[gx * gridDim.x + i].x;
		sum_y += d_f[gx * gridDim.x + i].y;
	}	
	__syncthreads();
	if(sum_x != 0.0){
		d_planets[gx].dir.x += sum_x* __MOVE;
	}
	
	__syncthreads();
	if(sum_y != 0.0){
		
		d_planets[gx].dir.y += sum_y* __MOVE;
	}
}



__device__ float calculate_dist(Vec pos1, Vec pos2){
	float dx = (pos1.x-pos2.x);
	float dy = (pos1.y-pos2.y);

	return sqrtf((dx * dx) + (dy * dy));
}

__device__ Vec calculate_f(Planet p1, Planet p2){
	
	float dist = calculate_dist(p1.pos, p2.pos) * __DIST;
	
	float dx = (p1.pos.x - p2.pos.x)/dist;
	float dy = (p1.pos.y - p2.pos.y)/dist;
	Vec f;

	if(p1.mass > 0 && p2.mass > 0 && dist != 0.0){
		float g = (__G * p1.mass)/(dist * dist);
		f.x = dx * g;
		f.y = dy * g;
	}else{
		f.x = 0.0;
		f.y = 0.0;
	}
	__syncthreads();
	return f;
}
