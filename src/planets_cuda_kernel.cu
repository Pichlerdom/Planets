
#include "planets.h"

__global__ void move_planets_kernel(Planet* d_planets){
	unsigned int i = blockDim.x * blockIdx.x +threadIdx.x;
	
	if(d_planets[i].mass > 0.0){
		d_planets[i].pos.x += (d_planets[i].dir.x * 0.001);
		d_planets[i].pos.y += (d_planets[i].dir.y * 0.001);
	}

	__syncthreads();
}


__global__ void calculate_dist(Planet* d_planets, float* d_dist){
	unsigned int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	unsigned int y = (blockDim.y * blockIdx.y) + threadIdx.y;
	int numPlan = blockDim.x * gridDim.x;
	unsigned int distIdx = (numPlan * x) + y;	

	Vec p1 = d_planets[x].pos;
	Vec p2 = d_planets[y].pos;
	float dx = (p1.x-p2.x);
	float dy = (p1.y-p2.y);

	d_dist[distIdx] = sqrtf((dx * dx) + (dy * dy));
	__syncthreads();
}

__global__ void hit_detection(Planet* d_planets, float *d_dist){
	unsigned int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	unsigned int y = (blockDim.y * blockIdx.y) + threadIdx.y;
	
	int numPlan = blockDim.x * gridDim.x;

	unsigned int i = (numPlan * x) + y;
	float r1 = d_planets[x].r;
	float r2 = d_planets[y].r;
	float mass1 = d_planets[x].mass;
	float mass2 = d_planets[y].mass;
	float dist = d_dist[i];

	if( mass2 > 0 && mass1 > 0 && r1 > r2 && 
		dist < (r2 + r1)){
		
		float mass = mass2/(mass1 );
		d_planets[x].dir.x -= (d_planets[x].dir.x - d_planets[y].dir.x)  * mass ;
		d_planets[x].dir.y -= (d_planets[x].dir.y - d_planets[y].dir.y)  * mass ;
		d_planets[x].mass += d_planets[y].mass;
		
		d_planets[x].r = log2f(d_planets[x].mass) * 0.18;
		d_planets[y].mass = -1.0;
		d_planets[y].r = 0.0;
	}
	__syncthreads();

}

__global__ void calculate_f_sum_reduction( Vec* d_f){
	__shared__ Vec sdata[PLANET_BLOCK_N * PLANET_BLOCK_N];
	
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int gx = blockIdx.x * blockDim.x + tx;
	unsigned int gy = blockIdx.y * blockDim.y + ty;

	int numPlan = blockDim.x * gridDim.x;

	unsigned int i = (gy * numPlan) + gx; 	

	sdata[ty * blockDim.x + tx] = d_f[i];
	
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
		d_f[i] = sdata[ty*blockDim.x];
	}
}


__global__ void calculate_f_sum(Planet* d_planets,Vec *d_f){

	unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int numPlan = blockDim.x * gridDim.x;
	unsigned int mass = d_planets[gx].mass * 2;
	float sum_x = 0.0;
	float sum_y = 0.0;
	for(int i = 0; i < numPlan; i += blockDim.x){
		sum_x += d_f[gx * numPlan + i].x;
		sum_y += d_f[gx * numPlan + i].y;
	}	
	__syncthreads();
	if(sum_x != 0.0){
		d_planets[gx].dir.x += sum_x/mass;
	}
	if(sum_y != 0.0){
		
		d_planets[gx].dir.y += sum_y/mass;
	}
}



__global__ void calculate_f(Planet* d_planets, Vec* d_f, float* d_dist){
	unsigned int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	unsigned int y = (blockDim.y * blockIdx.y) + threadIdx.y;
	
	unsigned int numPlan = blockDim.x * gridDim.x;
	unsigned int fIdx = (numPlan * x) + y;
		
	float dist = d_dist[fIdx];
	float mass1 = d_planets[x].mass;
	float mass2 = d_planets[y].mass;
	
	__syncthreads();
	Vec d1 = d_planets[x].pos;
	Vec d2 = d_planets[y].pos;
	float dx = d2.x - d1.x;
	float dy = d2.y - d1.y;

	__syncthreads();
	if(mass1 > 0 && mass2 > 0 && dist != 0.0){
		float g = (__G  * mass1 * mass2)/(dist * dist);
		d_f[fIdx].x = dx * (g/dist) ;
		d_f[fIdx].y = dy * (g/dist) ;
	}else{
		d_f[fIdx].x = 0.0 ;
		d_f[fIdx].y = 0.0 ;
	}
	__syncthreads();

}
