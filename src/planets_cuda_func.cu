#include "planets.h"
#include "quadtree.h"

void move_planets(PlanetsArr* container, PConfig *pconfig, GPU_Mem *gpu_mem){

	Planet *h_planets = container->planets;

	int numberOfPlanets = container->size_arr;
	dim3 block(PLANET_BLOCK_N,PLANET_BLOCK_N);	
	dim3 grid(container->size_arr/PLANET_BLOCK_N,container->size_arr/PLANET_BLOCK_N);
	
	realloc_cuda(container, gpu_mem);

//	printf("sa:%d  n:%d\n",container->size_arr,container->number);
	/*for(int i = 0; i < container->size_arr / PLANET_BLOCK_N; i++){
		printf("\n x:");
		for(int j = 0; j < PLANET_BLOCK_N; j++){
			printf("%.1f|",container->planets[i*PLANET_BLOCK_N + j].dir.x);
		}
		printf("\n y:");
		for(int j = 0; j < PLANET_BLOCK_N; j++){
			printf("%.1f|",container->planets[i*PLANET_BLOCK_N + j].dir.y);
		}
		printf("\n\n");
	}*/
	//	float *h_dist = (float *)malloc(numberOfPlanets * sizeof(float));	

	cudaThreadSynchronize();
		cudaMemcpy((void*)gpu_mem->d_planets,
				   (void *)h_planets,
					gpu_mem->size_arr * sizeof(Planet),
				   cudaMemcpyHostToDevice);

	calculate_f_sum_reduction<<<grid,block>>>(gpu_mem->d_planets,gpu_mem->d_f);

	calculate_f_sum<<<grid.x,block.x>>>(gpu_mem->d_planets,gpu_mem->d_f);

	move_planets_kernel<<<grid.y, block>>>(gpu_mem->d_planets);
	
	//hit_detection<<<grid, block>>>(gpu_mem->d_planets);
		
	cudaMemcpy(	(void *)h_planets,
				(void *)gpu_mem->d_planets,
				numberOfPlanets * sizeof(Planet),
				cudaMemcpyDeviceToHost);
//	cudaMemcpy((void*)h_dist, (void *)d_dist, numberOfPlanets * sizeof(float),cudaMemcpyDeviceToHost);	


	

}

void realloc_cuda(PlanetsArr* container, GPU_Mem *gpu_mem){
	if(container->size_arr != gpu_mem->size_arr){	
		printf("cuda realloc:%d !\n",gpu_mem->size_arr);
		cudaFree((void **)(gpu_mem->d_f));
		cudaFree((void **)(gpu_mem->d_planets));
		gpu_mem->size_arr = container->size_arr;
		if(cudaSuccess != cudaMalloc((void **)&(gpu_mem->d_planets),
									  gpu_mem->size_arr * sizeof(Planet))){
			printf("Planets memory allocation error!\n");
			return;
		}
		if(cudaSuccess != cudaMalloc((void **)&(gpu_mem->d_f),
									 gpu_mem->size_arr * PLANET_BLOCK_DIM * sizeof(Vec))){	
			printf("Force memory allocation error!\n"); 
			return;
		}

	}
}

void remove_dead_planets(PlanetsArr* container){
	int c = container->size_arr;
	/*printf("---------------------------------------\n");
	for(int i = 0; i < container->size_arr / PLANET_BLOCK_N; i++){
		for(int j = 0; j < PLANET_BLOCK_N; j++){
			printf("%4.1f|",container->planets[i*PLANET_BLOCK_N + j].mass);
		}
		printf("\n");
	}*/
	for(int i = 0; i < c ;i++){
		if(container->planets[i].mass < 0.0 ){
			for(int j = container->size_arr - 1; j >= i; j--){
				if(container->planets[j].mass > 0.0){
					container->planets[i] = container->planets[j];
					container->planets[j].mass = -1.0;
					break;	
				}
			}
			
		}
	}
	container->number = 0;
	while(container->planets[container->number].mass >= 0.0 && container->number < container->size_arr){
		container->number ++;
	}


	int delta  =container->size_arr-1 - container->number;
	if(delta >= PLANET_BLOCK_N){ 
		container->size_arr = container->size_arr - (delta -delta%PLANET_BLOCK_N);		
		container->planets = (Planet*) realloc(container->planets,container->size_arr * sizeof(Planet));

	}
	/*for(int i = 0; i < container->size_arr / PLANET_BLOCK_N; i++){
		for(int j = 0; j < PLANET_BLOCK_N; j++){
			printf("%4.1f|",container->planets[i*PLANET_BLOCK_N + j].mass);
		}
		printf("\n");
	}*/
}



void *main_calc_loop(void *arguments){
	//Display contains all the structures we need for rendering stuff
	ThreadArgs *args = (ThreadArgs*) arguments;
	PlanetsArr *container = args->container;
	PConfig *pconfig = args->pconfig;
	uint32_t *calctime = &(args->calctime);

	uint32_t currTime = SDL_GetTicks();
	uint32_t frameTime = 0u;
	GPU_Mem gpu_mem;
	gpu_mem.size_arr = 0;
	args->qtree = init_qtree(__QTREE_SIZE);
	QTree *qtree = args->qtree;

	while(!container->quit){
		
		currTime = SDL_GetTicks();
			
		move_planets(container, pconfig, &gpu_mem);
		
	
		pthread_mutex_lock(&(args->qtreeMutex));
		qtree = clear_qtree(qtree);
		construct_qtree(qtree, container->planets, container->number);

		args->qtree = qtree;
		pthread_mutex_unlock(&(args->qtreeMutex));

		container->number = collaps_tree(qtree, container);

	
	

	

		//FPS stuff
		frameTime = SDL_GetTicks() - currTime;
		

	//	printf("calc:%d\n", frameTime);	
		*calctime = frameTime;
		if(frameTime > MS_PER_TICK){
			frameTime = MS_PER_TICK;
		}
		SDL_Delay(MS_PER_TICK-frameTime);
	}
	
	printf("Calc thread exited:\n");
	return NULL;
}
