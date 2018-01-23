#include "planets.h"

void move_planets(PlanetsArr* container, PConfig *pconfig){

	Planet *h_planets = container->planets;

	int numberOfPlanets = container->size_arr;
	Planet *d_planets = NULL;
	Vec *d_f=NULL;

	dim3 block(PLANET_BLOCK_N,PLANET_BLOCK_N);	
	dim3 grid(container->size_arr/PLANET_BLOCK_N,container->size_arr/PLANET_BLOCK_N);

	printf("sa:%d  n:%d\n",container->size_arr,container->number);
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
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();

	if(cudaSuccess != cudaMalloc((void **)&(d_planets), numberOfPlanets * sizeof(Planet))){
		printf("Planets memory allocation error!\n");
		return;
	}
	if(cudaSuccess != cudaMalloc((void **)&(d_f), numberOfPlanets * PLANET_BLOCK_DIM * sizeof(Vec))){	
		printf("force memory allocation error!\n"); 
		return;
	}
	cudaMemcpy((void*)d_planets,(void *)h_planets, numberOfPlanets*sizeof(Planet),cudaMemcpyHostToDevice);

	hit_detection<<<grid, block>>>(d_planets);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();

	calculate_f_sum_reduction<<<grid,block>>>(d_planets,d_f);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();

	calculate_f_sum<<<grid.x,block.x>>>(d_planets,d_f);

	move_planets_kernel<<<grid.y, block>>>(d_planets);

	cudaMemcpy((void*)h_planets,(void *)d_planets, numberOfPlanets * sizeof(Planet),cudaMemcpyDeviceToHost);
	
//	cudaMemcpy((void*)h_dist, (void *)d_dist, numberOfPlanets * sizeof(float),cudaMemcpyDeviceToHost);	

	pthread_mutex_lock(&container->planetsMutex);

	remove_dead_planets(container);
	pthread_mutex_unlock(&container->planetsMutex);

	
	cudaFree((void **)(d_f));
	cudaFree((void **)(d_planets));
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
	
	uint32_t currTime = SDL_GetTicks();
	uint32_t frameTime = 0u;

	while(!container->quit){
		
		currTime = SDL_GetTicks();
		
		move_planets(container, pconfig);

		//FPS stuff
		frameTime = SDL_GetTicks() - currTime;
		
	
		printf("calc:%d\n", frameTime);	
		if(frameTime > MS_PER_TICK){
			frameTime = MS_PER_TICK;
		}
		SDL_Delay(MS_PER_TICK-frameTime);
	}
	
	printf("Calc thread exited:\n");
	return NULL;
}
