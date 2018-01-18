#include "planets.h"

void move_planets(PlanetsArr* container){

	Planet *h_planets = container->planets;

	int numberOfPlanets = container->size_arr;
	Planet *d_planets = NULL;
	float *d_dist= NULL;
	Vec *d_f=NULL;

	dim3 block(PLANET_BLOCK_N,PLANET_BLOCK_N);	
	dim3 grid(container->size_arr/PLANET_BLOCK_N,container->size_arr/PLANET_BLOCK_N);

//	printf("sa:%d  n:%d\n",container->size_arr,container->number);
/*	for(int i = 0; i < container->size_arr / PLANET_BLOCK_N; i++){
		printf("\n x:");
		for(int j = 0; j < PLANET_BLOCK_N; j++){
			printf("%.1f|",container->planets[i*PLANET_BLOCK_N + j].dir.x);
		}
		printf("\n y:");
		for(int j = 0; j < PLANET_BLOCK_N; j++){
			printf("%.1f|",container->planets[i*PLANET_BLOCK_N + j].dir.y);
		}
		printf("\n\n");
	}	*/
//	float *h_dist = (float *)malloc(numberOfPlanets * sizeof(float));	

	cudaMalloc((void **)&(d_planets), numberOfPlanets*sizeof(Planet));
	cudaMalloc((void **)&(d_dist), numberOfPlanets * numberOfPlanets * sizeof(float));	
	cudaMalloc((void **)&(d_f), numberOfPlanets * numberOfPlanets * sizeof(Vec));	

	cudaMemcpy((void*)d_planets,(void *)h_planets, numberOfPlanets*sizeof(Planet),cudaMemcpyHostToDevice);
	

	calculate_dist<<<grid, block>>>(d_planets, d_dist);

	calculate_f<<<grid, block>>>(d_planets, d_f, d_dist);		

	calculate_f_sum_reduction<<<grid,block>>>(d_f);

	calculate_f_sum<<<grid.x, block.x>>>(d_planets, d_f);
	
	move_planets_kernel<<<grid.y, block>>>(d_planets);

	hit_detection<<<grid, block>>>(d_planets, d_dist);

	cudaMemcpy((void*)h_planets,(void *)d_planets, numberOfPlanets * sizeof(Planet),cudaMemcpyDeviceToHost);
	
//	cudaMemcpy((void*)h_dist, (void *)d_dist, numberOfPlanets * sizeof(float),cudaMemcpyDeviceToHost);	

	remove_dead_planets(container);

	
	cudaFree((void **)(d_dist));
	cudaFree((void **)(d_f));
	cudaFree((void **)(d_planets));
//	free(h_dist);
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
