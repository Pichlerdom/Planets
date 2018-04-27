#include "planets.h"

void create_random_planet(PlanetsArr *container, PConfig *pconfig){
	bool found = false;
	int count = 0;
	Planet* planets = container->planets;
	
	int r = (int) pconfig->planet.r;
	planets[container->number].dir.x = 0;
	planets[container->number].dir.y = 0;
	while(!found){

		
		float dist_from_center = (rand()%(r * 1000))/1000.0+__MIN_R;
		float alpha = (((rand()%(90*100))/100.0) * M_PI)/180.0;
		planets[container->number].pos.x = (pconfig->screen.dim.width/2.0) + 
							sinf(alpha) * dist_from_center * 
							pow(-1,rand()%2);

		planets[container->number].pos.y = (pconfig->screen.dim.height/2.0) +
							cosf(alpha) * dist_from_center * 
							pow(-1,rand()%2);
	
		float speed = pconfig->planet.speed;
		float dy = -((pconfig->screen.dim.width)/2 - planets[container->number].pos.x);
		float dx = pconfig->screen.dim.height/2 - planets[container->number].pos.y;
	
		float dist = sqrt(dy * dy + dx * dx)*__DIST;

		planets[container->number].dir.x = dx * (speed / (dist));
		planets[container->number].dir.y = dy * (speed / (dist));

		int m_max = (int)pconfig->planet.mass_max;
		int m_min = (int)pconfig->planet.mass_min;
		planets[container->number].mass = (rand()%((m_max-m_min)*1000))/1000.0+m_min;

		if(rand()%2000 == 0){
			planets[container->number].mass *= 100;
		}
		planets[container->number].r = (sqrt(planets[container->number].mass) )/100;//log2f(planets[container->number].mass) * 0.20;

		found = true;
		for(int i = 0; i < container->number - 1; i++){
			float dy = planets[i].pos.x - planets[container->number].pos.x;
			float dx = planets[i].pos.y - planets[container->number].pos.y;
	
			float dist = sqrt(dy * dy + dx * dx)*__DIST;
			if(dist <= planets[i].r + planets[i].r){
				found = false;
				break;
			}
		}
		count++;
	}
	if(count > 200){
		printf("count:%d\n",count);
	}
}

PlanetsArr* init_planets(){
	PlanetsArr* container = (PlanetsArr*) calloc(1, sizeof(PlanetsArr));
	if(container == NULL){
		printf("Could not allocate memory for the Planets data structur!\n");
		exit(-1);
	}

	//start out by making the array a fixed size
	container->planets = (Planet*) calloc(STARTARRSIZE, sizeof(Planet));
	if(container->planets == NULL){
		free(container);
		printf("Could not allocate memory for the Planets data structur!\n");
		exit(-1);
	}
	container->size_arr = STARTARRSIZE;
	container->number = 0;
	container->quit = false;

	return container;
}

void fill_planets(PlanetsArr* container, PConfig *pconfig){
	//seed random number generator
	srand(time(0));	
	
	pthread_mutex_lock(&container->planetsMutex);

	for(int i = 0; i < NUMBER_OF_PLANETS; i++){
		create_random_planet(container, pconfig);	
		container->number++;
	}

	//point to the array that holdes all the planet structures
	container->planets[0].pos.x = pconfig->screen.dim.width/2;
	container->planets[0].pos.y = pconfig->screen.dim.height/2;
	container->planets[0].dir.x = 0;
	container->planets[0].dir.y = 0;

	container->planets[0].mass = __SEEDMASS;

	container->planets[0].r =(sqrt(container->planets[0].mass))/100;//log2f(container->planets[0].mass ) * 0.18;

	pthread_mutex_unlock(&container->planetsMutex);
}

int find_biggest_mass(PlanetsArr* container){
	int biggest = 0;
	for(int i = 0; i < container->number; i++){
		if(container->planets[biggest].mass < container->planets[i].mass){
			biggest = i;
		}
	}
	return biggest;
}
