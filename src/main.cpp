#include "planets.h"


int main(int agrc, char* args[])
{

	if(agrc == 2){
	
	PConfig *pconfig = get_planet_config();
	printf("%s\n", pconfig->screen.name);
	printf("%d|%d\n",pconfig->screen.dim.width,pconfig->screen.dim.height);
	}

	return 1;

	PlanetsArr *container = init_planets();
	fill_planets(container,pconfig);
	
	ThreadArgs *arguments = (ThreadArgs*)calloc(1,sizeof(ThreadArgs));
	arguments->container = container;
	arguments->pconfig = pconfig;
	pthread_t display_t, calc_t;

	pthread_create(&display_t,NULL,&main_display_loop,arguments);
	pthread_create(&calc_t,NULL,&main_calc_loop,args);


	pthread_join(display_t,NULL);
	pthread_join(calc_t,NULL);
	printf("Fertig.\n");	

	free(container->planets);
	free (container);
	return 0;
}

