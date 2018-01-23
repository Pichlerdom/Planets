





#include "planets.h"


PConfig * get_planet_config(){
	return get_planet_config(DEFAULT_CONFIG_LOC);
}

PConfig * get_planet_config(const char* configFileLoc){
	return NULL;

}

int read_config_file(char* configFileLoc, PConfig *pconfig){
	//open config file read only
	FILE *file_handle = fopen(configFileLoc, "r");	
	//did opening config file work?
	if(file_handle == NULL){
		printf("Could not open config file at: %s!", configFileLoc);
		return -2;
	}
	char *line_buffer = (char *)calloc(LINE_BUFFER_SIZE,sizeof(char));
	int line_count = 0;
	int c = 0;
	while ((c = getc(file_handle)) == EOF){
		if(c == END_OF_VALUE){
			line_buffer[line_count] = '\0';
			
			printf("config line: %s",line_buffer);
			decode_command(line_buffer, pconfig);
			line_count = 0;
		}else if(!isspace(c)){
			line_buffer[line_count] = (char) c;
			line_count ++;
			if(line_count == LINE_BUFFER_SIZE){
				line_count = 0;
			}
		}
	}
	free(line_buffer);
	fclose(file_handle);
	return 1;
}

void set_planet_config_default(PConfig *pconfig){

	pconfig->screen.name = DEFAULT_SCREEN_NAME;
	pconfig->screen.dim.height = DEFAULT_SCREEN_HEIGHT; 
	pconfig->screen.dim.width = DEFAULT_SCREEN_WIDTH;
	pconfig->scale = 1.0;
	pconfig->planet.speed = DEFAULT_PLANET_SPEED;
	pconfig->planet.r = DEFAULT_PLANET_R; 
	pconfig->planet.mass_max = DEFAULT_PLANET_M_MAX;
	pconfig->planet.mass_min = DEFAULT_PLANET_M_MIN; 

}

void decode_command(char* line, PConfig* pconfig){
	

}
