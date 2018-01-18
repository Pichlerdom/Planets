
PConfig * get_planet_config(){
	return get_planet_config(DEFAULT_CONFIG_LOC);
}

PConfig * get_planet_config(const char* configFileLoc){


}

int read_config_file(char* configFileLoc, PConfig *pconfig){
	//open config file read only
	FILE *file_handle = fopen(configFileLoc, "r");	
	//did opening config file work?
	if(file_handle == NULL){
		printf_s("Could not open config file at: %d!", configFileLoc);
		return -2;
	}
	char *line_buffer = (char *)calloc(LINE_BUFFER_SIZE,sizeof(char));
	int line_count = 0;
	int c = 0;
	while ((c = getc(file_handle)) == EOF){
		if(c == END_OF_VALUE) {
			line_buffer[linecount] = '\0';
			decode_command(line_buffer, pconfig);
			line_count = 0;
		}else(){
			line_buffer[line_count] = (char) c;
			line_count ++;
			if(line_count == LINE_BUFFER_SIZE){
				line_count = 0;
			}
		}
	}
}

void set_planet_config_default(PConfig *pconfig){
	

}

void decode_command(char* line, PConfig* pconfig){


}
