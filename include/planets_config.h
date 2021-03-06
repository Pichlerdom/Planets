#ifndef _PLANETS_CONFIG_H_
#define _PLANETS_CONFIG_H_

#define __G 0.001
#define __DIST 1
#define __MOVE 1

#define __SEEDMASS 50000000


#define __MIN_R 249

#define __QTREE_SIZE 3000000.0

#define PLANET_BLOCK_DIM 1024
#define PLANET_BLOCK_N 32
#define NUMBER_OF_PLANETS PLANET_BLOCK_N * PLANET_BLOCK_DIM
#define STARTARRSIZE NUMBER_OF_PLANETS

#define COMMENT_CHAR '-'
#define END_OF_COMMAND ':'
#define END_OF_VALUE ';'

#define CONFIG_SCREEN_NAME "screen_sname"
#define CONFIG_SCREEN_HEIGHT "screen_height"
#define CONFIG_SCREEN_WIDTH "screen_width"
#define CONFIG_PLANET_SPEED "planet_speed"
#define CONFIG_PLANET_R "planet_span_r"
#define CONFIG_PLANET_M_MAX "m_max"
#define CONFIG_PLANET_M_MIN "m_min"

#define DEFAULT_SCREEN_NAME "planets"
#define DEFAULT_SCREEN_HEIGHT 1000
#define DEFAULT_SCREEN_WIDTH 1000
#define DEFAULT_PLANET_SPEED 5

#define DEFAULT_PLANET_R 250000
#define DEFAULT_PLANET_M_MAX 500001
#define DEFAULT_PLANET_M_MIN 50000

#define LINE_BUFFER_SIZE 256

#define DEFAULT_CONFIG_LOC "config.txt"

PConfig * get_planet_config();

PConfig * get_planet_config(const char* configFileLoc);

int read_config_file(char* configFileLoc, PConfig *pconfig);

void set_planet_config_default(PConfig *pconfig);

void decode_command(char* line,PConfig* pconfig);

#endif
