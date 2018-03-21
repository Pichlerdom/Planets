typedef struct{
	float x,y;
}Vec;

typedef struct{
	Vec pos;
	Vec dir;
	float mass;
	float r;
}Planet;

typedef struct{
	Planet *planets;
	pthread_mutex_t planetsMutex;
	int size_arr;
	int number;
	bool quit;
}PlanetsArr;

typedef struct {
	struct{
		const char* name;
		struct{
			unsigned int height;
			unsigned int width;
		}dim;
	}screen;
	float scale;
	struct{
		float speed;
		float r;
		float mass_max;
		float mass_min;
	}planet;
}PConfig;

typedef struct{
	int arr_size;
	int tree_size;
	Planet *arr;
	float size;
}QTree;

typedef struct{
	PlanetsArr *container;
	PConfig *pconfig;
	uint32_t calctime;
	QTree *qtree;
	pthread_mutex_t qtreeMutex;
}ThreadArgs;

typedef struct {
	Vec *d_f;
	Planet *d_planets;
	int size_arr;	
}GPU_Mem;
