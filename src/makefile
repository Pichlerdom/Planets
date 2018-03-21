#c compiler
CC = g++

#cuda compiler
NVCC = nvcc 

# define c++ compiler flags
CFLAGS =-g -pthread -Wall $(SDL2CFLAGS) 

#define cuda compiler flags
NVCFLAGS =-g -lineinfo -maxrregcount=0 -gencode=arch=compute_50,code=\"sm_50,compute_50\" $(SDL2CFLAGS)

#define library paths
LFLAGS = $(SDL2LFLAGS)

#define library names
LIBS = -lcuda -lcudart -lSDL2 -lSDL2_image -lSDL2_ttf -lm -lpthread
#define SDL2 Options
SDL2CFLAGS = -D_REENRRANT
SDL2INCLUDES = -I/usr/include/SDL2
SDL2LFLAGS = -L"/usr/lib/x86_64-linux-gnu"

#define include directories
INCLUDES =-I../include $(SDL2INCLUDES)

#define object files
OBJS = $(SRCS:.cpp=.o) $(NVSRCS:.cu=.o)

# define the C source files
SRCS = main.cpp planets_display.cpp  planets_func.cpp planets_config.cpp quadtree.cpp

# define the cuda source files
NVSRCS = planets_cuda_kernel.cu planets_cuda_func.cu

# define the executable file 
MAIN = ../bin/planets

# add .cu suffix so we can compile cuda files
.SUFFIXES :.cu .cpp

.PHONY:all  

all:$(MAIN) clean


$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(INCLUDES) $(LFLAGS) $(LIBS) $(SDLCONF)

%.o:%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ 
	
%.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm *.o
