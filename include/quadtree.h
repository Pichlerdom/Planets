#ifndef _QUADTREE_H_
#define _QUADTREE_H_

#include <stdio.h>

#include <stdlib.h>
#include <stdio.h>
#include <planets.h>
#define PLANETS_PER_NODE 10



QTree* init_qtree(float size);

void construct_qtree(QTree *qtree, Planet *planets, int size);

void insert_planet(QTree *qtree, Planet *planet);

int collaps_tree(QTree *qtree, PlanetsArr* container);

QTree *clear_qtree(QTree *qtree, int size);
QTree* clear_qtree(QTree *qtree);
#endif

