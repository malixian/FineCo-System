#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

void RandomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}


#endif