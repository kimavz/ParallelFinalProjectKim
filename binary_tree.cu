#include <stdio.h>

__global__ void build_binary_tree(int *x, int *child, int *root, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;
	bool newBody = true;
	int rootValue = *root;

	// build binary tree
	int childPath;
	int temp;
	offset = 0;
	while((bodyIndex + offset) < n){

		if(newBody){
			newBody = false;

			temp = 0;
			childPath = 0;
			if(x[bodyIndex + offset] > rootValue){
				childPath = 1;
			}
		}
		int childIndex = child[temp*2 + childPath];

		// traverse tree until we hit leaf node
		while(childIndex >= 0){
			temp = childIndex;
			childPath = 0;
			if(x[bodyIndex + offset] > temp){
				childPath = 1;
			}

			childIndex = child[2*temp + childPath];
		}


		if(childIndex != -2){
			int locked = temp*2 + childPath;
			if(atomicCAS(&child[locked], childIndex, -2) == childIndex){
				if(childIndex == -1){
					child[locked] = x[bodyIndex + offset];
				}

				offset += stride;
				newBody = true;
			}
		}

		__syncthreads(); // not strictly needed
	}
}

int main(){
	int n = 32;

	int *h_x;	//host array
	int *d_x;	//device array
	int *h_root;
	int *d_root;
	int *h_child;
	int *d_child;

	// allocate memory
	h_x = (int*)malloc(n*sizeof(int));
	h_root = (int*)malloc(sizeof(int));
	h_child = (int*)malloc(2*(n+1)*sizeof(int));
	cudaMalloc((void**)&d_root, sizeof(int));
	cudaMalloc((void**)&d_x, n*sizeof(int));
	cudaMalloc((void**)&d_child, 2*(n+1)*sizeof(int));
	cudaMemset(d_child, -1, 2*(n+1)*sizeof(int));


	// fill h_temp and h_x arrays
	for(int i=0;i<n;i++){
		h_x[i] = i+1;
	}

	// shuffling the array
	for(int i=0;i<n;i++){
		int j = random() % (n-i);
		int temp = h_x[i];
		h_x[i] = h_x[i+j];
		h_x[i+j] = temp;
	}
	*h_root = h_x[0];

	for(int i=0;i<n;i++){
		printf("%d ", h_x[i]);
	}
	printf("\n");


	// copy data to device
	cudaMemcpy(d_root, h_root, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, n*sizeof(int), cudaMemcpyHostToDevice);


	// kernel call
	build_binary_tree<<< 16, 16>>>(d_x, d_child, d_root, n);


	// copy from device back to host
	cudaMemcpy(h_child, d_child, 2*(n+1)*sizeof(int), cudaMemcpyDeviceToHost);


	// print tree
	for(int i=0;i<2*(n+1);i++){
		printf("%d ", h_child[i]);
	}
	printf("\n");

	// free memory
	free(h_x);
	free(h_root);
	free(h_child);
	cudaFree(d_x);
	cudaFree(d_root);
	cudaFree(d_child);
}
