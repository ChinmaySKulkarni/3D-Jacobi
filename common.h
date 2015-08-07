#define access3D(array,i,j,k)  array[(i)*(block_dim+2)*(block_dim+2) + (j)*(block_dim+2) + (k)]
#define num_chare_x array_dim/block_dim
#define num_chare_y array_dim/block_dim
#define num_chare_z array_dim/block_dim
#define wrap(a,b)  ((a+b)%b)
#define THRESHOLD 1.0
