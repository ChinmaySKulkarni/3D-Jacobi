#include "Common.h"
#include "jacobi2d.decl.h"
#include <float.h>

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int array_dim;
/*readonly*/ int block_dim;

// We want to wrap entries around, and because mod operator % sometimes misbehaves on negative values, 
// I just wrote these simple wrappers that will make the mod work as expected. -1 maps to the highest value. 

class Main : public CBase_Main
{
public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;
    double startTime;
    double maxDiff;

    Main(CkArgMsg* m) 
    {
		if (m->argc < 3) {
          CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
          CkAbort("Abort");
        }

        // set iteration counter to zero
        iterations=0;
        maxDiff = 0.0;

        // store the main proxy
        mainProxy = thisProxy;

        array_dim = atoi(m->argv[1]);
        block_dim = atoi(m->argv[2]);
        if (array_dim < block_dim || array_dim % block_dim != 0)
          CkAbort("array_size % block_size != 0!");

        // print info
        CkPrintf("Running Jacobi on %d processors with (%d,%d,%d) elements\n", CkNumPes(), (num_chare_x) , (num_chare_y) , (num_chare_z) );


        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew((num_chare_x), (num_chare_y),  (num_chare_z));

        // save the total number of worker chares we have in this simulation
        num_chares = (num_chare_x)*(num_chare_y)*(num_chare_z);

        //Start the computation
        startTime = CmiWallTimer();
        recieve_count = 0;
        array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int idx,int idy,int idz, double diff) 
    {
		//CkPrintf("(%d,%d,%d) Reporting to Main iteration# %d\n",idx, idy, idz,iterations);
        recieve_count++;
        maxDiff = ((maxDiff > diff) ? (maxDiff) : (diff));
        if (num_chares == recieve_count) {
            if (iterations>0 && maxDiff<THRESHOLD) 
            {
               	CkPrintf("Completed %d iterations; last iteration time: %.6lf, maxDiff: %f\n", iterations, CmiWallTimer() - startTime,maxDiff);
                CkExit();
            } 
            else 
            {
                //CkPrintf("starting new iteration; iteration %d time: %.6lf, maxDiff: %f\n", iterations, CmiWallTimer() - startTime,maxDiff);
                recieve_count=0;
                iterations++;
                // Call begin_iteration on all worker chares in array
                array.begin_iteration();
                maxDiff = 0.0;
            }
        }
    }
};

class Jacobi: public CBase_Jacobi {
public:
    int task_done;
    int total_task;

    double *temperature;
    double *new_temperature;

    // Constructor, initialize values
    Jacobi() 
    {	
    	temperature = new double[(block_dim+2)*(block_dim+2)*(block_dim+2)];
    	new_temperature = new double[(block_dim+2)*(block_dim+2)*(block_dim+2)];
		total_task = 7;
		task_done = 0;
		int chare_dim = num_chare_x;

		if(wrap(thisIndex.x,chare_dim)==0 || wrap(thisIndex.x,chare_dim)==chare_dim-1)
			total_task--;
		if(wrap(thisIndex.y,chare_dim)==0 || wrap(thisIndex.y,chare_dim)==chare_dim-1)
			total_task--;
		if(wrap(thisIndex.z,chare_dim)==0 || wrap(thisIndex.z,chare_dim)==chare_dim-1)
			total_task--;
		
		for(int i=0;i<block_dim+2;++i)
		{
			for(int j=0;j<block_dim+2;++j)
			{
				for(int k=0;k<block_dim+2;++k)
				{
					access3D(temperature,i,j,k) = DBL_MIN;
				}
			}
		}
		BC();
    }

    // Enforce some boundary conditions
    //Cool around the box and heat it from center
    void BC()
    {
    	int chare_dim = (num_chare_x);
        if(thisIndex.x == 0)
		{
			// Heat left surface of leftmost chare's block
			for(int j=1;j<block_dim+1; ++j)
				for(int k=1;k<block_dim+1; ++k)
					access3D(temperature,1,j,k) = 100.0;
		}
		if(thisIndex.x == chare_dim-1)
		{
			// Heat left surface of leftmost chare's block
			for(int j=1;j<block_dim+1; ++j)
				for(int k=1;k<block_dim+1; ++k)
					access3D(temperature,block_dim,j,k) = 100.0;
		}
		if(thisIndex.y == 0)
		{
			// Heat top surface of topmost chare's block
			for(int i=1;i<block_dim+1; ++i)
				for(int k=1;k<block_dim+1; ++k)
					access3D(temperature,i,1,k) = 100.0;
		}
		if(thisIndex.y == chare_dim-1)
		{
			// Heat left surface of leftmost chare's block
			for(int i=1;i<block_dim+1; ++i)
				for(int k=1;k<block_dim+1; ++k)
					access3D(temperature,i,block_dim,k) = 100.0;
		}
		if(thisIndex.z == 0)
		{
			// Heat back surface of backmost chare's block
			for(int i=1;i<block_dim+1; ++i)
				for(int j=1;j<block_dim+1; ++j)
					access3D(temperature,i,j,1) = 100.0;
		}
		if(thisIndex.z == chare_dim-1)
		{
			// Heat left surface of leftmost chare's block
			for(int i=1;i<block_dim+1; ++i)
				for(int j=1;j<block_dim+1; ++j)
					access3D(temperature,i,j,block_dim) = 100.0;
		}
	}
	
    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}

    ~Jacobi() 
	{ 
		delete[] temperature;
		delete[] new_temperature;
    }

    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {

    	double left_edge[block_dim][block_dim];
    	double right_edge[block_dim][block_dim];
    	double front_edge[block_dim][block_dim];
    	double back_edge[block_dim][block_dim];
    	double top_edge[block_dim][block_dim];
    	double bottom_edge[block_dim][block_dim];
		
        for(int j=0;j<block_dim;++j)
        {
        	for(int k=0;k<block_dim;++k)
        	{
		        left_edge[j][k] =  access3D(temperature,1,j+1,k+1);
		        right_edge[j][k] = access3D(temperature,block_dim,j+1,k+1);
			}
        }
        for(int i=0;i<block_dim;++i)
        {
        	for(int j=0;j<block_dim;++j)
        	{		        
		        back_edge[i][j] =  access3D(temperature,i+1,j+1,1);
				front_edge[i][j] = access3D(temperature,i+1,j+1,block_dim);
			}
        }
        for(int i=0;i<block_dim;++i)
        {
        	for(int k=0;k<block_dim;++k)
        	{
				top_edge[i][k] =  access3D(temperature,i+1,1,k+1);
				bottom_edge[i][k] = access3D(temperature,i+1,block_dim,k+1);
			}
        }
        

		int message_len = block_dim*block_dim;
		int chare_dim = num_chare_x;
		
		// Send my left edge
		if(thisIndex.x>0)
        	thisProxy(thisIndex.x-1, thisIndex.y, thisIndex.z).ghostsFromRight(message_len, &left_edge[0][0]);
		// Send my right edge
		if(thisIndex.x<chare_dim-1)
        	thisProxy(thisIndex.x+1, thisIndex.y, thisIndex.z).ghostsFromLeft(message_len, &right_edge[0][0]);
		// Send my top edge
		if(thisIndex.y>0)
        	thisProxy(thisIndex.x, thisIndex.y-1, thisIndex.z).ghostsFromBottom(message_len, &top_edge[0][0]);
		// Send my bottom edge
		if(thisIndex.y<chare_dim-1)
        	thisProxy(thisIndex.x, thisIndex.y+1, thisIndex.z).ghostsFromTop(message_len, &bottom_edge[0][0]);
        // Send my back edge
        if(thisIndex.z>0)
        	thisProxy(thisIndex.x, thisIndex.y, thisIndex.z-1).ghostsFromFront(message_len, &back_edge[0][0]);
        // Send my front edge
        if(thisIndex.z<chare_dim-1)
        	thisProxy(thisIndex.x, thisIndex.y, thisIndex.z+1).ghostsFromBack(message_len, &front_edge[0][0]);
        
		check_and_compute();
    }

    void ghostsFromRight(int message_len, double* ghost_values) 
    {
    	for(int j=0;j<block_dim;++j)
        {
        	for(int k=0;k<block_dim;++k)
        	{
		        access3D(temperature,block_dim+1,j+1,k+1) = ghost_values[j*block_dim+k];
			}
        }
        check_and_compute();
    }

    void ghostsFromLeft(int message_len ,double* ghost_values) 
    {
        for(int j=0;j<block_dim;++j)
        {
        	for(int k=0;k<block_dim;++k)
        	{
		        access3D(temperature,0,j+1,k+1) = ghost_values[j*block_dim+k];
			}
        }
        check_and_compute();
    }

    void ghostsFromBottom(int message_len, double* ghost_values) 
    {
        for(int i=0;i<block_dim;++i)
        {
        	for(int k=0;k<block_dim;++k)
        	{
            	access3D(temperature,i+1,block_dim+1,k+1) = ghost_values[i*block_dim + k];
            }
        }
        check_and_compute();
    }

    void ghostsFromTop(int message_len, double* ghost_values) 
    {
        for(int i=0;i<block_dim;++i)
        {
        	for(int k=0;k<block_dim;++k)
        	{
            	access3D(temperature,i+1,0,k+1) = ghost_values[i*block_dim + k];
            }
        }
        check_and_compute();
    }
    
    void ghostsFromBack(int message_len, double* ghost_values) 
    {
        for(int i=0;i<block_dim;++i)
        {
        	for(int j=0;j<block_dim;++j)
        	{
            	access3D(temperature,i+1,j+1,0) = ghost_values[i*block_dim+j];
            }
        }
        check_and_compute();
    }
    
    void ghostsFromFront(int message_len, double* ghost_values) 
    {
        for(int i=0;i<block_dim;++i)
        {
        	for(int j=0;j<block_dim;++j)
        	{
            	access3D(temperature,i+1,j+1,block_dim+1) = ghost_values[i*block_dim+j];
            }
        }
        check_and_compute();
    }

    void check_and_compute() 
    {
    	double maxDiff=0.0;
    	task_done++;
		if (task_done==total_task) 
		{
			task_done = 0;
			maxDiff = compute();
			//printf("Reached Here (%d,%d,%d)\n",thisIndex.x,thisIndex.y,thisIndex.z);
			mainProxy.report(thisIndex.x,thisIndex.y,thisIndex.z,maxDiff);
		}
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    double compute() 
    {
		// We must create a new array for these values because we don't want to update any of the
		// the values in temperature[][] array until using them first. Other schemes could be used
		// to accomplish this same problem. We just put the new values in a temporary array
		// and write them to temperature[][] after all of the new values are computed.
		double maxDiff=0.0;
		int chare_dim = num_chare_x;
		for(int i=1;i<block_dim+1;++i)
		{
			for(int j=1;j<block_dim+1;++j)
			{
				for(int k=1;k<block_dim+1;++k)
				{
					double update = 0.0;
					int avg_count=0;
					update+=access3D(temperature,i,j,k); avg_count++;
					if(thisIndex.x>0 || i>1)
        			{
        				update+= access3D(temperature,i-1,j,k); avg_count++;
        			}
					if(thisIndex.x<chare_dim-1 || i<block_dim)
					{
        				update+=access3D(temperature,i+1,j,k); avg_count++;
        			}
        			if(thisIndex.y>0 || j>1)
        			{
        				update+=access3D(temperature,i,j-1,k); avg_count++;
        			}
					if(thisIndex.y<chare_dim-1 || j<block_dim)
					{
        				update+=access3D(temperature,i,j+1,k); avg_count++;
        			}
        			if(thisIndex.z>0 || k>1)
        			{
        				update+=access3D(temperature,i,j,k-1); avg_count++;
        			}
					if(thisIndex.z<chare_dim-1 || k<block_dim)
					{
        				update+=access3D(temperature,i,j,k+1); avg_count++;
        			}
        			access3D(new_temperature,i,j,k) = update/avg_count;
        		}
			}
		}

		for(int i=1;i<block_dim+1;++i)
		{
			for(int j=1;j<block_dim+1;++j)
			{
				for(int k=1;k<block_dim+1;++k)
				{
					double diff = access3D(temperature,i,j,k) - access3D(new_temperature,i,j,k);
					if (diff < 0) diff *= -1.0;
					maxDiff = ((maxDiff > diff) ? (maxDiff) : (diff));
					access3D(temperature,i,j,k) = access3D(new_temperature,i,j,k);
				}
			}
		}

		// Enforce the boundary conditions again
		BC();
		return maxDiff;
    }
};

#include "jacobi2d.def.h"
