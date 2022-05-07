#include <stdio.h>

/*  ************************************************** FIRST LAYER START ********************************************************* */
/*
    Layer 1: Normal 3D Convolution Layer
    Input: 225 * 225 * 3 (Padding of 1)
    Weight: 3 * 3 * 3 with a Stride of 2
    Output: 112 * 112 * 32
    Next Layer is a padding layer, so padding operation is handled in this layer itself & hence
    Final Output = 114 * 114 * 32
*/
//don't need to change partA as the blockSize is not changed
__global__ void executeFirstLayer_CONV3D_partA(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (blockIdx.y * 32 * 114)    // Position in the grid row-wise
                        + (blockIdx.z * 32)          // Position in the grid column-wise
                        + (threadIdx.x * 114)
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((blockIdx.y * 32 * 225) * stride) // Position in the grid row-wise
                       + (blockIdx.z * 32 * stride)         // Position in the grid column-wise
                       + (threadIdx.x * 225 * stride )
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}

//We need to add another dimension to the partB

__global__ void executeFirstLayer_CONV3D_partB(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (blockIdx.y * 32 * 114 + 64 *114)  // Position in the grid row-wise and there is no column-wise position
						+ (blockIdx.z * 16 )
                        + (threadIdx.x * 114)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((blockIdx.y * 32 * 225) * stride) + (64 * blockIdx.z * stride) // Position in the grid row-wise and column-wise
                       + (threadIdx.x * 225 * stride)
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}

__global__ void executeFirstLayer_CONV3D_partC(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (blockIdx.y * 16 * 114)                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 32 + 64 )             // Position in the grid column-wise
                        + (threadIdx.x * 114)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((64 * 225) * stride)
                       + (blockIdx.y * 16 * stride) + (blockIdx.z * 32 * stride)     // Position in the grid row-wise and column-wise
                       + (threadIdx.x * 225 * stride)
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}

__global__ void executeFirstLayer_CONV3D_partD(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (64 * 114) + 64                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 16 * 114)
						+ (blockIdx.z * 16)			// Position in the grid column-wise
                        + (threadIdx.x * 114)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((64* 64 * 225) * stride)
                       + (blockIdx.y * 16 * stride) + (blockIdx.z * 16 * stride)     // Position in the grid row-wise and column-wise
                       + (threadIdx.x * 225 * stride)
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}
/*  ************************************************** FIRST LAYER END ************************************************************ */

/*  ************************************************** SECOND LAYER START ********************************************************* */
/*
    Layer 2: Depthwise Separable Convolution Layer
    Input: 114 * 114 * 3 (After padding)
    Weight: 3 * 3 * 32 with a Stride of 1
    Output: 112 * 112 * 32
*/
__global__ void executeSecondLayer_DSC_partA(double *Layer2_Neurons_GPU,
                            double *Layer2_Weights_GPU,
                            double *Layer3_Neurons_GPU,
                            double *Layer2_Mean_GPU,
                            double *Layer2_StanDev_GPU,
                            double *Layer2_Gamma_GPU,
                            double *Layer2_Beta_GPU
                        )
{
	double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 112 * 112)   // channel to work with
                        + (blockIdx.y * 32 * 112)    // Position in the grid row-wise
                        + (blockIdx.z * 32)          // Position in the grid column-wise
                        + (threadIdx.x * 112)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (blockIdx.y * 32 * 114) // Position in the grid row-wise
                       + (blockIdx.z * 32)         // Position in the grid column-wise
                       + (threadIdx.x * 114)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114)] * Layer2_Weights_GPU[weight_Position + (row * 3)])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 1] * Layer2_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 2] * Layer2_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer2_Mean_GPU[filter_number]) / Layer2_StanDev_GPU[filter_number];
    Z = (Z * Layer2_Gamma_GPU[filter_number]) + Layer2_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer3_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSecondLayer_DSC_partB(double *Layer2_Neurons_GPU,
                                    double *Layer2_Weights_GPU,
                                    double *Layer3_Neurons_GPU,
                                    double *Layer2_Mean_GPU,
                                    double *Layer2_StanDev_GPU,
                                    double *Layer2_Gamma_GPU,
                                    double *Layer2_Beta_GPU
                                )
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 112 * 112)   // channel to work with
                        + (blockIdx.y * 16 * 112 + 96)  // Position in the grid row-wise and there is no column-wise position
                        + (threadIdx.x * 112)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position  = (blockIdx.y * 16 * 114)
                        + (96) // Position in the grid row-wise and column-wise
                        + (threadIdx.x * 114)
                        + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114)] * Layer2_Weights_GPU[weight_Position + (row * 3)])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 1] * Layer2_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 2] * Layer2_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer2_Mean_GPU[filter_number]) / Layer2_StanDev_GPU[filter_number];
    Z = (Z * Layer2_Gamma_GPU[filter_number]) + Layer2_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer3_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSecondLayer_DSC_partC(double *Layer2_Neurons_GPU,
                                    double *Layer2_Weights_GPU,
                                    double *Layer3_Neurons_GPU,
                                    double *Layer2_Mean_GPU,
                                    double *Layer2_StanDev_GPU,
                                    double *Layer2_Gamma_GPU,
                                    double *Layer2_Beta_GPU
                                )
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 112 * 112)   // channel to work with
                        + (96 * 112)                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 16)             // Position in the grid column-wise
                        + (threadIdx.x * 112)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (96 * 114)
                        + (blockIdx.y * 16)     // Position in the grid row-wise and column-wise
                        + (threadIdx.x * 114)
                        + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114)] * Layer2_Weights_GPU[weight_Position + (row * 3)])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 1] * Layer2_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 2] * Layer2_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer2_Mean_GPU[filter_number]) / Layer2_StanDev_GPU[filter_number];
    Z = (Z * Layer2_Gamma_GPU[filter_number]) + Layer2_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer3_Neurons_GPU[output_Position] = Z;
}
/*  ************************************************** SECOND LAYER END ********************************************************* */

/*  ************************************************** THIRD LAYER START ******************************************************** */
/*
    Layer 3: Pointwise Separable Convolution Layer
    Input: 112 * 112 * 32 (After padding)
    Weight: 1 * 1 * 32 * 64 with a Stride of 1
    Output: 113 * 113 * 64 (Padding of 1 is handled in this layer execution itself)
*/
__global__ void executeThirdLayer_PSC_partA(double *Layer3_Neurons_GPU,
    double *Layer3_Weights_GPU,
    double *Layer4_Neurons_GPU,
    double *Layer3_Mean_GPU,
    double *Layer3_StanDev_GPU,
    double *Layer3_Gamma_GPU,
    double *Layer3_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 113 * 113)   // channel to work with
                        + (blockIdx.y * 32 * 113)    // Position in the grid row-wise
                        + (blockIdx.z * 32)          // Position in the grid column-wise
                        + (threadIdx.x * 113)
                        + (threadIdx.y);

    int weight_Position = filter_number * 32;

    int input_Position = (blockIdx.y * 32 * 112) // Position in the grid row-wise
                       + (blockIdx.z * 32)         // Position in the grid column-wise
                       + (threadIdx.x * 112)
                       + (threadIdx.y);

    for(int channel = 0; channel < 32; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer3_Neurons_GPU[(channel * 112 * 112) + input_Position] * Layer3_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer3_Mean_GPU[filter_number]) / Layer3_StanDev_GPU[filter_number];
    Z = (Z * Layer3_Gamma_GPU[filter_number]) + Layer3_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer4_Neurons_GPU[output_Position] = Z;
}

__global__ void executeThirdLayer_PSC_partB(double *Layer3_Neurons_GPU,
    double *Layer3_Weights_GPU,
    double *Layer4_Neurons_GPU,
    double *Layer3_Mean_GPU,
    double *Layer3_StanDev_GPU,
    double *Layer3_Gamma_GPU,
    double *Layer3_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 113 * 113)   // channel to work with
                        + (blockIdx.y * 16 * 113 + 96)  // Position in the grid row-wise and there is no column-wise position
                        + (threadIdx.x * 113)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 32;

    int input_Position = (blockIdx.y * 16 * 112)         // Position in the grid row-wise
                       + (96)                   // Position in the grid column-wise
                       + (threadIdx.x * 112)
                       + (threadIdx.y);

    for(int channel = 0 ; channel < 32 ; channel++) // Channel loop as we have 32 input channels to work with
    {
        product += (Layer3_Neurons_GPU[(channel * 112 * 112) + input_Position] * Layer3_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer3_Mean_GPU[filter_number]) / Layer3_StanDev_GPU[filter_number];
    Z = (Z * Layer3_Gamma_GPU[filter_number]) + Layer3_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer4_Neurons_GPU[output_Position] = Z;
}

__global__ void executeThirdLayer_PSC_partC(double *Layer3_Neurons_GPU,
    double *Layer3_Weights_GPU,
    double *Layer4_Neurons_GPU,
    double *Layer3_Mean_GPU,
    double *Layer3_StanDev_GPU,
    double *Layer3_Gamma_GPU,
    double *Layer3_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 113 * 113)   // channel to work with
                        + (96 * 113)                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 16)             // Position in the grid column-wise
                        + (threadIdx.x * 113)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 32;

    int input_Position = (96 * 112)            // row-wise: the bottom part of the grid after 96th row
                       + (blockIdx.y * 16)     // column-wise: block number in the 6 blocks of 16 * 16 threads
                       + (threadIdx.x * 112)   // Position inside one the above block row-wise
                       + (threadIdx.y);        // Position inside one the above block column-wise

    for(int channel = 0 ; channel < 32 ; channel++) // Channel loop as we have 32 input channels to work with
    {
        product += (Layer3_Neurons_GPU[(channel * 112 * 112) + input_Position] * Layer3_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer3_Mean_GPU[filter_number]) / Layer3_StanDev_GPU[filter_number];
    Z = (Z * Layer3_Gamma_GPU[filter_number]) + Layer3_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer4_Neurons_GPU[output_Position] = Z;
}
/*  ************************************************** THIRD LAYER END ********************************************************* */

/*  ************************************************** FOURTH LAYER START ****************************************************** */
/*
    Layer 4: Depthwise Separable Convolution Layer
    Input: 113 * 113 * 64
    Weight: 3 * 3 * 64 with a Stride of 2
    Output: 56 * 56 * 64
*/
__global__ void executeFourthLayer_DSC_partA(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 113 * stride )
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer5_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFourthLayer_DSC_partB(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y + 32);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 113 * stride)
                       + (threadIdx.y * stride)
                       + (32 * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer5_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFourthLayer_DSC_partC(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32)
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (113 * 32 * stride)
                       + (threadIdx.x * 113 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer5_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFourthLayer_DSC_partD(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32)
                        + 32
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (113 * 32 * stride)
                       + (32 * stride)
                       + (threadIdx.x * 113 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++) // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer5_Neurons_GPU[output_Position] = Z;
}
/*  ************************************************** FOURTH LAYER END ****************************************************** */

/*  *************************************************** FIFTH LAYER START **************************************************** */
/*
    Layer 5: Pointwise Separable Convolution Layer
    Input: 56 * 56 * 64
    Weight: 1 * 1 * 64 * 128 with a Stride of 1
    Output: 58 * 58 * 128 (Padding for the next layer is handled here itself)
*/
__global__ void executeFifthLayer_PSC_partA(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int offset = 59;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 58 * 58)   // channel to work with
                        + (threadIdx.x * 58)
                        + (threadIdx.y);

    int weight_Position = filter_number * 64;

    int input_Position = (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer6_Neurons_GPU[output_Position + offset] = Z;
}

__global__ void executeFifthLayer_PSC_partB(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 59;

    // Output position
    int output_Position = (filter_number * 58 * 58)   // channel to work with
                        + (threadIdx.x * 58)
                        + (threadIdx.y + 32);

    int weight_Position = filter_number * 64;

    int input_Position = (threadIdx.x * 56)
                       + (threadIdx.y)
                       + (32);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer6_Neurons_GPU[output_Position + offset] = Z;
}

__global__ void executeFifthLayer_PSC_partC(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 59;

    // Output position
    int output_Position = (filter_number * 58 * 58)   // channel to work with
                        + (58 * 32)
                        + (threadIdx.x * 58)
                        + (threadIdx.y);

    int weight_Position = filter_number * 64;

    int input_Position = (56 * 32)
                       + (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer6_Neurons_GPU[output_Position + offset] = Z;
}

__global__ void executeFifthLayer_PSC_partD(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 59;

    // Output position
    int output_Position = (filter_number * 58 * 58)   // channel to work with
                        + (58 * 32)
                        + 32
                        + (threadIdx.x * 58)
                        + (threadIdx.y);

    int weight_Position = filter_number * 64;

    int input_Position = (56 * 32)
                       + (32)
                       + (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer6_Neurons_GPU[output_Position + offset] = Z;
}
/*  *************************************************** FIFTH LAYER END **************************************************** */

/*  *************************************************** SIXTH LAYER START ************************************************** */
/*
    Layer 6: Depthwise Separable Convolution Layer
    Input: 58 * 58 * 128
    Weight: 3 * 3 * 128 with a Stride of 1
    Output: 56 * 56 * 128
*/
__global__ void executeSixthLayer_DSC_partA(double *Layer6_Neurons_GPU,
    double *Layer6_Weights_GPU,
    double *Layer7_Neurons_GPU,
    double *Layer6_Mean_GPU,
    double *Layer6_StanDev_GPU,
    double *Layer6_Gamma_GPU,
    double *Layer6_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 58)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58)] * Layer6_Weights_GPU[weight_Position + (row * 3)])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 1] * Layer6_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 2] * Layer6_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer6_Mean_GPU[filter_number]) / Layer6_StanDev_GPU[filter_number];
    Z = (Z * Layer6_Gamma_GPU[filter_number]) + Layer6_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer7_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSixthLayer_DSC_partB(double *Layer6_Neurons_GPU,
    double *Layer6_Weights_GPU,
    double *Layer7_Neurons_GPU,
    double *Layer6_Mean_GPU,
    double *Layer6_StanDev_GPU,
    double *Layer6_Gamma_GPU,
    double *Layer6_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y + 32);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 58)
                       + (threadIdx.y)
                       + (32);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58)] * Layer6_Weights_GPU[weight_Position + (row * 3)])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 1] * Layer6_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 2] * Layer6_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer6_Mean_GPU[filter_number]) / Layer6_StanDev_GPU[filter_number];
    Z = (Z * Layer6_Gamma_GPU[filter_number]) + Layer6_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer7_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSixthLayer_DSC_partC(double *Layer6_Neurons_GPU,
    double *Layer6_Weights_GPU,
    double *Layer7_Neurons_GPU,
    double *Layer6_Mean_GPU,
    double *Layer6_StanDev_GPU,
    double *Layer6_Gamma_GPU,
    double *Layer6_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32)
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (58 * 32)
                       + (threadIdx.x * 58)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58)] * Layer6_Weights_GPU[weight_Position + (row * 3)])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 1] * Layer6_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 2] * Layer6_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer6_Mean_GPU[filter_number]) / Layer6_StanDev_GPU[filter_number];
    Z = (Z * Layer6_Gamma_GPU[filter_number]) + Layer6_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer7_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSixthLayer_DSC_partD(double *Layer6_Neurons_GPU,
    double *Layer6_Weights_GPU,
    double *Layer7_Neurons_GPU,
    double *Layer6_Mean_GPU,
    double *Layer6_StanDev_GPU,
    double *Layer6_Gamma_GPU,
    double *Layer6_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32)
                        + 32
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (58 * 32)
                       + (32)
                       + (threadIdx.x * 58)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58)] * Layer6_Weights_GPU[weight_Position + (row * 3)])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 1] * Layer6_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer6_Neurons_GPU[(filter_number * 58 * 58) + input_Position + (row * 58) + 2] * Layer6_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer6_Mean_GPU[filter_number]) / Layer6_StanDev_GPU[filter_number];
    Z = (Z * Layer6_Gamma_GPU[filter_number]) + Layer6_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer7_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** SIXTH LAYER END **************************************************** */

/*  *************************************************** SEVENTH LAYER START ************************************************ */
/*
    Layer 7: Pointwise Separable Convolution Layer
    Input: 56 * 56 * 128
    Weight: 1 * 1 * 128 * 128 with a Stride of 1
    Output: 57 * 57 * 128  (Padding for the next layer is handled in this layer itself)
*/
__global__ void executeSeventhLayer_PSC_partA(double *Layer7_Neurons_GPU,
    double *Layer7_Weights_GPU,
    double *Layer8_Neurons_GPU,
    double *Layer7_Mean_GPU,
    double *Layer7_StanDev_GPU,
    double *Layer7_Gamma_GPU,
    double *Layer7_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 57 * 57)   // channel to work with
                        + (threadIdx.x * 57)
                        + (threadIdx.y);

    int weight_Position = filter_number * 128;

    int input_Position = (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 128; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer7_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer7_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer7_Mean_GPU[filter_number]) / Layer7_StanDev_GPU[filter_number];
    Z = (Z * Layer7_Gamma_GPU[filter_number]) + Layer7_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer8_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSeventhLayer_PSC_partB(double *Layer7_Neurons_GPU,
    double *Layer7_Weights_GPU,
    double *Layer8_Neurons_GPU,
    double *Layer7_Mean_GPU,
    double *Layer7_StanDev_GPU,
    double *Layer7_Gamma_GPU,
    double *Layer7_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 57 * 57)   // channel to work with
                        + (threadIdx.x * 57)
                        + (threadIdx.y + 32);

    int weight_Position = filter_number * 128;

    int input_Position = (threadIdx.x * 56)
                       + (threadIdx.y)
                       + (32);

    for(int channel = 0; channel < 128 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer7_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer7_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer7_Mean_GPU[filter_number]) / Layer7_StanDev_GPU[filter_number];
    Z = (Z * Layer7_Gamma_GPU[filter_number]) + Layer7_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer8_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSeventhLayer_PSC_partC(double *Layer7_Neurons_GPU,
    double *Layer7_Weights_GPU,
    double *Layer8_Neurons_GPU,
    double *Layer7_Mean_GPU,
    double *Layer7_StanDev_GPU,
    double *Layer7_Gamma_GPU,
    double *Layer7_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 57 * 57)   // channel to work with
                        + (57 * 32)
                        + (threadIdx.x * 57)
                        + (threadIdx.y);

    int weight_Position = filter_number * 128;

    int input_Position = (56 * 32)
                       + (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 128 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer7_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer7_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer7_Mean_GPU[filter_number]) / Layer7_StanDev_GPU[filter_number];
    Z = (Z * Layer7_Gamma_GPU[filter_number]) + Layer7_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer8_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSeventhLayer_PSC_partD(double *Layer7_Neurons_GPU,
    double *Layer7_Weights_GPU,
    double *Layer8_Neurons_GPU,
    double *Layer7_Mean_GPU,
    double *Layer7_StanDev_GPU,
    double *Layer7_Gamma_GPU,
    double *Layer7_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 57 * 57)   // channel to work with
                        + (57 * 32)
                        + 32
                        + (threadIdx.x * 57)
                        + (threadIdx.y);

    int weight_Position = filter_number * 128;

    int input_Position = (56 * 32)
                       + (32)
                       + (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 128 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer7_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer7_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer7_Mean_GPU[filter_number]) / Layer7_StanDev_GPU[filter_number];
    Z = (Z * Layer7_Gamma_GPU[filter_number]) + Layer7_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer8_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** SEVENTH LAYER END **************************************************** */

/*  *************************************************** EIGHTH LAYER START ************************************************** */
/*
    Layer 8: Depthwise Separable Convolution Layer
    Input: 57 * 57 * 128
    Weight: 3 * 3 * 128  with a Stride of 2
    Output: 28 * 28 * 128
*/
__global__ void executeEighthLayer_DSC(double *Layer8_Neurons_GPU,
    double *Layer8_Weights_GPU,
    double *Layer9_Neurons_GPU,
    double *Layer8_Mean_GPU,
    double *Layer8_StanDev_GPU,
    double *Layer8_Gamma_GPU,
    double *Layer8_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int stride = 2;

    // Output position
    int output_Position = (filter_number * 28 * 28)   // channel to work with
                        + (threadIdx.x * 28)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 57 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer8_Neurons_GPU[(filter_number * 57 * 57) + input_Position + (row * 57)] * Layer8_Weights_GPU[weight_Position + (row * 3)])
                + (Layer8_Neurons_GPU[(filter_number * 57 * 57) + input_Position + (row * 57) + 1] * Layer8_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer8_Neurons_GPU[(filter_number * 57 * 57) + input_Position + (row * 57) + 2] * Layer8_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer8_Mean_GPU[filter_number]) / Layer8_StanDev_GPU[filter_number];
    Z = (Z * Layer8_Gamma_GPU[filter_number]) + Layer8_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer9_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** EIGHTH LAYER END **************************************************** */

/*  *************************************************** NINTH LAYER START ************************************************** */
/*
    Layer 9: Pointwise Separable Convolution Layer
    Input: 28 * 28 * 128
    Weight: 1 * 1 * 128 * 256  with a Stride of 1
    Output: 30 * 30 * 256 (Handling the padding for the next layer)
*/
__global__ void executeNinthLayer_PSC(double *Layer9_Neurons_GPU,
    double *Layer9_Weights_GPU,
    double *Layer10_Neurons_GPU,
    double *Layer9_Mean_GPU,
    double *Layer9_StanDev_GPU,
    double *Layer9_Gamma_GPU,
    double *Layer9_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 31;

    // Output position
    int output_Position = (filter_number * 30 * 30)   // channel to work with
                        + (threadIdx.x * 30)
                        + (threadIdx.y);

    int weight_Position = filter_number * 128;

    int input_Position = (threadIdx.x * 28)
                        + (threadIdx.y);

    for(int channel = 0; channel < 128; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer9_Neurons_GPU[(channel * 28 * 28) + input_Position] * Layer9_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer9_Mean_GPU[filter_number]) / Layer9_StanDev_GPU[filter_number];
    Z = (Z * Layer9_Gamma_GPU[filter_number]) + Layer9_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer10_Neurons_GPU[output_Position + offset] = Z;
}

/*  *************************************************** NINTH LAYER END **************************************************** */

/*  *************************************************** TENTH LAYER START ************************************************** */
/*
    Layer 10: Depthwise Separable Convolution Layer
    Input: 30 * 30 * 256
    Weight: 3 * 3 * 256  with a Stride of 1
    Output: 28 * 28 * 256
*/
__global__ void executeTenthLayer_DSC(double *Layer10_Neurons_GPU,
    double *Layer10_Weights_GPU,
    double *Layer11_Neurons_GPU,
    double *Layer10_Mean_GPU,
    double *Layer10_StanDev_GPU,
    double *Layer10_Gamma_GPU,
    double *Layer10_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 28 * 28)   // channel to work with
                        + (threadIdx.x * 28)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 30)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer10_Neurons_GPU[(filter_number * 30 * 30) + input_Position + (row * 30)] * Layer10_Weights_GPU[weight_Position + (row * 3)])
                + (Layer10_Neurons_GPU[(filter_number * 30 * 30) + input_Position + (row * 30) + 1] * Layer10_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer10_Neurons_GPU[(filter_number * 30 * 30) + input_Position + (row * 30) + 2] * Layer10_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer10_Mean_GPU[filter_number]) / Layer10_StanDev_GPU[filter_number];
    Z = (Z * Layer10_Gamma_GPU[filter_number]) + Layer10_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer11_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** TENTH LAYER END **************************************************** */

/*  *************************************************** ELEVENTH LAYER START ************************************************** */
/*
    Layer 11: Pointwise Separable Convolution Layer
    Input: 28 * 28 * 256
    Weight: 1 * 1 * 256 * 256  with a Stride of 1
    Output: 29 * 29 * 256  (Padding for the next layer is handled in this layer itself)
*/
__global__ void executeEleventhLayer_PSC(double *Layer11_Neurons_GPU,
    double *Layer11_Weights_GPU,
    double *Layer12_Neurons_GPU,
    double *Layer11_Mean_GPU,
    double *Layer11_StanDev_GPU,
    double *Layer11_Gamma_GPU,
    double *Layer11_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 29 * 29)   // channel to work with
                        + (threadIdx.x * 29)
                        + (threadIdx.y);

    int weight_Position = filter_number * 256;

    int input_Position = (threadIdx.x * 28)
                        + (threadIdx.y);

    for(int channel = 0; channel < 256; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer11_Neurons_GPU[(channel * 28 * 28) + input_Position] * Layer11_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer11_Mean_GPU[filter_number]) / Layer11_StanDev_GPU[filter_number];
    Z = (Z * Layer11_Gamma_GPU[filter_number]) + Layer11_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer12_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** ELEVENTH LAYER END **************************************************** */

/*  *************************************************** TWELFTH LAYER START ************************************************** */
/*
    Layer 12: Depthwise Separable Convolution Layer
    Input: 29 * 29 * 256
    Weight: 3 * 3 * 256 with a Stride of 2
    Output: 14 * 14 * 256
*/
__global__ void executeTwelfthLayer_DSC(double *Layer12_Neurons_GPU,
    double *Layer12_Weights_GPU,
    double *Layer13_Neurons_GPU,
    double *Layer12_Mean_GPU,
    double *Layer12_StanDev_GPU,
    double *Layer12_Gamma_GPU,
    double *Layer12_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int stride = 2;

    // Output position
    int output_Position = (filter_number * 14 * 14)   // channel to work with
                        + (threadIdx.x * 14)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 29 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer12_Neurons_GPU[(filter_number * 29 * 29) + input_Position + (row * 29)] * Layer12_Weights_GPU[weight_Position + (row * 3)])
                + (Layer12_Neurons_GPU[(filter_number * 29 * 29) + input_Position + (row * 29) + 1] * Layer12_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer12_Neurons_GPU[(filter_number * 29 * 29) + input_Position + (row * 29) + 2] * Layer12_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer12_Mean_GPU[filter_number]) / Layer12_StanDev_GPU[filter_number];
    Z = (Z * Layer12_Gamma_GPU[filter_number]) + Layer12_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer13_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** TWELFTH LAYER END **************************************************** */

/*  *************************************************** THIRTEENTH LAYER START ************************************************** */
/*
    Layer 13: Pointwise Separable Convolution Layer
    Input: 14 * 14 * 256
    Weight: 1 * 1 * 256 * 512 with a Stride of 1
    Output: 16 * 16 * 512  (Handling padding for next layer)
*/
__global__ void executeThirteenthLayer_PSC(double *Layer13_Neurons_GPU,
    double *Layer13_Weights_GPU,
    double *Layer14_Neurons_GPU,
    double *Layer13_Mean_GPU,
    double *Layer13_StanDev_GPU,
    double *Layer13_Gamma_GPU,
    double *Layer13_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 17;

    // Output position
    int output_Position = (filter_number * 16 * 16)   // channel to work with
                        + (threadIdx.x * 16)
                        + (threadIdx.y);

    int weight_Position = filter_number * 256;

    int input_Position = (threadIdx.x * 14)
                        + (threadIdx.y);

    for(int channel = 0; channel < 256; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer13_Neurons_GPU[(channel * 14 * 14) + input_Position] * Layer13_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer13_Mean_GPU[filter_number]) / Layer13_StanDev_GPU[filter_number];
    Z = (Z * Layer13_Gamma_GPU[filter_number]) + Layer13_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer14_Neurons_GPU[output_Position + offset] = Z;
}

/*  *************************************************** THIRTEENTH LAYER END **************************************************** */

/*  *************************************************** FOURTEENTH LAYER START ************************************************** */
/*
    Layer 14: Depthwise Separable Convolution Layer
    Input: 16 * 16 * 512
    Weight: 3 * 3 * 512 with a Stride of 1
    Output: 14 * 14 * 512  (Handling padding for next layer)
*/
__global__ void executeFourteenthLayer_DSC(double *Layer14_Neurons_GPU,
    double *Layer14_Weights_GPU,
    double *Layer15_Neurons_GPU,
    double *Layer14_Mean_GPU,
    double *Layer14_StanDev_GPU,
    double *Layer14_Gamma_GPU,
    double *Layer14_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 14 * 14)   // channel to work with
                        + (threadIdx.x * 14)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 16)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer14_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16)] * Layer14_Weights_GPU[weight_Position + (row * 3)])
                + (Layer14_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 1] * Layer14_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer14_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 2] * Layer14_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer14_Mean_GPU[filter_number]) / Layer14_StanDev_GPU[filter_number];
    Z = (Z * Layer14_Gamma_GPU[filter_number]) + Layer14_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer15_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** FOURTEENTH LAYER END **************************************************** */

/*  *************************************************** FIFTEENTH LAYER START ************************************************** */
/*
    Layer 15: Pointwise Separable Convolution Layer
    Input: 14 * 14 * 512
    Weight: 1 * 1 * 512 * 512 with a Stride of 1
    Output: 16 * 16 * 512  (Handling padding for next layer)
*/
__global__ void executeFifteenthLayer_PSC(double *Layer15_Neurons_GPU,
    double *Layer15_Weights_GPU,
    double *Layer16_Neurons_GPU,
    double *Layer15_Mean_GPU,
    double *Layer15_StanDev_GPU,
    double *Layer15_Gamma_GPU,
    double *Layer15_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 17;

    // Output position
    int output_Position = (filter_number * 16 * 16)   // channel to work with
                        + (threadIdx.x * 16)
                        + (threadIdx.y);

    int weight_Position = filter_number * 512;

    int input_Position = (threadIdx.x * 14)
                        + (threadIdx.y);

    for(int channel = 0; channel < 512 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer15_Neurons_GPU[(channel * 14 * 14) + input_Position] * Layer15_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer15_Mean_GPU[filter_number]) / Layer15_StanDev_GPU[filter_number];
    Z = (Z * Layer15_Gamma_GPU[filter_number]) + Layer15_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer16_Neurons_GPU[output_Position + offset] = Z;
}

/*  *************************************************** FIFTEENTH LAYER END **************************************************** */

/*  *************************************************** SIXTEENTH LAYER START ************************************************** */
/*
    Layer 16: Depthwise Separable Convolution Layer
    Input: 16 * 16 * 512
    Weight: 3 * 3 * 512 with a Stride of 1
    Output: 14 * 14 * 512  (Handling padding for next layer)
*/
__global__ void executeSixteenthLayer_DSC(double *Layer16_Neurons_GPU,
    double *Layer16_Weights_GPU,
    double *Layer17_Neurons_GPU,
    double *Layer16_Mean_GPU,
    double *Layer16_StanDev_GPU,
    double *Layer16_Gamma_GPU,
    double *Layer16_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 14 * 14)   // channel to work with
                        + (threadIdx.x * 14)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 16)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer16_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16)] * Layer16_Weights_GPU[weight_Position + (row * 3)])
                + (Layer16_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 1] * Layer16_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer16_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 2] * Layer16_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer16_Mean_GPU[filter_number]) / Layer16_StanDev_GPU[filter_number];
    Z = (Z * Layer16_Gamma_GPU[filter_number]) + Layer16_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer17_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** SIXTEENTH LAYER END **************************************************** */

/*  *************************************************** SEVENTEENTH LAYER START ************************************************** */
/*
    Layer 17: Pointwise Separable Convolution Layer
    Input: 14 * 14 * 512
    Weight: 1 * 1 * 512 * 512 with a Stride of 1
    Output: 16 * 16 * 512  (Handling padding for next layer)
*/
__global__ void executeSeventeenthLayer_PSC(double *Layer17_Neurons_GPU,
    double *Layer17_Weights_GPU,
    double *Layer18_Neurons_GPU,
    double *Layer17_Mean_GPU,
    double *Layer17_StanDev_GPU,
    double *Layer17_Gamma_GPU,
    double *Layer17_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 17;

    // Output position
    int output_Position = (filter_number * 16 * 16)   // channel to work with
                        + (threadIdx.x * 16)
                        + (threadIdx.y);

    int weight_Position = filter_number * 512;

    int input_Position = (threadIdx.x * 14)
                        + (threadIdx.y);

    for(int channel = 0; channel < 512 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer17_Neurons_GPU[(channel * 14 * 14) + input_Position] * Layer17_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer17_Mean_GPU[filter_number]) / Layer17_StanDev_GPU[filter_number];
    Z = (Z * Layer17_Gamma_GPU[filter_number]) + Layer17_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer18_Neurons_GPU[output_Position + offset] = Z;
}

/*  *************************************************** SEVENTEENTH LAYER END **************************************************** */

/*  *************************************************** EIGHTEENTH LAYER START ************************************************** */
/*
    Layer 18: Depthwise Separable Convolution Layer
    Input: 16 * 16 * 512
    Weight: 3 * 3 * 512 with a Stride of 1
    Output: 14 * 14 * 512
*/
__global__ void executeEighteenthLayer_DSC(double *Layer18_Neurons_GPU,
    double *Layer18_Weights_GPU,
    double *Layer19_Neurons_GPU,
    double *Layer18_Mean_GPU,
    double *Layer18_StanDev_GPU,
    double *Layer18_Gamma_GPU,
    double *Layer18_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 14 * 14)   // channel to work with
                        + (threadIdx.x * 14)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 16)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer18_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16)] * Layer18_Weights_GPU[weight_Position + (row * 3)])
                + (Layer18_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 1] * Layer18_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer18_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 2] * Layer18_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer18_Mean_GPU[filter_number]) / Layer18_StanDev_GPU[filter_number];
    Z = (Z * Layer18_Gamma_GPU[filter_number]) + Layer18_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer19_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** EIGHTEENTH LAYER END **************************************************** */

/*  *************************************************** NINETEENTH LAYER START ************************************************** */
/*
    Layer 19: Pointwise Separable Convolution Layer
    Input: 14 * 14 * 512
    Weight: 1 * 1 * 512 * 512 with a Stride of 1
    Output: 16 * 16 * 512  (Handling padding for next layer)
*/
__global__ void executeNineteenthLayer_PSC(double *Layer19_Neurons_GPU,
    double *Layer19_Weights_GPU,
    double *Layer20_Neurons_GPU,
    double *Layer19_Mean_GPU,
    double *Layer19_StanDev_GPU,
    double *Layer19_Gamma_GPU,
    double *Layer19_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 17;

    // Output position
    int output_Position = (filter_number * 16 * 16)   // channel to work with
                        + (threadIdx.x * 16)
                        + (threadIdx.y);

    int weight_Position = filter_number * 512;

    int input_Position = (threadIdx.x * 14)
                        + (threadIdx.y);

    for(int channel = 0; channel < 512 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer19_Neurons_GPU[(channel * 14 * 14) + input_Position] * Layer19_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer19_Mean_GPU[filter_number]) / Layer19_StanDev_GPU[filter_number];
    Z = (Z * Layer19_Gamma_GPU[filter_number]) + Layer19_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer20_Neurons_GPU[output_Position + offset] = Z;
}

/*  *************************************************** NINETEENTH LAYER END **************************************************** */

/*  *************************************************** TWENTY LAYER START ************************************************** */
/*
    Layer 20: Depthwise Separable Convolution Layer
    Input: 16 * 16 * 512
    Weight: 3 * 3 * 512 with a Stride of 1
    Output: 14 * 14 * 512
*/
__global__ void executeTwentyLayer_DSC(double *Layer20_Neurons_GPU,
    double *Layer20_Weights_GPU,
    double *Layer21_Neurons_GPU,
    double *Layer20_Mean_GPU,
    double *Layer20_StanDev_GPU,
    double *Layer20_Gamma_GPU,
    double *Layer20_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 14 * 14)   // channel to work with
                        + (threadIdx.x * 14)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 16)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer20_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16)] * Layer20_Weights_GPU[weight_Position + (row * 3)])
                + (Layer20_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 1] * Layer20_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer20_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 2] * Layer20_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer20_Mean_GPU[filter_number]) / Layer20_StanDev_GPU[filter_number];
    Z = (Z * Layer20_Gamma_GPU[filter_number]) + Layer20_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer21_Neurons_GPU[output_Position] = Z;
}
/*  *************************************************** TWENTY LAYER END **************************************************** */

/*  *************************************************** TWENTYONE LAYER START ************************************************** */
/*
    Layer 21: Pointwise Separable Convolution Layer
    Input: 14 * 14 * 512
    Weight: 1 * 1 * 512 * 512 with a Stride of 1
    Output: 16 * 16 * 512  (Handling padding for next layer)
*/
__global__ void executeTwentyOneLayer_PSC(double *Layer21_Neurons_GPU,
    double *Layer21_Weights_GPU,
    double *Layer22_Neurons_GPU,
    double *Layer21_Mean_GPU,
    double *Layer21_StanDev_GPU,
    double *Layer21_Gamma_GPU,
    double *Layer21_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 17;

    // Output position
    int output_Position = (filter_number * 16 * 16)   // channel to work with
                        + (threadIdx.x * 16)
                        + (threadIdx.y);

    int weight_Position = filter_number * 512;

    int input_Position = (threadIdx.x * 14)
                        + (threadIdx.y);

    for(int channel = 0; channel < 512 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer21_Neurons_GPU[(channel * 14 * 14) + input_Position] * Layer21_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer21_Mean_GPU[filter_number]) / Layer21_StanDev_GPU[filter_number];
    Z = (Z * Layer21_Gamma_GPU[filter_number]) + Layer21_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer22_Neurons_GPU[output_Position + offset] = Z;
}

/*  *************************************************** TWENTYONE LAYER END **************************************************** */

/*  *************************************************** TWENTYTWO LAYER START ************************************************** */
/*
    Layer 22: Depthwise Separable Convolution Layer
    Input: 16 * 16 * 512
    Weight: 3 * 3 * 512 with a Stride of 1
    Output: 14 * 14 * 512
*/
__global__ void executeTwentyTwoLayer_DSC(double *Layer22_Neurons_GPU,
    double *Layer22_Weights_GPU,
    double *Layer23_Neurons_GPU,
    double *Layer22_Mean_GPU,
    double *Layer22_StanDev_GPU,
    double *Layer22_Gamma_GPU,
    double *Layer22_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 14 * 14)   // channel to work with
                        + (threadIdx.x * 14)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 16)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer22_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16)] * Layer22_Weights_GPU[weight_Position + (row * 3)])
                + (Layer22_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 1] * Layer22_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer22_Neurons_GPU[(filter_number * 16 * 16) + input_Position + (row * 16) + 2] * Layer22_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer22_Mean_GPU[filter_number]) / Layer22_StanDev_GPU[filter_number];
    Z = (Z * Layer22_Gamma_GPU[filter_number]) + Layer22_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer23_Neurons_GPU[output_Position] = Z;
}
/*  *************************************************** TWENTYTWO LAYER END **************************************************** */

/*  *************************************************** TWENTYTHREE LAYER START ************************************************** */
/*
    Layer 23: Pointwise Separable Convolution Layer
    Input: 14 * 14 * 512
    Weight: 1 * 1 * 512 * 512 with a Stride of 1
    Output: 14 * 14 * 512  (Handling padding for next layer)
*/
__global__ void executeTwentyThreeLayer_PSC(double *Layer23_Neurons_GPU,
    double *Layer23_Weights_GPU,
    double *Layer24_Neurons_GPU,
    double *Layer23_Mean_GPU,
    double *Layer23_StanDev_GPU,
    double *Layer23_Gamma_GPU,
    double *Layer23_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 15 * 15)   // channel to work with
                        + (threadIdx.x * 15)
                        + (threadIdx.y);

    int weight_Position = filter_number * 512;

    int input_Position = (threadIdx.x * 14)
                        + (threadIdx.y);

    for(int channel = 0; channel < 512 ; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer23_Neurons_GPU[(channel * 14 * 14) + input_Position] * Layer23_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer23_Mean_GPU[filter_number]) / Layer23_StanDev_GPU[filter_number];
    Z = (Z * Layer23_Gamma_GPU[filter_number]) + Layer23_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer24_Neurons_GPU[output_Position] = Z;
}
/*  *************************************************** TWENTYTHREE LAYER END **************************************************** */

/*  *************************************************** TWENTYFOUR LAYER START ************************************************** */
/*
    Layer 24: Depthwise Separable Convolution Layer
    Input: 15 * 15 * 512
    Weight: 3 * 3 * 512 with a Stride of 2
    Output: 14 * 14 * 512  (Handling padding for next layer)
*/
__global__ void executeTwentyFourLayer_DSC(double *Layer24_Neurons_GPU,
    double *Layer24_Weights_GPU,
    double *Layer25_Neurons_GPU,
    double *Layer24_Mean_GPU,
    double *Layer24_StanDev_GPU,
    double *Layer24_Gamma_GPU,
    double *Layer24_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int stride = 2;

    // Output position
    int output_Position = (filter_number * 7 * 7)   // channel to work with
                        + (threadIdx.x * 7)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 15 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer24_Neurons_GPU[(filter_number * 15 * 15) + input_Position + (row * 15)] * Layer24_Weights_GPU[weight_Position + (row * 3)])
                + (Layer24_Neurons_GPU[(filter_number * 15 * 15) + input_Position + (row * 15) + 1] * Layer24_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer24_Neurons_GPU[(filter_number * 15 * 15) + input_Position + (row * 15) + 2] * Layer24_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer24_Mean_GPU[filter_number]) / Layer24_StanDev_GPU[filter_number];
    Z = (Z * Layer24_Gamma_GPU[filter_number]) + Layer24_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer25_Neurons_GPU[output_Position] = Z;
}
/*  *************************************************** TWENTYFOUR LAYER END **************************************************** */

/*  *************************************************** TWENTYFIVE LAYER START ************************************************** */
/*
    Layer 25: Pointwise Separable Convolution Layer
    Input: 7 * 7 * 512
    Weight: 1 * 1 * 512 * 1024 with a Stride of 1
    Output: 9 * 9 * 1024 (Handling padding for next layer)
*/
__global__ void executeTwentyFiveLayer_PSC(double *Layer25_Neurons_GPU,
    double *Layer25_Weights_GPU,
    double *Layer26_Neurons_GPU,
    double *Layer25_Mean_GPU,
    double *Layer25_StanDev_GPU,
    double *Layer25_Gamma_GPU,
    double *Layer25_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;
    int offset = 10;

    // Output position
    int output_Position = (filter_number * 9 * 9)   // channel to work with
                        + (threadIdx.x * 9)
                        + (threadIdx.y);

    int weight_Position = filter_number * 512;

    int input_Position = (threadIdx.x * 7)
                        + (threadIdx.y);

    for(int channel = 0; channel < 512 ; channel++)
    {
        product += (Layer25_Neurons_GPU[(channel * 7 * 7) + input_Position] * Layer25_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer25_Mean_GPU[filter_number]) / Layer25_StanDev_GPU[filter_number];
    Z = (Z * Layer25_Gamma_GPU[filter_number]) + Layer25_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer26_Neurons_GPU[output_Position + offset] = Z;
}
/*  *************************************************** TWENTYFIVE LAYER END **************************************************** */

/*  *************************************************** TWENTYSIX LAYER START ************************************************** */
/*
    Layer 26: Depthwise Separable Convolution Layer
    Input: 9 * 9 * 1024
    Weight: 3 * 3 * 1024 with a Stride of 1
    Output: 7 * 7 * 1024  (Handling padding for next layer)
*/
__global__ void executeTwentySixLayer_DSC(double *Layer26_Neurons_GPU,
    double *Layer26_Weights_GPU,
    double *Layer27_Neurons_GPU,
    double *Layer26_Mean_GPU,
    double *Layer26_StanDev_GPU,
    double *Layer26_Gamma_GPU,
    double *Layer26_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 7 * 7)   // channel to work with
                        + (threadIdx.x * 7)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 9)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer26_Neurons_GPU[(filter_number * 9 * 9) + input_Position + (row * 9)] * Layer26_Weights_GPU[weight_Position + (row * 3)])
                + (Layer26_Neurons_GPU[(filter_number * 9 * 9) + input_Position + (row * 9) + 1] * Layer26_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer26_Neurons_GPU[(filter_number * 9 * 9) + input_Position + (row * 9) + 2] * Layer26_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer26_Mean_GPU[filter_number]) / Layer26_StanDev_GPU[filter_number];
    Z = (Z * Layer26_Gamma_GPU[filter_number]) + Layer26_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer27_Neurons_GPU[output_Position] = Z;
}
/*  *************************************************** TWENTYSIX LAYER END **************************************************** */

/*  *************************************************** TWENTYSEVEN LAYER START ************************************************** */
/*
    Layer 27: Pointwise Separable Convolution Layer
    Input: 7 * 7 * 1024
    Weight: 1 * 1 * 1024 * 1024 with a Stride of 1
    Output: 7 * 7 * 1024
*/
__global__ void executeTwentySevenLayer_PSC(double *Layer27_Neurons_GPU,
    double *Layer27_Weights_GPU,
    double *Layer28_Neurons_GPU,
    double *Layer27_Mean_GPU,
    double *Layer27_StanDev_GPU,
    double *Layer27_Gamma_GPU,
    double *Layer27_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 7 * 7)   // channel to work with
                        + (threadIdx.x * 7)
                        + (threadIdx.y);

    int weight_Position = filter_number * 1024;

    int input_Position = (threadIdx.x * 7)
                        + (threadIdx.y);

    for(int channel = 0; channel < 1024 ; channel++)
    {
        product += (Layer27_Neurons_GPU[(channel * 7 * 7) + input_Position] * Layer27_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer27_Mean_GPU[filter_number]) / Layer27_StanDev_GPU[filter_number];
    Z = (Z * Layer27_Gamma_GPU[filter_number]) + Layer27_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0;

    Layer28_Neurons_GPU[output_Position] = Z;
}
/*  *************************************************** TWENTYSEVEN LAYER END **************************************************** */

/*  *************************************************** TWENTYEIGHT LAYER START ************************************************** */
/*
    Layer 28: Global Average Pooling Layer
    Input: 7 * 7 * 1024
    Weight: None
    Output: 1 * 1 * 1024
*/
__global__ void executeTwentyEightLayer_AvgPooling(double *Layer28_Neurons_GPU,
    double *Layer29_Neurons_GPU
)
{
    double sum = 0.0;
    int filter_number = threadIdx.x * 32 + threadIdx.y;

    // Output position
    int output_Position = filter_number;

    int input_Position_start = filter_number * 49;
    for(int row = 0 ; row < 7 ; row++)
        for(int col = 0 ; col < 7 ; col++)
            sum += Layer28_Neurons_GPU[input_Position_start + row * 7 + col];

    sum = sum / 49;
    Layer29_Neurons_GPU[output_Position] = sum;
}
/*  *************************************************** TWENTYEIGHT LAYER END **************************************************** */

/*  *************************************************** TWENTYNINE LAYER START ************************************************** */
/*
    Layer 29: Fully Connected Layer
    Input: 1 * 1 * 1024
    Weight: 1000 * 1024
    Bias: 1000
    Output: 1000
*/
__global__ void executeTwentyNineLayer_FullyConnected(double *Layer29_Neurons_GPU,
    double *Layer30_Neurons_GPU,
    double *Layer29_Weights_GPU,
    double *Layer29_Bias_GPU
)
{
    double product = 0.0;
    int filter_number = threadIdx.x;

    // Output position
    int output_Position = filter_number;

    int weight_Position = filter_number * 1024;

    for(int channel = 0; channel < 1024 ; channel++)
    {
        product += (Layer29_Neurons_GPU[channel] * Layer29_Weights_GPU[weight_Position + channel]);
    }

    //Adding Bias to the output
    product += Layer29_Bias_GPU[filter_number];

    Layer30_Neurons_GPU[output_Position] = product;
}
/*  *************************************************** TWENTYNINE LAYER END **************************************************** */