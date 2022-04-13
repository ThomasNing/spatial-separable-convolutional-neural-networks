#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#include "MobileNets_kernel.cu"

#define INPUT_LAYER_SIZE 225 * 225 * 3
#define FIRST_LAYER_WEIGHT_SIZE 32 * 3 * 3 * 3
#define FIRST_LAYER_OUTPUT_SIZE 114 * 114 * 32
#define FIRST_LAYER_CHANNELS 32

#define SECOND_LAYER_WEIGHT_SIZE 32 * 3 * 3
#define SECOND_LAYER_OUTPUT_SIZE 112 * 112 * 32
#define SECOND_LAYER_CHANNELS 32

#define THIRD_LAYER_WEIGHT_SIZE 64 * 32
#define THIRD_LAYER_OUTPUT_SIZE 113 * 113 * 64
#define THIRD_LAYER_CHANNELS 64

#define FOURTH_LAYER_WEIGHT_SIZE 3 * 3 * 64
#define FOURTH_LAYER_OUTPUT_SIZE 56 * 56 * 64
#define FOURTH_LAYER_CHANNELS 64

#define FIFTH_LAYER_WEIGHT_SIZE 64 * 128
#define FIFTH_LAYER_OUTPUT_SIZE 58 * 58 * 128
#define FIFTH_LAYER_CHANNELS 128

#define SIXTH_LAYER_WEIGHT_SIZE 3 * 3 * 128
#define SIXTH_LAYER_OUTPUT_SIZE 56 * 56 * 128
#define SIXTH_LAYER_CHANNELS 128

#define SEVENTH_LAYER_WEIGHT_SIZE 128 * 128
#define SEVENTH_LAYER_OUTPUT_SIZE 57 * 57 * 128
#define SEVENTH_LAYER_CHANNELS 128

#define EIGHTH_LAYER_WEIGHT_SIZE 3 * 3 * 128
#define EIGHTH_LAYER_OUTPUT_SIZE 28 * 28 * 128
#define EIGHTH_LAYER_CHANNELS 128

#define NINTH_LAYER_WEIGHT_SIZE 128 * 256
#define NINTH_LAYER_OUTPUT_SIZE 30 * 30 * 256
#define NINTH_LAYER_CHANNELS 256

#define TENTH_LAYER_WEIGHT_SIZE 9 * 256
#define TENTH_LAYER_OUTPUT_SIZE 28 * 28 * 256
#define TENTH_LAYER_CHANNELS 256

#define ELEVENTH_LAYER_WEIGHT_SIZE 256 * 256
#define ELEVENTH_LAYER_OUTPUT_SIZE 29 * 29 * 256
#define ELEVENTH_LAYER_CHANNELS 256

#define TWELFTH_LAYER_WEIGHT_SIZE 9 * 256
#define TWELFTH_LAYER_OUTPUT_SIZE 14 * 14 * 256
#define TWELFTH_LAYER_CHANNELS 256

#define THIRTEENTH_LAYER_WEIGHT_SIZE 512 * 256
#define THIRTEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define THIRTEENTH_LAYER_CHANNELS 512

#define FOURTEENTH_LAYER_WEIGHT_SIZE 512 * 9
#define FOURTEENTH_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define FOURTEENTH_LAYER_CHANNELS 512

#define FIFTEENTH_LAYER_WEIGHT_SIZE 512 * 512
#define FIFTEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define FIFTEENTH_LAYER_CHANNELS 512

#define SIXTEENTH_LAYER_WEIGHT_SIZE 512 * 9
#define SIXTEENTH_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define SIXTEENTH_LAYER_CHANNELS 512

#define SEVENTEENTH_LAYER_WEIGHT_SIZE 512 * 512
#define SEVENTEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define SEVENTEENTH_LAYER_CHANNELS 512

#define EIGHTEENTH_LAYER_WEIGHT_SIZE 512 * 9
#define EIGHTEENTH_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define EIGHTEENTH_LAYER_CHANNELS 512

#define NINETEENTH_LAYER_WEIGHT_SIZE 512 * 512
#define NINETEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define NINETEENTH_LAYER_CHANNELS 512

#define TWENTY_LAYER_WEIGHT_SIZE 512 * 9
#define TWENTY_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define TWENTY_LAYER_CHANNELS 512

#define TWENTYONE_LAYER_WEIGHT_SIZE 512 * 512
#define TWENTYONE_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define TWENTYONE_LAYER_CHANNELS 512

#define TWENTYTWO_LAYER_WEIGHT_SIZE 512 * 9
#define TWENTYTWO_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define TWENTYTWO_LAYER_CHANNELS 512

#define TWENTYTHREE_LAYER_WEIGHT_SIZE 512 * 512
#define TWENTYTHREE_LAYER_OUTPUT_SIZE 15 * 15 * 512
#define TWENTYTHREE_LAYER_CHANNELS 512

#define TWENTYFOUR_LAYER_WEIGHT_SIZE 9 * 512
#define TWENTYFOUR_LAYER_OUTPUT_SIZE 7 * 7 * 512
#define TWENTYFOUR_LAYER_CHANNELS 512

#define TWENTYFIVE_LAYER_WEIGHT_SIZE 1024 * 512
#define TWENTYFIVE_LAYER_OUTPUT_SIZE 9 * 9 * 1024
#define TWENTYFIVE_LAYER_CHANNELS 1024

#define TWENTYSIX_LAYER_WEIGHT_SIZE 1024 * 9
#define TWENTYSIX_LAYER_OUTPUT_SIZE 7 * 7 * 1024
#define TWENTYSIX_LAYER_CHANNELS 1024

#define TWENTYSEVEN_LAYER_WEIGHT_SIZE 1024 * 1024
#define TWENTYSEVEN_LAYER_OUTPUT_SIZE 7 * 7 * 1024
#define TWENTYSEVEN_LAYER_CHANNELS 1024

// Global Average Pooling Layer
#define TWENTYEIGHT_LAYER_OUTPUT_SIZE 1024

// Fully Connected Layer
#define TWENTYNINE_LAYER_OUTPUT_SIZE 1000
#define TWENTYNINE_LAYER_WEIGHT_SIZE 1024 * 1000

// Function declarations
void NeuralNetwork();
void read_File(const char *weightFileName, double *Layer1_Weights_CPU);
void read_Input_File(const char *inputFileName, double *Layer1_Neurons_CPU);

void Read_First_Layer_Data(double *Layer1_Neurons_CPU,
                           double *Layer1_Weights_CPU,
                           double *Layer1_Mean_CPU,
                           double *Layer1_StanDev_CPU,
                           double *Layer1_Gamma_CPU,
                           double *Layer1_Beta_CPU);

void Execute_First_Layer(double *Layer2_Neurons_GPU);

void Read_SecondLayer_Data(double *Layer1_Weights_CPU,
                           double *Layer2_Mean_CPU,
                           double *Layer2_StanDev_CPU,
                           double *Layer2_Gamma_CPU,
                           double *Layer2_Beta_CPU);

void Execute_Second_Layer(
    double *Layer2_Neurons_GPU,
    double *Layer3_Neurons_GPU);

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU,
                          double *Layer3_Mean_CPU,
                          double *Layer3_StanDev_CPU,
                          double *Layer3_Gamma_CPU,
                          double *Layer3_Beta_CPU);

void Execute_Third_Layer(
    double *Layer3_Neurons_GPU,
    double *Layer4_Neurons_GPU);

void Read_FourthLayer_Data(double *Layer4_Weights_CPU,
                           double *Layer4_Mean_CPU,
                           double *Layer4_StanDev_CPU,
                           double *Layer4_Gamma_CPU,
                           double *Layer4_Beta_CPU);

void Execute_Fourth_Layer(
    double *Layer4_Neurons_GPU,
    double *Layer5_Neurons_GPU);

void Read_FifthLayer_Data(double *Layer5_Weights_CPU,
                          double *Layer5_Mean_CPU,
                          double *Layer5_StanDev_CPU,
                          double *Layer5_Gamma_CPU,
                          double *Layer5_Beta_CPU);

void Execute_Fifth_Layer(
    double *Layer5_Neurons_GPU,
    double *Layer6_Neurons_GPU);

void Read_SixthLayer_Data(double *Layer6_Weights_CPU,
                          double *Layer6_Mean_CPU,
                          double *Layer6_StanDev_CPU,
                          double *Layer6_Gamma_CPU,
                          double *Layer6_Beta_CPU);

void Execute_Sixth_Layer(
    double *Layer6_Neurons_GPU,
    double *Layer7_Neurons_GPU);

void Read_SeventhLayer_Data(double *Layer7_Weights_CPU,
                            double *Layer7_Mean_CPU,
                            double *Layer7_StanDev_CPU,
                            double *Layer7_Gamma_CPU,
                            double *Layer7_Beta_CPU);

void Execute_Seventh_Layer(
    double *Layer7_Neurons_GPU,
    double *Layer8_Neurons_GPU);

void Read_EighthLayer_Data(double *Layer8_Weights_CPU,
                           double *Layer8_Mean_CPU,
                           double *Layer8_StanDev_CPU,
                           double *Layer8_Gamma_CPU,
                           double *Layer8_Beta_CPU);

void Execute_Eighth_Layer(
    double *Layer8_Neurons_GPU,
    double *Layer9_Neurons_GPU);

void Read_NinthLayer_Data(double *Layer9_Weights_CPU,
                          double *Layer9_Mean_CPU,
                          double *Layer9_StanDev_CPU,
                          double *Layer9_Gamma_CPU,
                          double *Layer9_Beta_CPU);

void Execute_Ninth_Layer(
    double *Layer9_Neurons_GPU,
    double *Layer10_Neurons_GPU);

void Read_TenthLayer_Data(double *Layer10_Weights_CPU,
                          double *Layer10_Mean_CPU,
                          double *Layer10_StanDev_CPU,
                          double *Layer10_Gamma_CPU,
                          double *Layer10_Beta_CPU);

void Execute_Tenth_Layer(
    double *Layer10_Neurons_GPU,
    double *Layer11_Neurons_GPU);

void Read_EleventhLayer_Data(double *Layer11_Weights_CPU,
                             double *Layer11_Mean_CPU,
                             double *Layer11_StanDev_CPU,
                             double *Layer11_Gamma_CPU,
                             double *Layer11_Beta_CPU);

void Execute_Eleventh_Layer(
    double *Layer11_Neurons_GPU,
    double *Layer12_Neurons_GPU);

void Read_TwelvethLayer_Data(double *Layer12_Weights_CPU,
                             double *Layer12_Mean_CPU,
                             double *Layer12_StanDev_CPU,
                             double *Layer12_Gamma_CPU,
                             double *Layer12_Beta_CPU);

void Execute_Twelveth_Layer(
    double *Layer12_Neurons_GPU,
    double *Layer13_Neurons_GPU);

void Read_ThirteenthLayer_Data(double *Layer13_Weights_CPU,
                               double *Layer13_Mean_CPU,
                               double *Layer13_StanDev_CPU,
                               double *Layer13_Gamma_CPU,
                               double *Layer13_Beta_CPU);

void Execute_Thirteenth_Layer(
    double *Layer13_Neurons_GPU,
    double *Layer14_Neurons_GPU);

void Read_FourteenthLayer_Data(double *Layer14_Weights_CPU,
                               double *Layer14_Mean_CPU,
                               double *Layer14_StanDev_CPU,
                               double *Layer14_Gamma_CPU,
                               double *Layer14_Beta_CPU);

void Execute_Fourteenth_Layer(
    double *Layer14_Neurons_GPU,
    double *Layer15_Neurons_GPU);

void Read_FifteenthLayer_Data(double *Layer15_Weights_CPU,
                              double *Layer15_Mean_CPU,
                              double *Layer15_StanDev_CPU,
                              double *Layer15_Gamma_CPU,
                              double *Layer15_Beta_CPU);

void Execute_Fifteenth_Layer(
    double *Layer15_Neurons_GPU,
    double *Layer16_Neurons_GPU);

void Read_SixteenthLayer_Data(double *Layer16_Weights_CPU,
                              double *Layer16_Mean_CPU,
                              double *Layer16_StanDev_CPU,
                              double *Layer16_Gamma_CPU,
                              double *Layer16_Beta_CPU);

void Execute_Sixteenth_Layer(
    double *Layer16_Neurons_GPU,
    double *Layer17_Neurons_GPU);

void Read_SeventeenthLayer_Data(double *Layer17_Weights_CPU,
                                double *Layer17_Mean_CPU,
                                double *Layer17_StanDev_CPU,
                                double *Layer17_Gamma_CPU,
                                double *Layer17_Beta_CPU);

void Execute_Seventeenth_Layer(
    double *Layer17_Neurons_GPU,
    double *Layer18_Neurons_GPU);

void Read_EighteenthLayer_Data(double *Layer18_Weights_CPU,
                               double *Layer18_Mean_CPU,
                               double *Layer18_StanDev_CPU,
                               double *Layer18_Gamma_CPU,
                               double *Layer18_Beta_CPU);

void Execute_Eighteenth_Layer(
    double *Layer18_Neurons_GPU,
    double *Layer19_Neurons_GPU);

void Read_NineteenthLayer_Data(double *Layer19_Weights_CPU,
                               double *Layer19_Mean_CPU,
                               double *Layer19_StanDev_CPU,
                               double *Layer19_Gamma_CPU,
                               double *Layer19_Beta_CPU);

void Execute_Nineteenth_Layer(
    double *Layer19_Neurons_GPU,
    double *Layer20_Neurons_GPU);

void Read_TwentyLayer_Data(double *Layer20_Weights_CPU,
                           double *Layer20_Mean_CPU,
                           double *Layer20_StanDev_CPU,
                           double *Layer20_Gamma_CPU,
                           double *Layer20_Beta_CPU);

void Execute_Twenty_Layer(
    double *Layer20_Neurons_GPU,
    double *Layer21_Neurons_GPU);

void Read_TwentyOneLayer_Data(double *Layer21_Weights_CPU,
                              double *Layer21_Mean_CPU,
                              double *Layer21_StanDev_CPU,
                              double *Layer21_Gamma_CPU,
                              double *Layer21_Beta_CPU);

void Execute_TwentyOne_Layer(
    double *Layer21_Neurons_GPU,
    double *Layer22_Neurons_GPU);

void Read_TwentyTwoLayer_Data(double *Layer22_Weights_CPU,
                              double *Layer22_Mean_CPU,
                              double *Layer22_StanDev_CPU,
                              double *Layer22_Gamma_CPU,
                              double *Layer22_Beta_CPU);

void Execute_TwentyTwo_Layer(
    double *Layer22_Neurons_GPU,
    double *Layer23_Neurons_GPU);

void Read_TwentyThreeLayer_Data(double *Layer23_Weights_CPU,
                                double *Layer23_Mean_CPU,
                                double *Layer23_StanDev_CPU,
                                double *Layer23_Gamma_CPU,
                                double *Layer23_Beta_CPU);

void Execute_TwentyThree_Layer(
    double *Layer23_Neurons_GPU,
    double *Layer24_Neurons_GPU);

void Read_TwentyFourLayer_Data(double *Layer24_Weights_CPU,
                               double *Layer24_Mean_CPU,
                               double *Layer24_StanDev_CPU,
                               double *Layer24_Gamma_CPU,
                               double *Layer24_Beta_CPU);

void Execute_TwentyFour_Layer(
    double *Layer24_Neurons_GPU,
    double *Layer25_Neurons_GPU);

void Read_TwentyFiveLayer_Data(double *Layer25_Weights_CPU,
                               double *Layer25_Mean_CPU,
                               double *Layer25_StanDev_CPU,
                               double *Layer25_Gamma_CPU,
                               double *Layer25_Beta_CPU);

void Execute_TwentyFive_Layer(
    double *Layer25_Neurons_GPU,
    double *Layer26_Neurons_GPU);

void Read_TwentySixLayer_Data(double *Layer26_Weights_CPU,
                              double *Layer26_Mean_CPU,
                              double *Layer26_StanDev_CPU,
                              double *Layer26_Gamma_CPU,
                              double *Layer26_Beta_CPU);

void Execute_TwentySix_Layer(
    double *Layer26_Neurons_GPU,
    double *Layer27_Neurons_GPU);

void Read_TwentySevenLayer_Data(double *Layer27_Weights_CPU,
                                double *Layer27_Mean_CPU,
                                double *Layer27_StanDev_CPU,
                                double *Layer27_Gamma_CPU,
                                double *Layer27_Beta_CPU);

void Execute_TwentySeven_Layer(
    double *Layer27_Neurons_GPU,
    double *Layer28_Neurons_GPU);

// Global Average Pooling Layer
void Execute_TwentyEight_Layer(
    double *Layer28_Neurons_GPU,
    double *Layer29_Neurons_GPU);

// Fully Connected Layer
void Execute_TwentyNine_Layer(
    double *Layer29_Neurons_GPU,
    double *Layer30_Neurons_GPU);

void Read_TwentyNineLayer_Data(double *Layer29_Weights_CPU,
                               double *Layer29_Bias_CPU);

int main()
{

    struct timespec start, stop;
    double time;
    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }
    NeuralNetwork();
    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }

    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;

    printf("Execution time is %f ns\n", time * 1e9);
}

void NeuralNetwork()
{
    FILE *fOutput;
    int value;

    /* ************************************************ FIRST LAYER ******************************************************** */
    double *Layer2_Neurons_GPU = NULL;
    cudaMalloc((void **)&Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);

    Execute_First_Layer(Layer2_Neurons_GPU);

    // Saving output of the first layer: Initially Not Saved
    bool SAVE_FIRST_LAYER_WEIGHTS = true;
    if (SAVE_FIRST_LAYER_WEIGHTS)
    {

        double *Layer2_Neurons_CPU = (double *)malloc(sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer2_Neurons_CPU, Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FirstLayer/output.txt", "w");
        value = FIRST_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer2_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer2_Neurons_CPU);
    }
    // printf("\n Layer 1 Execution complete !!!");
    /* ************************************************ FIRST LAYER COMPLETE *********************************************** */

    /* ************************************************ SECOND LAYER ******************************************************** */
    double *Layer3_Neurons_GPU;
    cudaMalloc((void **)&Layer3_Neurons_GPU, sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);

    Execute_Second_Layer(Layer2_Neurons_GPU, Layer3_Neurons_GPU);

    bool SAVE_SECOND_LAYER_WEIGHTS = false;
    if (SAVE_SECOND_LAYER_WEIGHTS)
    {

        double *Layer3_Neurons_CPU = (double *)malloc(sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer3_Neurons_CPU, Layer3_Neurons_GPU, sizeof(double) * SECOND_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SecondLayer/output.txt", "w");
        value = SECOND_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer3_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer3_Neurons_CPU);
    }
    cudaFree(Layer2_Neurons_GPU);
    // printf("\n Layer 2 Execution complete !!!");
    /* ************************************************ SECOND LAYER COMPLETE *********************************************** */

    /* ************************************************ THIRD LAYER ******************************************************** */
    double *Layer4_Neurons_GPU;
    cudaMalloc((void **)&Layer4_Neurons_GPU, sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);

    Execute_Third_Layer(Layer3_Neurons_GPU, Layer4_Neurons_GPU);

    bool SAVE_THIRD_LAYER_WEIGHTS = false;
    if (SAVE_THIRD_LAYER_WEIGHTS)
    {
        double *Layer4_Neurons_CPU = (double *)malloc(sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer4_Neurons_CPU, Layer4_Neurons_GPU, sizeof(double) * THIRD_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/ThirdLayer/output.txt", "w");
        value = THIRD_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer4_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer4_Neurons_CPU);
    }
    cudaFree(Layer3_Neurons_GPU);
    // printf("\n Layer 3 Execution complete !!!");
    /* ************************************************ THIRD LAYER COMPLETE *********************************************** */

    /* ************************************************ FOURTH LAYER ******************************************************** */
    double *Layer5_Neurons_GPU;
    cudaMalloc((void **)&Layer5_Neurons_GPU, sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE);

    Execute_Fourth_Layer(Layer4_Neurons_GPU, Layer5_Neurons_GPU);

    bool SAVE_FOURTH_LAYER_WEIGHTS = false;
    if (SAVE_FOURTH_LAYER_WEIGHTS)
    {
        double *Layer5_Neurons_CPU = (double *)malloc(sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer5_Neurons_CPU, Layer5_Neurons_GPU, sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FourthLayer/output.txt", "w");
        value = FOURTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer5_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer5_Neurons_CPU);
    }
    cudaFree(Layer4_Neurons_GPU);
    // printf("\n Layer 4 Execution complete !!!");
    /* ************************************************ FOURTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FIFTH LAYER ******************************************************** */
    double *Layer6_Neurons_GPU;
    cudaMalloc((void **)&Layer6_Neurons_GPU, sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE);

    Execute_Fifth_Layer(Layer5_Neurons_GPU, Layer6_Neurons_GPU);

    bool SAVE_FIFTH_LAYER_WEIGHTS = false;
    if (SAVE_FIFTH_LAYER_WEIGHTS)
    {
        double *Layer6_Neurons_CPU = (double *)malloc(sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer6_Neurons_CPU, Layer6_Neurons_GPU, sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FifthLayer/output.txt", "w");
        value = FIFTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer6_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer6_Neurons_CPU);
    }
    cudaFree(Layer5_Neurons_GPU);
    // printf("\n Layer 5 Execution complete !!!");
    /* ************************************************ FIFTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SIXTH LAYER ******************************************************** */
    double *Layer7_Neurons_GPU;
    cudaMalloc((void **)&Layer7_Neurons_GPU, sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE);

    Execute_Sixth_Layer(Layer6_Neurons_GPU, Layer7_Neurons_GPU);

    bool SAVE_SIXTH_LAYER_WEIGHTS = false;
    if (SAVE_SIXTH_LAYER_WEIGHTS)
    {
        double *Layer7_Neurons_CPU = (double *)malloc(sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer7_Neurons_CPU, Layer7_Neurons_GPU, sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SixthLayer/output.txt", "w");
        value = SIXTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer7_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer7_Neurons_CPU);
    }
    cudaFree(Layer6_Neurons_GPU);
    // printf("\n Layer 6 Execution complete !!!");
    /* ************************************************ SIXTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SEVENTH LAYER START ******************************************************** */
    double *Layer8_Neurons_GPU;
    cudaMalloc((void **)&Layer8_Neurons_GPU, sizeof(double) * SEVENTH_LAYER_OUTPUT_SIZE);

    Execute_Seventh_Layer(Layer7_Neurons_GPU, Layer8_Neurons_GPU);

    bool SAVE_SEVENTH_LAYER_WEIGHTS = false;
    if (SAVE_SEVENTH_LAYER_WEIGHTS)
    {
        double *Layer8_Neurons_CPU = (double *)malloc(sizeof(double) * SEVENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer8_Neurons_CPU, Layer8_Neurons_GPU, sizeof(double) * SEVENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SeventhLayer/output.txt", "w");
        value = SEVENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer8_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer8_Neurons_CPU);
    }
    cudaFree(Layer7_Neurons_GPU);
    // printf("\n Layer 7 Execution complete !!!");
    /* ************************************************ SEVENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ EIGHTH LAYER START ******************************************************** */
    double *Layer9_Neurons_GPU;
    cudaMalloc((void **)&Layer9_Neurons_GPU, sizeof(double) * EIGHTH_LAYER_OUTPUT_SIZE);

    Execute_Eighth_Layer(Layer8_Neurons_GPU, Layer9_Neurons_GPU);

    bool SAVE_EIGHTH_LAYER_WEIGHTS = false;
    if (SAVE_EIGHTH_LAYER_WEIGHTS)
    {
        double *Layer9_Neurons_CPU = (double *)malloc(sizeof(double) * EIGHTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer9_Neurons_CPU, Layer9_Neurons_GPU, sizeof(double) * EIGHTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/EighthLayer/output.txt", "w");
        value = EIGHTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer9_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer9_Neurons_CPU);
    }
    cudaFree(Layer8_Neurons_GPU);
    // printf("\n Layer 8 Execution complete !!!");
    /* ************************************************ EIGHTH LAYER COMPLETE *********************************************** */

    /* ************************************************ NINTH LAYER START ******************************************************** */
    double *Layer10_Neurons_GPU;
    cudaMalloc((void **)&Layer10_Neurons_GPU, sizeof(double) * NINTH_LAYER_OUTPUT_SIZE);

    Execute_Ninth_Layer(Layer9_Neurons_GPU, Layer10_Neurons_GPU);

    bool SAVE_NINTH_LAYER_WEIGHTS = false;
    if (SAVE_NINTH_LAYER_WEIGHTS)
    {
        double *Layer10_Neurons_CPU = (double *)malloc(sizeof(double) * NINTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer10_Neurons_CPU, Layer10_Neurons_GPU, sizeof(double) * NINTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/NinthLayer/output.txt", "w");
        value = NINTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer10_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer10_Neurons_CPU);
    }
    cudaFree(Layer9_Neurons_GPU);
    // printf("\n Layer 9 Execution complete !!!");
    /* ************************************************ NINTH LAYER COMPLETE *********************************************** */

    /* ************************************************ TENTH LAYER START ******************************************************** */
    double *Layer11_Neurons_GPU;
    cudaMalloc((void **)&Layer11_Neurons_GPU, sizeof(double) * TENTH_LAYER_OUTPUT_SIZE);

    Execute_Tenth_Layer(Layer10_Neurons_GPU, Layer11_Neurons_GPU);

    bool SAVE_TENTH_LAYER_WEIGHTS = false;
    if (SAVE_TENTH_LAYER_WEIGHTS)
    {
        double *Layer11_Neurons_CPU = (double *)malloc(sizeof(double) * TENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer11_Neurons_CPU, Layer11_Neurons_GPU, sizeof(double) * TENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TenthLayer/output.txt", "w");
        value = TENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer11_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer11_Neurons_CPU);
    }
    cudaFree(Layer10_Neurons_GPU);
    // printf("\n Layer 10 Execution complete !!!");
    /* ************************************************ TENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ ELEVENTH LAYER START ******************************************************** */
    double *Layer12_Neurons_GPU;
    cudaMalloc((void **)&Layer12_Neurons_GPU, sizeof(double) * ELEVENTH_LAYER_OUTPUT_SIZE);

    Execute_Eleventh_Layer(Layer11_Neurons_GPU, Layer12_Neurons_GPU);

    bool SAVE_ELEVENTH_LAYER_WEIGHTS = false;
    if (SAVE_ELEVENTH_LAYER_WEIGHTS)
    {
        double *Layer12_Neurons_CPU = (double *)malloc(sizeof(double) * ELEVENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer12_Neurons_CPU, Layer12_Neurons_GPU, sizeof(double) * ELEVENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/EleventhLayer/output.txt", "w");
        value = ELEVENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer12_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer12_Neurons_CPU);
    }
    cudaFree(Layer11_Neurons_GPU);
    // printf("\n Layer 11 Execution complete !!!");
    /* ************************************************ ELEVENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ TWELVETH LAYER START ******************************************************** */
    double *Layer13_Neurons_GPU;
    cudaMalloc((void **)&Layer13_Neurons_GPU, sizeof(double) * TWELFTH_LAYER_OUTPUT_SIZE);

    Execute_Twelveth_Layer(Layer12_Neurons_GPU, Layer13_Neurons_GPU);

    bool SAVE_TWELVETH_LAYER_WEIGHTS = false;
    if (SAVE_TWELVETH_LAYER_WEIGHTS)
    {
        double *Layer13_Neurons_CPU = (double *)malloc(sizeof(double) * TWELFTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer13_Neurons_CPU, Layer13_Neurons_GPU, sizeof(double) * TWELFTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwelvethLayer/output.txt", "w");
        value = TWELFTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer13_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer13_Neurons_CPU);
    }
    cudaFree(Layer12_Neurons_GPU);
    // printf("\n Layer 12 Execution complete !!!");
    /* ************************************************ TWELVETH LAYER COMPLETE *********************************************** */

    /* ************************************************ THIRTEENTH LAYER START ******************************************************** */
    double *Layer14_Neurons_GPU;
    cudaMalloc((void **)&Layer14_Neurons_GPU, sizeof(double) * THIRTEENTH_LAYER_OUTPUT_SIZE);

    Execute_Thirteenth_Layer(Layer13_Neurons_GPU, Layer14_Neurons_GPU);

    bool SAVE_THIRTEENTH_LAYER_WEIGHTS = false;
    if (SAVE_THIRTEENTH_LAYER_WEIGHTS)
    {
        double *Layer14_Neurons_CPU = (double *)malloc(sizeof(double) * THIRTEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer14_Neurons_CPU, Layer14_Neurons_GPU, sizeof(double) * THIRTEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/ThirteenthLayer/output.txt", "w");
        value = THIRTEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer14_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer14_Neurons_CPU);
    }
    cudaFree(Layer13_Neurons_GPU);
    // printf("\n Layer 13 Execution complete !!!");
    /* ************************************************ THIRTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FOURTEENTH LAYER START ******************************************************** */
    double *Layer15_Neurons_GPU;
    cudaMalloc((void **)&Layer15_Neurons_GPU, sizeof(double) * FOURTEENTH_LAYER_OUTPUT_SIZE);

    Execute_Fourteenth_Layer(Layer14_Neurons_GPU, Layer15_Neurons_GPU);

    bool SAVE_FOURTEENTH_LAYER_WEIGHTS = false;
    if (SAVE_FOURTEENTH_LAYER_WEIGHTS)
    {
        double *Layer15_Neurons_CPU = (double *)malloc(sizeof(double) * FOURTEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer15_Neurons_CPU, Layer15_Neurons_GPU, sizeof(double) * FOURTEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FourteenthLayer/output.txt", "w");
        value = FOURTEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer15_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer15_Neurons_CPU);
    }
    cudaFree(Layer14_Neurons_GPU);
    // printf("\n Layer 14 Execution complete !!!");
    /* ************************************************ FOURTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FIFTEENTH LAYER START ******************************************************** */
    double *Layer16_Neurons_GPU;
    cudaMalloc((void **)&Layer16_Neurons_GPU, sizeof(double) * FIFTEENTH_LAYER_OUTPUT_SIZE);

    Execute_Fifteenth_Layer(Layer15_Neurons_GPU, Layer16_Neurons_GPU);

    bool SAVE_FIFTEENTH_LAYER_WEIGHTS = false;
    if (SAVE_FIFTEENTH_LAYER_WEIGHTS)
    {
        double *Layer16_Neurons_CPU = (double *)malloc(sizeof(double) * FIFTEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer16_Neurons_CPU, Layer16_Neurons_GPU, sizeof(double) * FIFTEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FifteenthLayer/output.txt", "w");
        value = FIFTEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer16_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer16_Neurons_CPU);
    }
    cudaFree(Layer15_Neurons_GPU);
    // printf("\n Layer 15 Execution complete !!!");
    /* ************************************************ FIFTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SIXTEENTH LAYER START ******************************************************** */
    double *Layer17_Neurons_GPU;
    cudaMalloc((void **)&Layer17_Neurons_GPU, sizeof(double) * SIXTEENTH_LAYER_OUTPUT_SIZE);

    Execute_Sixteenth_Layer(Layer16_Neurons_GPU, Layer17_Neurons_GPU);

    bool SAVE_SIXTEENTH_LAYER_WEIGHTS = false;
    if (SAVE_SIXTEENTH_LAYER_WEIGHTS)
    {
        double *Layer17_Neurons_CPU = (double *)malloc(sizeof(double) * SIXTEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer17_Neurons_CPU, Layer17_Neurons_GPU, sizeof(double) * SIXTEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SixteenthLayer/output.txt", "w");
        value = SIXTEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer17_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer17_Neurons_CPU);
    }
    cudaFree(Layer16_Neurons_GPU);
    // printf("\n Layer 16 Execution complete !!!");
    /* ************************************************ SIXTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SEVENTEENTH LAYER START ******************************************************** */
    double *Layer18_Neurons_GPU;
    cudaMalloc((void **)&Layer18_Neurons_GPU, sizeof(double) * SEVENTEENTH_LAYER_OUTPUT_SIZE);

    Execute_Seventeenth_Layer(Layer17_Neurons_GPU, Layer18_Neurons_GPU);

    bool SAVE_SEVENTEENTH_LAYER_WEIGHTS = false;
    if (SAVE_SEVENTEENTH_LAYER_WEIGHTS)
    {
        double *Layer18_Neurons_CPU = (double *)malloc(sizeof(double) * SEVENTEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer18_Neurons_CPU, Layer18_Neurons_GPU, sizeof(double) * SEVENTEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SeventeenthLayer/output.txt", "w");
        value = SEVENTEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer18_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer18_Neurons_CPU);
    }
    cudaFree(Layer17_Neurons_GPU);
    // printf("\n Layer 17 Execution complete !!!");
    /* ************************************************ SEVENTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ EIGHTEENTH LAYER START ******************************************************** */
    double *Layer19_Neurons_GPU;
    cudaMalloc((void **)&Layer19_Neurons_GPU, sizeof(double) * EIGHTEENTH_LAYER_OUTPUT_SIZE);

    Execute_Eighteenth_Layer(Layer18_Neurons_GPU, Layer19_Neurons_GPU);

    bool SAVE_EIGHTEENTH_LAYER_WEIGHTS = false;
    if (SAVE_EIGHTEENTH_LAYER_WEIGHTS)
    {
        double *Layer19_Neurons_CPU = (double *)malloc(sizeof(double) * EIGHTEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer19_Neurons_CPU, Layer19_Neurons_GPU, sizeof(double) * EIGHTEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/EighteenthLayer/output.txt", "w");
        value = EIGHTEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer19_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer19_Neurons_CPU);
    }
    cudaFree(Layer18_Neurons_GPU);
    // printf("\n Layer 18 Execution complete !!!");
    /* ************************************************ EIGHTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ NINETEENTH LAYER START ******************************************************** */
    double *Layer20_Neurons_GPU;
    cudaMalloc((void **)&Layer20_Neurons_GPU, sizeof(double) * NINETEENTH_LAYER_OUTPUT_SIZE);

    Execute_Nineteenth_Layer(Layer19_Neurons_GPU, Layer20_Neurons_GPU);

    bool SAVE_NINETEENTH_LAYER_WEIGHTS = false;
    if (SAVE_NINETEENTH_LAYER_WEIGHTS)
    {
        double *Layer20_Neurons_CPU = (double *)malloc(sizeof(double) * NINETEENTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer20_Neurons_CPU, Layer20_Neurons_GPU, sizeof(double) * NINETEENTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/NineteenthLayer/output.txt", "w");
        value = NINETEENTH_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer20_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer20_Neurons_CPU);
    }
    cudaFree(Layer19_Neurons_GPU);
    // printf("\n Layer 19 Execution complete !!!");
    /* ************************************************ NINETEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY LAYER START ******************************************************** */
    double *Layer21_Neurons_GPU;
    cudaMalloc((void **)&Layer21_Neurons_GPU, sizeof(double) * TWENTY_LAYER_OUTPUT_SIZE);

    Execute_Twenty_Layer(Layer20_Neurons_GPU, Layer21_Neurons_GPU);

    bool SAVE_TWENTY_LAYER_WEIGHTS = false;
    if (SAVE_TWENTY_LAYER_WEIGHTS)
    {
        double *Layer21_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTY_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer21_Neurons_CPU, Layer21_Neurons_GPU, sizeof(double) * TWENTY_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyLayer/output.txt", "w");
        value = TWENTY_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer21_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer21_Neurons_CPU);
    }
    cudaFree(Layer20_Neurons_GPU);
    // printf("\n Layer 20 Execution complete !!!");
    /* ************************************************ TWENTY LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYONE LAYER START ******************************************************** */
    double *Layer22_Neurons_GPU;
    cudaMalloc((void **)&Layer22_Neurons_GPU, sizeof(double) * TWENTYONE_LAYER_OUTPUT_SIZE);

    Execute_TwentyOne_Layer(Layer21_Neurons_GPU, Layer22_Neurons_GPU);

    bool SAVE_TWENTYONE_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYONE_LAYER_WEIGHTS)
    {
        double *Layer22_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYONE_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer22_Neurons_CPU, Layer22_Neurons_GPU, sizeof(double) * TWENTYONE_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyOneLayer/output.txt", "w");
        value = TWENTYONE_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer22_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer22_Neurons_CPU);
    }
    cudaFree(Layer21_Neurons_GPU);
    // printf("\n Layer 21 Execution complete !!!");
    /* ************************************************ TWENTYONE LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYTWO LAYER START ******************************************************** */
    double *Layer23_Neurons_GPU;
    cudaMalloc((void **)&Layer23_Neurons_GPU, sizeof(double) * TWENTYTWO_LAYER_OUTPUT_SIZE);

    Execute_TwentyTwo_Layer(Layer22_Neurons_GPU, Layer23_Neurons_GPU);

    bool SAVE_TWENTYTWO_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYTWO_LAYER_WEIGHTS)
    {
        double *Layer23_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYTWO_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer23_Neurons_CPU, Layer23_Neurons_GPU, sizeof(double) * TWENTYTWO_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyTwoLayer/output.txt", "w");
        value = TWENTYTWO_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer23_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer23_Neurons_CPU);
    }
    cudaFree(Layer22_Neurons_GPU);
    // printf("\n Layer 22 Execution complete !!!");
    /* ************************************************ TWENTYTWO LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYTHREE LAYER START ******************************************************** */
    double *Layer24_Neurons_GPU;
    cudaMalloc((void **)&Layer24_Neurons_GPU, sizeof(double) * TWENTYTHREE_LAYER_OUTPUT_SIZE);

    Execute_TwentyThree_Layer(Layer23_Neurons_GPU, Layer24_Neurons_GPU);

    bool SAVE_TWENTYTHREE_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYTHREE_LAYER_WEIGHTS)
    {
        double *Layer24_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYTHREE_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer24_Neurons_CPU, Layer24_Neurons_GPU, sizeof(double) * TWENTYTHREE_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyThreeLayer/output.txt", "w");
        value = TWENTYTHREE_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer24_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer24_Neurons_CPU);
    }
    cudaFree(Layer23_Neurons_GPU);
    // printf("\n Layer 23 Execution complete !!!");
    /* ************************************************ TWENTYTHREE LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYFOUR LAYER START ******************************************************** */
    double *Layer25_Neurons_GPU;
    cudaMalloc((void **)&Layer25_Neurons_GPU, sizeof(double) * TWENTYFOUR_LAYER_OUTPUT_SIZE);

    Execute_TwentyFour_Layer(Layer24_Neurons_GPU, Layer25_Neurons_GPU);

    bool SAVE_TWENTYFOUR_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYFOUR_LAYER_WEIGHTS)
    {
        double *Layer25_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYFOUR_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer25_Neurons_CPU, Layer25_Neurons_GPU, sizeof(double) * TWENTYFOUR_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyFourLayer/output.txt", "w");
        value = TWENTYFOUR_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer25_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer25_Neurons_CPU);
    }
    cudaFree(Layer24_Neurons_GPU);
    // printf("\n Layer 24 Execution complete !!!");
    /* ************************************************ TWENTYFOUR LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYFIVE LAYER START ******************************************************** */
    double *Layer26_Neurons_GPU;
    cudaMalloc((void **)&Layer26_Neurons_GPU, sizeof(double) * TWENTYFIVE_LAYER_OUTPUT_SIZE);

    Execute_TwentyFive_Layer(Layer25_Neurons_GPU, Layer26_Neurons_GPU);

    bool SAVE_TWENTYFIVE_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYFIVE_LAYER_WEIGHTS)
    {
        double *Layer26_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYFIVE_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer26_Neurons_CPU, Layer26_Neurons_GPU, sizeof(double) * TWENTYFIVE_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyFiveLayer/output.txt", "w");
        value = TWENTYFIVE_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer26_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer26_Neurons_CPU);
    }
    cudaFree(Layer25_Neurons_GPU);
    // printf("\n Layer 25 Execution complete !!!");
    /* ************************************************ TWENTYFIVE LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYSIX LAYER START ******************************************************** */
    double *Layer27_Neurons_GPU;
    cudaMalloc((void **)&Layer27_Neurons_GPU, sizeof(double) * TWENTYSIX_LAYER_OUTPUT_SIZE);

    Execute_TwentySix_Layer(Layer26_Neurons_GPU, Layer27_Neurons_GPU);

    bool SAVE_TWENTYSIX_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYSIX_LAYER_WEIGHTS)
    {
        double *Layer27_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYSIX_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer27_Neurons_CPU, Layer27_Neurons_GPU, sizeof(double) * TWENTYSIX_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentySixLayer/output.txt", "w");
        value = TWENTYSIX_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer27_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer27_Neurons_CPU);
    }
    cudaFree(Layer26_Neurons_GPU);
    // printf("\n Layer 26 Execution complete !!!");
    /* ************************************************ TWENTYSIX LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYSEVEN LAYER START ******************************************************** */
    double *Layer28_Neurons_GPU;
    cudaMalloc((void **)&Layer28_Neurons_GPU, sizeof(double) * TWENTYSEVEN_LAYER_OUTPUT_SIZE);

    Execute_TwentySeven_Layer(Layer27_Neurons_GPU, Layer28_Neurons_GPU);

    bool SAVE_TWENTYSEVEN_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYSEVEN_LAYER_WEIGHTS)
    {
        double *Layer28_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYSEVEN_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer28_Neurons_CPU, Layer28_Neurons_GPU, sizeof(double) * TWENTYSEVEN_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentySevenLayer/output.txt", "w");
        value = TWENTYSEVEN_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer28_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer28_Neurons_CPU);
    }
    cudaFree(Layer27_Neurons_GPU);
    // printf("\n Layer 27 Execution complete !!!");
    /* ************************************************ TWENTYSEVEN LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYEIGHT LAYER START ******************************************************** */
    double *Layer29_Neurons_GPU;
    cudaMalloc((void **)&Layer29_Neurons_GPU, sizeof(double) * TWENTYEIGHT_LAYER_OUTPUT_SIZE);

    Execute_TwentyEight_Layer(Layer28_Neurons_GPU, Layer29_Neurons_GPU);

    bool SAVE_TWENTYEIGHT_LAYER_WEIGHTS = false;
    if (SAVE_TWENTYEIGHT_LAYER_WEIGHTS)
    {
        double *Layer29_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYEIGHT_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer29_Neurons_CPU, Layer29_Neurons_GPU, sizeof(double) * TWENTYEIGHT_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyEightLayer/output.txt", "w");
        value = TWENTYEIGHT_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer29_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer29_Neurons_CPU);
    }
    cudaFree(Layer28_Neurons_GPU);
    // printf("\n Layer 28 Execution complete !!!");
    /* ************************************************ TWENTYEIGHT LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTYNINE LAYER START ******************************************************** */
    double *Layer30_Neurons_GPU;
    cudaMalloc((void **)&Layer30_Neurons_GPU, sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE);

    Execute_TwentyNine_Layer(Layer29_Neurons_GPU, Layer30_Neurons_GPU);

    bool SAVE_TWENTYNINE_LAYER_WEIGHTS = true;
    if (SAVE_TWENTYNINE_LAYER_WEIGHTS)
    {
        double *Layer30_Neurons_CPU = (double *)malloc(sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer30_Neurons_CPU, Layer30_Neurons_GPU, sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        fOutput = fopen("data/TwentyNineLayer/output_w.txt", "w");
        value = TWENTYNINE_LAYER_OUTPUT_SIZE;
        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer30_Neurons_CPU[i]);
        }
        fclose(fOutput);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyNineLayer/output.txt", "w");
        value = TWENTYNINE_LAYER_OUTPUT_SIZE;
        double sum = 0.0;
        for (int i = 0; i < value; i++)
        {
            sum += exp(Layer30_Neurons_CPU[i]);
        }

        for (int i = 0; i < value; i++)
        {
            Layer30_Neurons_CPU[i] = (exp(Layer30_Neurons_CPU[i]) / sum);
        }

        for (int i = 0; i < value; i++)
        {
            fprintf(fOutput, "%0.6lf\n", Layer30_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer30_Neurons_CPU);
    }
    cudaFree(Layer29_Neurons_GPU);
    // printf("\n Layer 29 Execution complete !!!");
    /* ************************************************ TWENTYNINE LAYER COMPLETE *********************************************** */

    printf("\n\n Processing Done !!! \n\n");

    cudaFree(Layer30_Neurons_GPU);
}

void Execute_First_Layer(double *Layer2_Neurons_GPU)
{
    double *Layer1_Neurons_CPU = (double *)malloc(sizeof(double) * INPUT_LAYER_SIZE);
    double *Layer1_Weights_CPU = (double *)malloc(sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    double *Layer1_Mean_CPU = (double *)malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double *Layer1_StanDev_CPU = (double *)malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double *Layer1_Gamma_CPU = (double *)malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double *Layer1_Beta_CPU = (double *)malloc(sizeof(double) * FIRST_LAYER_CHANNELS);

    Read_First_Layer_Data(
        Layer1_Neurons_CPU,
        Layer1_Weights_CPU,
        Layer1_Mean_CPU,
        Layer1_StanDev_CPU,
        Layer1_Gamma_CPU,
        Layer1_Beta_CPU);

    // Copy memory from Host to Kernel
    double *Layer1_Weights_GPU,
        *Layer1_Neurons_GPU,
        *Layer1_Mean_GPU,
        *Layer1_StanDev_GPU,
        *Layer1_Gamma_GPU,
        *Layer1_Beta_GPU;

    cudaMalloc((void **)&Layer1_Neurons_GPU, sizeof(double) * INPUT_LAYER_SIZE);
    cudaMalloc((void **)&Layer1_Weights_GPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer1_Mean_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer1_StanDev_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer1_Gamma_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer1_Beta_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);

    cudaMemcpy(Layer1_Neurons_GPU, Layer1_Neurons_CPU, sizeof(double) * INPUT_LAYER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Weights_GPU, Layer1_Weights_CPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Mean_GPU, Layer1_Mean_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_StanDev_GPU, Layer1_StanDev_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Gamma_GPU, Layer1_Gamma_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Beta_GPU, Layer1_Beta_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer1_Mean_CPU);
    free(Layer1_StanDev_CPU);
    free(Layer1_Gamma_CPU);
    free(Layer1_Beta_CPU);

    // Kernel Launch
    dim3 gridSizeA(32, 3, 3);
    dim3 blockSizeA(32, 32);

    executeFirstLayer_CONV3D_partA<<<gridSizeA, blockSizeA>>>(Layer1_Neurons_GPU,
                                                              Layer1_Weights_GPU,
                                                              Layer2_Neurons_GPU,
                                                              Layer1_Mean_GPU,
                                                              Layer1_StanDev_GPU,
                                                              Layer1_Gamma_GPU,
                                                              Layer1_Beta_GPU);

    dim3 gridSizeB(32, 7);
    dim3 blockSizeB(16, 16);

    executeFirstLayer_CONV3D_partB<<<gridSizeB, blockSizeB>>>(Layer1_Neurons_GPU,
                                                              Layer1_Weights_GPU,
                                                              Layer2_Neurons_GPU,
                                                              Layer1_Mean_GPU,
                                                              Layer1_StanDev_GPU,
                                                              Layer1_Gamma_GPU,
                                                              Layer1_Beta_GPU);

    dim3 gridSizeC(32, 6);
    dim3 blockSizeC(16, 16);

    executeFirstLayer_CONV3D_partC<<<gridSizeC, blockSizeC>>>(Layer1_Neurons_GPU,
                                                              Layer1_Weights_GPU,
                                                              Layer2_Neurons_GPU,
                                                              Layer1_Mean_GPU,
                                                              Layer1_StanDev_GPU,
                                                              Layer1_Gamma_GPU,
                                                              Layer1_Beta_GPU);

    cudaDeviceSynchronize();

    // First Layer GPU Memory Free
    cudaFree(Layer1_Neurons_GPU);
    cudaFree(Layer1_Weights_GPU);
    cudaFree(Layer1_Mean_GPU);
    cudaFree(Layer1_StanDev_GPU);
    cudaFree(Layer1_Gamma_GPU);
    cudaFree(Layer1_Beta_GPU);
}

void Read_First_Layer_Data(
    double *Layer1_Neurons_CPU,
    double *Layer1_Weights_CPU,
    double *Layer1_Mean_CPU,
    double *Layer1_StanDev_CPU,
    double *Layer1_Gamma_CPU,
    double *Layer1_Beta_CPU)
{
    read_Input_File("data/FirstLayer/InputFiles/inputsNorm.txt", Layer1_Neurons_CPU);
    read_File("data/FirstLayer/weightsNorm.txt", Layer1_Weights_CPU);
    read_File("data/FirstLayer/First_Layer_Mean.txt", Layer1_Mean_CPU);
    read_File("data/FirstLayer/First_Layer_StanDev.txt", Layer1_StanDev_CPU);
    read_File("data/FirstLayer/First_Layer_Gamma.txt", Layer1_Gamma_CPU);
    read_File("data/FirstLayer/First_Layer_Beta.txt", Layer1_Beta_CPU);
}

void Execute_Second_Layer(
    double *Layer2_Neurons_GPU,
    double *Layer3_Neurons_GPU)
{
    double *Layer2_Weights_CPU = (double *)malloc(sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
    double *Layer2_Mean_CPU = (double *)malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double *Layer2_StanDev_CPU = (double *)malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double *Layer2_Gamma_CPU = (double *)malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double *Layer2_Beta_CPU = (double *)malloc(sizeof(double) * SECOND_LAYER_CHANNELS);

    Read_SecondLayer_Data(Layer2_Weights_CPU,
                          Layer2_Mean_CPU,
                          Layer2_StanDev_CPU,
                          Layer2_Gamma_CPU,
                          Layer2_Beta_CPU);

    double *Layer2_Weights_GPU,
        *Layer2_Mean_GPU,
        *Layer2_StanDev_GPU,
        *Layer2_Gamma_GPU,
        *Layer2_Beta_GPU;
    ;

    cudaMalloc((void **)&Layer2_Weights_GPU, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer2_Mean_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer2_StanDev_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer2_Gamma_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer2_Beta_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);

    cudaMemcpy(Layer2_Weights_GPU, Layer2_Weights_CPU, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Mean_GPU, Layer2_Mean_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_StanDev_GPU, Layer2_StanDev_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Gamma_GPU, Layer2_Gamma_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Beta_GPU, Layer2_Beta_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer2_Weights_CPU);
    free(Layer2_Mean_CPU);
    free(Layer2_StanDev_CPU);
    free(Layer2_Gamma_CPU);
    free(Layer2_Beta_CPU);

    dim3 gridSizeA(32, 3, 3);
    dim3 blockSizeA(32, 32);
    executeSecondLayer_DSC_partA<<<gridSizeA, blockSizeA>>>(Layer2_Neurons_GPU,
                                                            Layer2_Weights_GPU,
                                                            Layer3_Neurons_GPU,
                                                            Layer2_Mean_GPU,
                                                            Layer2_StanDev_GPU,
                                                            Layer2_Gamma_GPU,
                                                            Layer2_Beta_GPU);

    dim3 gridSizeB(32, 7);
    dim3 blockSizeB(16, 16);
    executeSecondLayer_DSC_partB<<<gridSizeB, blockSizeB>>>(Layer2_Neurons_GPU,
                                                            Layer2_Weights_GPU,
                                                            Layer3_Neurons_GPU,
                                                            Layer2_Mean_GPU,
                                                            Layer2_StanDev_GPU,
                                                            Layer2_Gamma_GPU,
                                                            Layer2_Beta_GPU);

    dim3 gridSizeC(32, 6);
    dim3 blockSizeC(16, 16);
    executeSecondLayer_DSC_partC<<<gridSizeC, blockSizeC>>>(Layer2_Neurons_GPU,
                                                            Layer2_Weights_GPU,
                                                            Layer3_Neurons_GPU,
                                                            Layer2_Mean_GPU,
                                                            Layer2_StanDev_GPU,
                                                            Layer2_Gamma_GPU,
                                                            Layer2_Beta_GPU);

    cudaFree(Layer2_Weights_GPU);
    cudaFree(Layer2_Mean_GPU);
    cudaFree(Layer2_StanDev_GPU);
    cudaFree(Layer2_Gamma_GPU);
    cudaFree(Layer2_Beta_GPU);
}

void Read_SecondLayer_Data(double *Layer2_Weights_CPU,
                           double *Layer2_Mean_CPU,
                           double *Layer2_StanDev_CPU,
                           double *Layer2_Gamma_CPU,
                           double *Layer2_Beta_CPU)
{
    read_File("data/SecondLayer/weightsNorm.txt", Layer2_Weights_CPU);
    read_File("data/SecondLayer/Second_Layer_Mean.txt", Layer2_Mean_CPU);
    read_File("data/SecondLayer/Second_Layer_StanDev.txt", Layer2_StanDev_CPU);
    read_File("data/SecondLayer/Second_Layer_Gamma.txt", Layer2_Gamma_CPU);
    read_File("data/SecondLayer/Second_Layer_Beta.txt", Layer2_Beta_CPU);
}

void Execute_Third_Layer(
    double *Layer3_Neurons_GPU,
    double *Layer4_Neurons_GPU)
{
    double *Layer3_Weights_CPU = (double *)malloc(sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    double *Layer3_Mean_CPU = (double *)malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double *Layer3_StanDev_CPU = (double *)malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double *Layer3_Gamma_CPU = (double *)malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double *Layer3_Beta_CPU = (double *)malloc(sizeof(double) * THIRD_LAYER_CHANNELS);

    Read_ThirdLayer_Data(Layer3_Weights_CPU,
                         Layer3_Mean_CPU,
                         Layer3_StanDev_CPU,
                         Layer3_Gamma_CPU,
                         Layer3_Beta_CPU);

    double *Layer3_Weights_GPU,
        *Layer3_Mean_GPU,
        *Layer3_StanDev_GPU,
        *Layer3_Gamma_GPU,
        *Layer3_Beta_GPU;

    cudaMalloc((void **)&Layer3_Weights_GPU, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer3_Mean_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer3_StanDev_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer3_Gamma_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer3_Beta_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);

    cudaMemcpy(Layer3_Weights_GPU, Layer3_Weights_CPU, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Mean_GPU, Layer3_Mean_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_StanDev_GPU, Layer3_StanDev_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Gamma_GPU, Layer3_Gamma_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Beta_GPU, Layer3_Beta_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer3_Weights_CPU);
    free(Layer3_Mean_CPU);
    free(Layer3_StanDev_CPU);
    free(Layer3_Gamma_CPU);
    free(Layer3_Beta_CPU);

    // Execution of the Third Layer
    dim3 gridSizeThirdLayerA(64, 3, 3);
    dim3 blockSizeThirdLayerA(32, 32);
    executeThirdLayer_PSC_partA<<<gridSizeThirdLayerA, blockSizeThirdLayerA>>>(Layer3_Neurons_GPU,
                                                                               Layer3_Weights_GPU,
                                                                               Layer4_Neurons_GPU,
                                                                               Layer3_Mean_GPU,
                                                                               Layer3_StanDev_GPU,
                                                                               Layer3_Gamma_GPU,
                                                                               Layer3_Beta_GPU);

    dim3 gridSizeThirdLayerB(64, 7);
    dim3 blockSizeThirdLayerB(16, 16);
    executeThirdLayer_PSC_partB<<<gridSizeThirdLayerB, blockSizeThirdLayerB>>>(Layer3_Neurons_GPU,
                                                                               Layer3_Weights_GPU,
                                                                               Layer4_Neurons_GPU,
                                                                               Layer3_Mean_GPU,
                                                                               Layer3_StanDev_GPU,
                                                                               Layer3_Gamma_GPU,
                                                                               Layer3_Beta_GPU);

    dim3 gridSizeThirdLayerC(64, 6);
    dim3 blockSizeThirdLayerC(16, 16);
    executeThirdLayer_PSC_partC<<<gridSizeThirdLayerC, blockSizeThirdLayerC>>>(Layer3_Neurons_GPU,
                                                                               Layer3_Weights_GPU,
                                                                               Layer4_Neurons_GPU,
                                                                               Layer3_Mean_GPU,
                                                                               Layer3_StanDev_GPU,
                                                                               Layer3_Gamma_GPU,
                                                                               Layer3_Beta_GPU);

    cudaDeviceSynchronize();

    cudaFree(Layer3_Weights_GPU);
    cudaFree(Layer3_Mean_GPU);
    cudaFree(Layer3_StanDev_GPU);
    cudaFree(Layer3_Gamma_GPU);
    cudaFree(Layer3_Beta_GPU);
}

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU,
                          double *Layer3_Mean_CPU,
                          double *Layer3_StanDev_CPU,
                          double *Layer3_Gamma_CPU,
                          double *Layer3_Beta_CPU)
{
    read_File("data/ThirdLayer/weightsNorm.txt", Layer3_Weights_CPU);
    read_File("data/ThirdLayer/Third_Layer_Mean.txt", Layer3_Mean_CPU);
    read_File("data/ThirdLayer/Third_Layer_StanDev.txt", Layer3_StanDev_CPU);
    read_File("data/ThirdLayer/Third_Layer_Gamma.txt", Layer3_Gamma_CPU);
    read_File("data/ThirdLayer/Third_Layer_Beta.txt", Layer3_Beta_CPU);
}

void Execute_Fourth_Layer(
    double *Layer4_Neurons_GPU,
    double *Layer5_Neurons_GPU)
{
    double *Layer4_Weights_CPU = (double *)malloc(sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE);
    double *Layer4_Mean_CPU = (double *)malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double *Layer4_StanDev_CPU = (double *)malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double *Layer4_Gamma_CPU = (double *)malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double *Layer4_Beta_CPU = (double *)malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);

    Read_FourthLayer_Data(Layer4_Weights_CPU,
                          Layer4_Mean_CPU,
                          Layer4_StanDev_CPU,
                          Layer4_Gamma_CPU,
                          Layer4_Beta_CPU);

    double *Layer4_Weights_GPU,
        *Layer4_Mean_GPU,
        *Layer4_StanDev_GPU,
        *Layer4_Gamma_GPU,
        *Layer4_Beta_GPU;

    cudaMalloc((void **)&Layer4_Weights_GPU, sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer4_Mean_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer4_StanDev_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer4_Gamma_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer4_Beta_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);

    cudaMemcpy(Layer4_Weights_GPU, Layer4_Weights_CPU, sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Mean_GPU, Layer4_Mean_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_StanDev_GPU, Layer4_StanDev_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Gamma_GPU, Layer4_Gamma_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Beta_GPU, Layer4_Beta_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer4_Weights_CPU);
    free(Layer4_Mean_CPU);
    free(Layer4_StanDev_CPU);
    free(Layer4_Gamma_CPU);
    free(Layer4_Beta_CPU);

    dim3 gridSizeFourthLayer(64);
    dim3 blockSizeFourthLayerA(32, 32);
    executeFourthLayer_DSC_partA<<<gridSizeFourthLayer, blockSizeFourthLayerA>>>(Layer4_Neurons_GPU,
                                                                                 Layer4_Weights_GPU,
                                                                                 Layer5_Neurons_GPU,
                                                                                 Layer4_Mean_GPU,
                                                                                 Layer4_StanDev_GPU,
                                                                                 Layer4_Gamma_GPU,
                                                                                 Layer4_Beta_GPU);

    dim3 blockSizeFourthLayerB(32, 24);
    executeFourthLayer_DSC_partB<<<gridSizeFourthLayer, blockSizeFourthLayerB>>>(Layer4_Neurons_GPU,
                                                                                 Layer4_Weights_GPU,
                                                                                 Layer5_Neurons_GPU,
                                                                                 Layer4_Mean_GPU,
                                                                                 Layer4_StanDev_GPU,
                                                                                 Layer4_Gamma_GPU,
                                                                                 Layer4_Beta_GPU);

    dim3 blockSizeFourthLayerC(24, 32);
    executeFourthLayer_DSC_partC<<<gridSizeFourthLayer, blockSizeFourthLayerC>>>(Layer4_Neurons_GPU,
                                                                                 Layer4_Weights_GPU,
                                                                                 Layer5_Neurons_GPU,
                                                                                 Layer4_Mean_GPU,
                                                                                 Layer4_StanDev_GPU,
                                                                                 Layer4_Gamma_GPU,
                                                                                 Layer4_Beta_GPU);

    dim3 blockSizeFourthLayerD(24, 24);
    executeFourthLayer_DSC_partD<<<gridSizeFourthLayer, blockSizeFourthLayerD>>>(Layer4_Neurons_GPU,
                                                                                 Layer4_Weights_GPU,
                                                                                 Layer5_Neurons_GPU,
                                                                                 Layer4_Mean_GPU,
                                                                                 Layer4_StanDev_GPU,
                                                                                 Layer4_Gamma_GPU,
                                                                                 Layer4_Beta_GPU);

    cudaFree(Layer4_Weights_GPU);
    cudaFree(Layer4_Mean_GPU);
    cudaFree(Layer4_StanDev_GPU);
    cudaFree(Layer4_Gamma_GPU);
    cudaFree(Layer4_Beta_GPU);
}

void Read_FourthLayer_Data(double *Layer4_Weights_CPU,
                           double *Layer4_Mean_CPU,
                           double *Layer4_StanDev_CPU,
                           double *Layer4_Gamma_CPU,
                           double *Layer4_Beta_CPU)
{
    read_File("data/FourthLayer/weightsNorm.txt", Layer4_Weights_CPU);
    read_File("data/FourthLayer/Fourth_Layer_Mean.txt", Layer4_Mean_CPU);
    read_File("data/FourthLayer/Fourth_Layer_StanDev.txt", Layer4_StanDev_CPU);
    read_File("data/FourthLayer/Fourth_Layer_Gamma.txt", Layer4_Gamma_CPU);
    read_File("data/FourthLayer/Fourth_Layer_Beta.txt", Layer4_Beta_CPU);
}

void Execute_Fifth_Layer(
    double *Layer5_Neurons_GPU,
    double *Layer6_Neurons_GPU)
{
    double *Layer5_Weights_CPU = (double *)malloc(sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE);
    double *Layer5_Mean_CPU = (double *)malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double *Layer5_StanDev_CPU = (double *)malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double *Layer5_Gamma_CPU = (double *)malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double *Layer5_Beta_CPU = (double *)malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);

    Read_FifthLayer_Data(Layer5_Weights_CPU,
                         Layer5_Mean_CPU,
                         Layer5_StanDev_CPU,
                         Layer5_Gamma_CPU,
                         Layer5_Beta_CPU);

    double *Layer5_Weights_GPU,
        *Layer5_Mean_GPU,
        *Layer5_StanDev_GPU,
        *Layer5_Gamma_GPU,
        *Layer5_Beta_GPU;

    cudaMalloc((void **)&Layer5_Weights_GPU, sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer5_Mean_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer5_StanDev_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer5_Gamma_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer5_Beta_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);

    cudaMemcpy(Layer5_Weights_GPU, Layer5_Weights_CPU, sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Mean_GPU, Layer5_Mean_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_StanDev_GPU, Layer5_StanDev_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Gamma_GPU, Layer5_Gamma_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Beta_GPU, Layer5_Beta_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer5_Weights_CPU);
    free(Layer5_Mean_CPU);
    free(Layer5_StanDev_CPU);
    free(Layer5_Gamma_CPU);
    free(Layer5_Beta_CPU);

    dim3 gridSizeFifthLayer(128);
    dim3 blockSizeFifthLayerA(32, 32);
    executeFifthLayer_PSC_partA<<<gridSizeFifthLayer, blockSizeFifthLayerA>>>(Layer5_Neurons_GPU,
                                                                              Layer5_Weights_GPU,
                                                                              Layer6_Neurons_GPU,
                                                                              Layer5_Mean_GPU,
                                                                              Layer5_StanDev_GPU,
                                                                              Layer5_Gamma_GPU,
                                                                              Layer5_Beta_GPU);

    dim3 blockSizeFifthLayerB(32, 24);
    executeFifthLayer_PSC_partB<<<gridSizeFifthLayer, blockSizeFifthLayerB>>>(Layer5_Neurons_GPU,
                                                                              Layer5_Weights_GPU,
                                                                              Layer6_Neurons_GPU,
                                                                              Layer5_Mean_GPU,
                                                                              Layer5_StanDev_GPU,
                                                                              Layer5_Gamma_GPU,
                                                                              Layer5_Beta_GPU);

    dim3 blockSizeFifthLayerC(24, 32);
    executeFifthLayer_PSC_partC<<<gridSizeFifthLayer, blockSizeFifthLayerC>>>(Layer5_Neurons_GPU,
                                                                              Layer5_Weights_GPU,
                                                                              Layer6_Neurons_GPU,
                                                                              Layer5_Mean_GPU,
                                                                              Layer5_StanDev_GPU,
                                                                              Layer5_Gamma_GPU,
                                                                              Layer5_Beta_GPU);

    dim3 blockSizeFifthLayerD(24, 24);
    executeFifthLayer_PSC_partD<<<gridSizeFifthLayer, blockSizeFifthLayerD>>>(Layer5_Neurons_GPU,
                                                                              Layer5_Weights_GPU,
                                                                              Layer6_Neurons_GPU,
                                                                              Layer5_Mean_GPU,
                                                                              Layer5_StanDev_GPU,
                                                                              Layer5_Gamma_GPU,
                                                                              Layer5_Beta_GPU);

    cudaFree(Layer5_Weights_GPU);
    cudaFree(Layer5_Mean_GPU);
    cudaFree(Layer5_StanDev_GPU);
    cudaFree(Layer5_Gamma_GPU);
    cudaFree(Layer5_Beta_GPU);
}

void Read_FifthLayer_Data(double *Layer5_Weights_CPU,
                          double *Layer5_Mean_CPU,
                          double *Layer5_StanDev_CPU,
                          double *Layer5_Gamma_CPU,
                          double *Layer5_Beta_CPU

)
{
    read_File("data/FifthLayer/weightsNorm.txt", Layer5_Weights_CPU);
    read_File("data/FifthLayer/Fifth_Layer_Mean.txt", Layer5_Mean_CPU);
    read_File("data/FifthLayer/Fifth_Layer_StanDev.txt", Layer5_StanDev_CPU);
    read_File("data/FifthLayer/Fifth_Layer_Gamma.txt", Layer5_Gamma_CPU);
    read_File("data/FifthLayer/Fifth_Layer_Beta.txt", Layer5_Beta_CPU);
}

void Execute_Sixth_Layer(
    double *Layer6_Neurons_GPU,
    double *Layer7_Neurons_GPU)
{
    double *Layer6_Weights_CPU = (double *)malloc(sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE);
    double *Layer6_Mean_CPU = (double *)malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double *Layer6_StanDev_CPU = (double *)malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double *Layer6_Gamma_CPU = (double *)malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double *Layer6_Beta_CPU = (double *)malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);

    Read_SixthLayer_Data(Layer6_Weights_CPU,
                         Layer6_Mean_CPU,
                         Layer6_StanDev_CPU,
                         Layer6_Gamma_CPU,
                         Layer6_Beta_CPU);

    double *Layer6_Weights_GPU,
        *Layer6_Mean_GPU,
        *Layer6_StanDev_GPU,
        *Layer6_Gamma_GPU,
        *Layer6_Beta_GPU;

    cudaMalloc((void **)&Layer6_Weights_GPU, sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer6_Mean_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer6_StanDev_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer6_Gamma_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer6_Beta_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);

    cudaMemcpy(Layer6_Weights_GPU, Layer6_Weights_CPU, sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Mean_GPU, Layer6_Mean_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_StanDev_GPU, Layer6_StanDev_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Gamma_GPU, Layer6_Gamma_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Beta_GPU, Layer6_Beta_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer6_Weights_CPU);
    free(Layer6_Mean_CPU);
    free(Layer6_StanDev_CPU);
    free(Layer6_Gamma_CPU);
    free(Layer6_Beta_CPU);

    dim3 gridSizeSixthLayer(128);
    dim3 blockSizeSixthLayerA(32, 32);
    executeSixthLayer_DSC_partA<<<gridSizeSixthLayer, blockSizeSixthLayerA>>>(Layer6_Neurons_GPU,
                                                                              Layer6_Weights_GPU,
                                                                              Layer7_Neurons_GPU,
                                                                              Layer6_Mean_GPU,
                                                                              Layer6_StanDev_GPU,
                                                                              Layer6_Gamma_GPU,
                                                                              Layer6_Beta_GPU);

    dim3 blockSizeSixthLayerB(32, 24);
    executeSixthLayer_DSC_partB<<<gridSizeSixthLayer, blockSizeSixthLayerB>>>(Layer6_Neurons_GPU,
                                                                              Layer6_Weights_GPU,
                                                                              Layer7_Neurons_GPU,
                                                                              Layer6_Mean_GPU,
                                                                              Layer6_StanDev_GPU,
                                                                              Layer6_Gamma_GPU,
                                                                              Layer6_Beta_GPU);

    dim3 blockSizeSixthLayerC(24, 32);
    executeSixthLayer_DSC_partC<<<gridSizeSixthLayer, blockSizeSixthLayerC>>>(Layer6_Neurons_GPU,
                                                                              Layer6_Weights_GPU,
                                                                              Layer7_Neurons_GPU,
                                                                              Layer6_Mean_GPU,
                                                                              Layer6_StanDev_GPU,
                                                                              Layer6_Gamma_GPU,
                                                                              Layer6_Beta_GPU);

    dim3 blockSizeSixthLayerD(24, 24);
    executeSixthLayer_DSC_partD<<<gridSizeSixthLayer, blockSizeSixthLayerD>>>(Layer6_Neurons_GPU,
                                                                              Layer6_Weights_GPU,
                                                                              Layer7_Neurons_GPU,
                                                                              Layer6_Mean_GPU,
                                                                              Layer6_StanDev_GPU,
                                                                              Layer6_Gamma_GPU,
                                                                              Layer6_Beta_GPU);

    cudaFree(Layer6_Weights_GPU);
    cudaFree(Layer6_Mean_GPU);
    cudaFree(Layer6_StanDev_GPU);
    cudaFree(Layer6_Gamma_GPU);
    cudaFree(Layer6_Beta_GPU);
}

void Read_SixthLayer_Data(double *Layer6_Weights_CPU,
                          double *Layer6_Mean_CPU,
                          double *Layer6_StanDev_CPU,
                          double *Layer6_Gamma_CPU,
                          double *Layer6_Beta_CPU)
{
    read_File("data/SixthLayer/weightsNorm.txt", Layer6_Weights_CPU);
    read_File("data/SixthLayer/Sixth_Layer_Mean.txt", Layer6_Mean_CPU);
    read_File("data/SixthLayer/Sixth_Layer_StanDev.txt", Layer6_StanDev_CPU);
    read_File("data/SixthLayer/Sixth_Layer_Gamma.txt", Layer6_Gamma_CPU);
    read_File("data/SixthLayer/Sixth_Layer_Beta.txt", Layer6_Beta_CPU);
}

void Execute_Seventh_Layer(
    double *Layer7_Neurons_GPU,
    double *Layer8_Neurons_GPU)
{
    double *Layer7_Weights_CPU = (double *)malloc(sizeof(double) * SEVENTH_LAYER_WEIGHT_SIZE);
    double *Layer7_Mean_CPU = (double *)malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);
    double *Layer7_StanDev_CPU = (double *)malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);
    double *Layer7_Gamma_CPU = (double *)malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);
    double *Layer7_Beta_CPU = (double *)malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);

    Read_SeventhLayer_Data(Layer7_Weights_CPU,
                           Layer7_Mean_CPU,
                           Layer7_StanDev_CPU,
                           Layer7_Gamma_CPU,
                           Layer7_Beta_CPU);

    double *Layer7_Weights_GPU,
        *Layer7_Mean_GPU,
        *Layer7_StanDev_GPU,
        *Layer7_Gamma_GPU,
        *Layer7_Beta_GPU;

    cudaMalloc((void **)&Layer7_Weights_GPU, sizeof(double) * SEVENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer7_Mean_GPU, sizeof(double) * SEVENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer7_StanDev_GPU, sizeof(double) * SEVENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer7_Gamma_GPU, sizeof(double) * SEVENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer7_Beta_GPU, sizeof(double) * SEVENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer7_Weights_GPU, Layer7_Weights_CPU, sizeof(double) * SEVENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer7_Mean_GPU, Layer7_Mean_CPU, sizeof(double) * SEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer7_StanDev_GPU, Layer7_StanDev_CPU, sizeof(double) * SEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer7_Gamma_GPU, Layer7_Gamma_CPU, sizeof(double) * SEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer7_Beta_GPU, Layer7_Beta_CPU, sizeof(double) * SEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer7_Weights_CPU);
    free(Layer7_Mean_CPU);
    free(Layer7_StanDev_CPU);
    free(Layer7_Gamma_CPU);
    free(Layer7_Beta_CPU);

    dim3 gridSizeSeventhLayer(128);
    dim3 blockSizeSeventhLayerA(32, 32);
    executeSeventhLayer_PSC_partA<<<gridSizeSeventhLayer, blockSizeSeventhLayerA>>>(Layer7_Neurons_GPU,
                                                                                    Layer7_Weights_GPU,
                                                                                    Layer8_Neurons_GPU,
                                                                                    Layer7_Mean_GPU,
                                                                                    Layer7_StanDev_GPU,
                                                                                    Layer7_Gamma_GPU,
                                                                                    Layer7_Beta_GPU);

    dim3 blockSizeSeventhLayerB(32, 24);
    executeSeventhLayer_PSC_partB<<<gridSizeSeventhLayer, blockSizeSeventhLayerB>>>(Layer7_Neurons_GPU,
                                                                                    Layer7_Weights_GPU,
                                                                                    Layer8_Neurons_GPU,
                                                                                    Layer7_Mean_GPU,
                                                                                    Layer7_StanDev_GPU,
                                                                                    Layer7_Gamma_GPU,
                                                                                    Layer7_Beta_GPU);

    dim3 blockSizeSeventhLayerC(24, 32);
    executeSeventhLayer_PSC_partC<<<gridSizeSeventhLayer, blockSizeSeventhLayerC>>>(Layer7_Neurons_GPU,
                                                                                    Layer7_Weights_GPU,
                                                                                    Layer8_Neurons_GPU,
                                                                                    Layer7_Mean_GPU,
                                                                                    Layer7_StanDev_GPU,
                                                                                    Layer7_Gamma_GPU,
                                                                                    Layer7_Beta_GPU);

    dim3 blockSizeSeventhLayerD(24, 24);
    executeSeventhLayer_PSC_partD<<<gridSizeSeventhLayer, blockSizeSeventhLayerD>>>(Layer7_Neurons_GPU,
                                                                                    Layer7_Weights_GPU,
                                                                                    Layer8_Neurons_GPU,
                                                                                    Layer7_Mean_GPU,
                                                                                    Layer7_StanDev_GPU,
                                                                                    Layer7_Gamma_GPU,
                                                                                    Layer7_Beta_GPU);

    cudaFree(Layer7_Weights_GPU);
    cudaFree(Layer7_Mean_GPU);
    cudaFree(Layer7_StanDev_GPU);
    cudaFree(Layer7_Gamma_GPU);
    cudaFree(Layer7_Beta_GPU);
}

void Read_SeventhLayer_Data(double *Layer7_Weights_CPU,
                            double *Layer7_Mean_CPU,
                            double *Layer7_StanDev_CPU,
                            double *Layer7_Gamma_CPU,
                            double *Layer7_Beta_CPU)
{
    read_File("data/SeventhLayer/weightsNorm.txt", Layer7_Weights_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_Mean.txt", Layer7_Mean_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_StanDev.txt", Layer7_StanDev_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_Gamma.txt", Layer7_Gamma_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_Beta.txt", Layer7_Beta_CPU);
}

void Execute_Eighth_Layer(
    double *Layer8_Neurons_GPU,
    double *Layer9_Neurons_GPU)
{
    double *Layer8_Weights_CPU = (double *)malloc(sizeof(double) * EIGHTH_LAYER_WEIGHT_SIZE);
    double *Layer8_Mean_CPU = (double *)malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);
    double *Layer8_StanDev_CPU = (double *)malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);
    double *Layer8_Gamma_CPU = (double *)malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);
    double *Layer8_Beta_CPU = (double *)malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);

    Read_EighthLayer_Data(Layer8_Weights_CPU,
                          Layer8_Mean_CPU,
                          Layer8_StanDev_CPU,
                          Layer8_Gamma_CPU,
                          Layer8_Beta_CPU);

    double *Layer8_Weights_GPU,
        *Layer8_Mean_GPU,
        *Layer8_StanDev_GPU,
        *Layer8_Gamma_GPU,
        *Layer8_Beta_GPU;

    cudaMalloc((void **)&Layer8_Weights_GPU, sizeof(double) * EIGHTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer8_Mean_GPU, sizeof(double) * EIGHTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer8_StanDev_GPU, sizeof(double) * EIGHTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer8_Gamma_GPU, sizeof(double) * EIGHTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer8_Beta_GPU, sizeof(double) * EIGHTH_LAYER_CHANNELS);

    cudaMemcpy(Layer8_Weights_GPU, Layer8_Weights_CPU, sizeof(double) * EIGHTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer8_Mean_GPU, Layer8_Mean_CPU, sizeof(double) * EIGHTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer8_StanDev_GPU, Layer8_StanDev_CPU, sizeof(double) * EIGHTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer8_Gamma_GPU, Layer8_Gamma_CPU, sizeof(double) * EIGHTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer8_Beta_GPU, Layer8_Beta_CPU, sizeof(double) * EIGHTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer8_Weights_CPU);
    free(Layer8_Mean_CPU);
    free(Layer8_StanDev_CPU);
    free(Layer8_Gamma_CPU);
    free(Layer8_Beta_CPU);

    dim3 gridSizeEighthLayer(128);
    dim3 blockSizeEighth(28, 28);
    executeEighthLayer_DSC<<<gridSizeEighthLayer, blockSizeEighth>>>(Layer8_Neurons_GPU,
                                                                     Layer8_Weights_GPU,
                                                                     Layer9_Neurons_GPU,
                                                                     Layer8_Mean_GPU,
                                                                     Layer8_StanDev_GPU,
                                                                     Layer8_Gamma_GPU,
                                                                     Layer8_Beta_GPU);

    cudaFree(Layer8_Weights_GPU);
    cudaFree(Layer8_Mean_GPU);
    cudaFree(Layer8_StanDev_GPU);
    cudaFree(Layer8_Gamma_GPU);
    cudaFree(Layer8_Beta_GPU);
}

void Read_EighthLayer_Data(double *Layer8_Weights_CPU,
                           double *Layer8_Mean_CPU,
                           double *Layer8_StanDev_CPU,
                           double *Layer8_Gamma_CPU,
                           double *Layer8_Beta_CPU)
{
    read_File("data/EighthLayer/weightsNorm.txt", Layer8_Weights_CPU);
    read_File("data/EighthLayer/Eighth_Layer_Mean.txt", Layer8_Mean_CPU);
    read_File("data/EighthLayer/Eighth_Layer_StanDev.txt", Layer8_StanDev_CPU);
    read_File("data/EighthLayer/Eighth_Layer_Gamma.txt", Layer8_Gamma_CPU);
    read_File("data/EighthLayer/Eighth_Layer_Beta.txt", Layer8_Beta_CPU);
}

void Execute_Ninth_Layer(
    double *Layer9_Neurons_GPU,
    double *Layer10_Neurons_GPU)
{
    double *Layer9_Weights_CPU = (double *)malloc(sizeof(double) * NINTH_LAYER_WEIGHT_SIZE);
    double *Layer9_Mean_CPU = (double *)malloc(sizeof(double) * NINTH_LAYER_CHANNELS);
    double *Layer9_StanDev_CPU = (double *)malloc(sizeof(double) * NINTH_LAYER_CHANNELS);
    double *Layer9_Gamma_CPU = (double *)malloc(sizeof(double) * NINTH_LAYER_CHANNELS);
    double *Layer9_Beta_CPU = (double *)malloc(sizeof(double) * NINTH_LAYER_CHANNELS);

    Read_NinthLayer_Data(Layer9_Weights_CPU,
                         Layer9_Mean_CPU,
                         Layer9_StanDev_CPU,
                         Layer9_Gamma_CPU,
                         Layer9_Beta_CPU);

    double *Layer9_Weights_GPU,
        *Layer9_Mean_GPU,
        *Layer9_StanDev_GPU,
        *Layer9_Gamma_GPU,
        *Layer9_Beta_GPU;

    cudaMalloc((void **)&Layer9_Weights_GPU, sizeof(double) * NINTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer9_Mean_GPU, sizeof(double) * NINTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer9_StanDev_GPU, sizeof(double) * NINTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer9_Gamma_GPU, sizeof(double) * NINTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer9_Beta_GPU, sizeof(double) * NINTH_LAYER_CHANNELS);

    cudaMemcpy(Layer9_Weights_GPU, Layer9_Weights_CPU, sizeof(double) * NINTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer9_Mean_GPU, Layer9_Mean_CPU, sizeof(double) * NINTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer9_StanDev_GPU, Layer9_StanDev_CPU, sizeof(double) * NINTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer9_Gamma_GPU, Layer9_Gamma_CPU, sizeof(double) * NINTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer9_Beta_GPU, Layer9_Beta_CPU, sizeof(double) * NINTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer9_Weights_CPU);
    free(Layer9_Mean_CPU);
    free(Layer9_StanDev_CPU);
    free(Layer9_Gamma_CPU);
    free(Layer9_Beta_CPU);

    dim3 gridSizeNinthLayer(256);
    dim3 blockSizeNinth(28, 28);
    executeNinthLayer_PSC<<<gridSizeNinthLayer, blockSizeNinth>>>(Layer9_Neurons_GPU,
                                                                  Layer9_Weights_GPU,
                                                                  Layer10_Neurons_GPU,
                                                                  Layer9_Mean_GPU,
                                                                  Layer9_StanDev_GPU,
                                                                  Layer9_Gamma_GPU,
                                                                  Layer9_Beta_GPU);

    cudaFree(Layer9_Weights_GPU);
    cudaFree(Layer9_Mean_GPU);
    cudaFree(Layer9_StanDev_GPU);
    cudaFree(Layer9_Gamma_GPU);
    cudaFree(Layer9_Beta_GPU);
}

void Read_NinthLayer_Data(double *Layer9_Weights_CPU,
                          double *Layer9_Mean_CPU,
                          double *Layer9_StanDev_CPU,
                          double *Layer9_Gamma_CPU,
                          double *Layer9_Beta_CPU)
{
    read_File("data/NinthLayer/weightsNorm.txt", Layer9_Weights_CPU);
    read_File("data/NinthLayer/Ninth_Layer_Mean.txt", Layer9_Mean_CPU);
    read_File("data/NinthLayer/Ninth_Layer_StanDev.txt", Layer9_StanDev_CPU);
    read_File("data/NinthLayer/Ninth_Layer_Gamma.txt", Layer9_Gamma_CPU);
    read_File("data/NinthLayer/Ninth_Layer_Beta.txt", Layer9_Beta_CPU);
}

void Execute_Tenth_Layer(
    double *Layer10_Neurons_GPU,
    double *Layer11_Neurons_GPU)
{
    double *Layer10_Weights_CPU = (double *)malloc(sizeof(double) * TENTH_LAYER_WEIGHT_SIZE);
    double *Layer10_Mean_CPU = (double *)malloc(sizeof(double) * TENTH_LAYER_CHANNELS);
    double *Layer10_StanDev_CPU = (double *)malloc(sizeof(double) * TENTH_LAYER_CHANNELS);
    double *Layer10_Gamma_CPU = (double *)malloc(sizeof(double) * TENTH_LAYER_CHANNELS);
    double *Layer10_Beta_CPU = (double *)malloc(sizeof(double) * TENTH_LAYER_CHANNELS);

    Read_TenthLayer_Data(Layer10_Weights_CPU,
                         Layer10_Mean_CPU,
                         Layer10_StanDev_CPU,
                         Layer10_Gamma_CPU,
                         Layer10_Beta_CPU);

    double *Layer10_Weights_GPU,
        *Layer10_Mean_GPU,
        *Layer10_StanDev_GPU,
        *Layer10_Gamma_GPU,
        *Layer10_Beta_GPU;

    cudaMalloc((void **)&Layer10_Weights_GPU, sizeof(double) * TENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer10_Mean_GPU, sizeof(double) * TENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer10_StanDev_GPU, sizeof(double) * TENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer10_Gamma_GPU, sizeof(double) * TENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer10_Beta_GPU, sizeof(double) * TENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer10_Weights_GPU, Layer10_Weights_CPU, sizeof(double) * TENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer10_Mean_GPU, Layer10_Mean_CPU, sizeof(double) * TENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer10_StanDev_GPU, Layer10_StanDev_CPU, sizeof(double) * TENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer10_Gamma_GPU, Layer10_Gamma_CPU, sizeof(double) * TENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer10_Beta_GPU, Layer10_Beta_CPU, sizeof(double) * TENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer10_Weights_CPU);
    free(Layer10_Mean_CPU);
    free(Layer10_StanDev_CPU);
    free(Layer10_Gamma_CPU);
    free(Layer10_Beta_CPU);

    dim3 gridSizeTenthLayer(256);
    dim3 blockSizeTenth(28, 28);
    executeTenthLayer_DSC<<<gridSizeTenthLayer, blockSizeTenth>>>(Layer10_Neurons_GPU,
                                                                  Layer10_Weights_GPU,
                                                                  Layer11_Neurons_GPU,
                                                                  Layer10_Mean_GPU,
                                                                  Layer10_StanDev_GPU,
                                                                  Layer10_Gamma_GPU,
                                                                  Layer10_Beta_GPU);

    cudaFree(Layer10_Weights_GPU);
    cudaFree(Layer10_Mean_GPU);
    cudaFree(Layer10_StanDev_GPU);
    cudaFree(Layer10_Gamma_GPU);
    cudaFree(Layer10_Beta_GPU);
}

void Read_TenthLayer_Data(double *Layer10_Weights_CPU,
                          double *Layer10_Mean_CPU,
                          double *Layer10_StanDev_CPU,
                          double *Layer10_Gamma_CPU,
                          double *Layer10_Beta_CPU)
{
    read_File("data/TenthLayer/weightsNorm.txt", Layer10_Weights_CPU);
    read_File("data/TenthLayer/Tenth_Layer_Mean.txt", Layer10_Mean_CPU);
    read_File("data/TenthLayer/Tenth_Layer_StanDev.txt", Layer10_StanDev_CPU);
    read_File("data/TenthLayer/Tenth_Layer_Gamma.txt", Layer10_Gamma_CPU);
    read_File("data/TenthLayer/Tenth_Layer_Beta.txt", Layer10_Beta_CPU);
}

void Execute_Eleventh_Layer(
    double *Layer11_Neurons_GPU,
    double *Layer12_Neurons_GPU)
{
    double *Layer11_Weights_CPU = (double *)malloc(sizeof(double) * ELEVENTH_LAYER_WEIGHT_SIZE);
    double *Layer11_Mean_CPU = (double *)malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    double *Layer11_StanDev_CPU = (double *)malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    double *Layer11_Gamma_CPU = (double *)malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    double *Layer11_Beta_CPU = (double *)malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);

    Read_EleventhLayer_Data(Layer11_Weights_CPU,
                            Layer11_Mean_CPU,
                            Layer11_StanDev_CPU,
                            Layer11_Gamma_CPU,
                            Layer11_Beta_CPU);

    double *Layer11_Weights_GPU,
        *Layer11_Mean_GPU,
        *Layer11_StanDev_GPU,
        *Layer11_Gamma_GPU,
        *Layer11_Beta_GPU;

    cudaMalloc((void **)&Layer11_Weights_GPU, sizeof(double) * ELEVENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer11_Mean_GPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer11_StanDev_GPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer11_Gamma_GPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer11_Beta_GPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer11_Weights_GPU, Layer11_Weights_CPU, sizeof(double) * ELEVENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer11_Mean_GPU, Layer11_Mean_CPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer11_StanDev_GPU, Layer11_StanDev_CPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer11_Gamma_GPU, Layer11_Gamma_CPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer11_Beta_GPU, Layer11_Beta_CPU, sizeof(double) * ELEVENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer11_Weights_CPU);
    free(Layer11_Mean_CPU);
    free(Layer11_StanDev_CPU);
    free(Layer11_Gamma_CPU);
    free(Layer11_Beta_CPU);

    dim3 gridSizeEleventhLayer(256);
    dim3 blockSizeEleventh(28, 28);
    executeEleventhLayer_PSC<<<gridSizeEleventhLayer, blockSizeEleventh>>>(Layer11_Neurons_GPU,
                                                                           Layer11_Weights_GPU,
                                                                           Layer12_Neurons_GPU,
                                                                           Layer11_Mean_GPU,
                                                                           Layer11_StanDev_GPU,
                                                                           Layer11_Gamma_GPU,
                                                                           Layer11_Beta_GPU);

    cudaFree(Layer11_Weights_GPU);
    cudaFree(Layer11_Mean_GPU);
    cudaFree(Layer11_StanDev_GPU);
    cudaFree(Layer11_Gamma_GPU);
    cudaFree(Layer11_Beta_GPU);
}

void Read_EleventhLayer_Data(double *Layer11_Weights_CPU,
                             double *Layer11_Mean_CPU,
                             double *Layer11_StanDev_CPU,
                             double *Layer11_Gamma_CPU,
                             double *Layer11_Beta_CPU)
{
    read_File("data/EleventhLayer/weightsNorm.txt", Layer11_Weights_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_Mean.txt", Layer11_Mean_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_StanDev.txt", Layer11_StanDev_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_Gamma.txt", Layer11_Gamma_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_Beta.txt", Layer11_Beta_CPU);
}

void Execute_Twelveth_Layer(
    double *Layer12_Neurons_GPU,
    double *Layer13_Neurons_GPU)
{
    double *Layer12_Weights_CPU = (double *)malloc(sizeof(double) * TWELFTH_LAYER_WEIGHT_SIZE);
    double *Layer12_Mean_CPU = (double *)malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);
    double *Layer12_StanDev_CPU = (double *)malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);
    double *Layer12_Gamma_CPU = (double *)malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);
    double *Layer12_Beta_CPU = (double *)malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);

    Read_TwelvethLayer_Data(Layer12_Weights_CPU,
                            Layer12_Mean_CPU,
                            Layer12_StanDev_CPU,
                            Layer12_Gamma_CPU,
                            Layer12_Beta_CPU);

    double *Layer12_Weights_GPU,
        *Layer12_Mean_GPU,
        *Layer12_StanDev_GPU,
        *Layer12_Gamma_GPU,
        *Layer12_Beta_GPU;

    cudaMalloc((void **)&Layer12_Weights_GPU, sizeof(double) * TWELFTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer12_Mean_GPU, sizeof(double) * TWELFTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer12_StanDev_GPU, sizeof(double) * TWELFTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer12_Gamma_GPU, sizeof(double) * TWELFTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer12_Beta_GPU, sizeof(double) * TWELFTH_LAYER_CHANNELS);

    cudaMemcpy(Layer12_Weights_GPU, Layer12_Weights_CPU, sizeof(double) * TWELFTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer12_Mean_GPU, Layer12_Mean_CPU, sizeof(double) * TWELFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer12_StanDev_GPU, Layer12_StanDev_CPU, sizeof(double) * TWELFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer12_Gamma_GPU, Layer12_Gamma_CPU, sizeof(double) * TWELFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer12_Beta_GPU, Layer12_Beta_CPU, sizeof(double) * TWELFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer12_Weights_CPU);
    free(Layer12_Mean_CPU);
    free(Layer12_StanDev_CPU);
    free(Layer12_Gamma_CPU);
    free(Layer12_Beta_CPU);

    dim3 gridSizeTwelvethLayer(256);
    dim3 blockSizeTwelveth(14, 14);
    executeTwelfthLayer_DSC<<<gridSizeTwelvethLayer, blockSizeTwelveth>>>(Layer12_Neurons_GPU,
                                                                          Layer12_Weights_GPU,
                                                                          Layer13_Neurons_GPU,
                                                                          Layer12_Mean_GPU,
                                                                          Layer12_StanDev_GPU,
                                                                          Layer12_Gamma_GPU,
                                                                          Layer12_Beta_GPU);

    cudaFree(Layer12_Weights_GPU);
    cudaFree(Layer12_Mean_GPU);
    cudaFree(Layer12_StanDev_GPU);
    cudaFree(Layer12_Gamma_GPU);
    cudaFree(Layer12_Beta_GPU);
}

void Read_TwelvethLayer_Data(double *Layer12_Weights_CPU,
                             double *Layer12_Mean_CPU,
                             double *Layer12_StanDev_CPU,
                             double *Layer12_Gamma_CPU,
                             double *Layer12_Beta_CPU)
{
    read_File("data/TwelvethLayer/weightsNorm.txt", Layer12_Weights_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_Mean.txt", Layer12_Mean_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_StanDev.txt", Layer12_StanDev_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_Gamma.txt", Layer12_Gamma_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_Beta.txt", Layer12_Beta_CPU);
}

void Execute_Thirteenth_Layer(
    double *Layer13_Neurons_GPU,
    double *Layer14_Neurons_GPU)
{
    double *Layer13_Weights_CPU = (double *)malloc(sizeof(double) * THIRTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer13_Mean_CPU = (double *)malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    double *Layer13_StanDev_CPU = (double *)malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    double *Layer13_Gamma_CPU = (double *)malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    double *Layer13_Beta_CPU = (double *)malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);

    Read_ThirteenthLayer_Data(Layer13_Weights_CPU,
                              Layer13_Mean_CPU,
                              Layer13_StanDev_CPU,
                              Layer13_Gamma_CPU,
                              Layer13_Beta_CPU);

    double *Layer13_Weights_GPU,
        *Layer13_Mean_GPU,
        *Layer13_StanDev_GPU,
        *Layer13_Gamma_GPU,
        *Layer13_Beta_GPU;

    cudaMalloc((void **)&Layer13_Weights_GPU, sizeof(double) * THIRTEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer13_Mean_GPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer13_StanDev_GPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer13_Gamma_GPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer13_Beta_GPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer13_Weights_GPU, Layer13_Weights_CPU, sizeof(double) * THIRTEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer13_Mean_GPU, Layer13_Mean_CPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer13_StanDev_GPU, Layer13_StanDev_CPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer13_Gamma_GPU, Layer13_Gamma_CPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer13_Beta_GPU, Layer13_Beta_CPU, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer13_Weights_CPU);
    free(Layer13_Mean_CPU);
    free(Layer13_StanDev_CPU);
    free(Layer13_Gamma_CPU);
    free(Layer13_Beta_CPU);

    dim3 gridSizeThirteenthLayer(512);
    dim3 blockSizeThirteenth(14, 14);
    executeThirteenthLayer_PSC<<<gridSizeThirteenthLayer, blockSizeThirteenth>>>(Layer13_Neurons_GPU,
                                                                                 Layer13_Weights_GPU,
                                                                                 Layer14_Neurons_GPU,
                                                                                 Layer13_Mean_GPU,
                                                                                 Layer13_StanDev_GPU,
                                                                                 Layer13_Gamma_GPU,
                                                                                 Layer13_Beta_GPU);

    cudaFree(Layer13_Weights_GPU);
    cudaFree(Layer13_Mean_GPU);
    cudaFree(Layer13_StanDev_GPU);
    cudaFree(Layer13_Gamma_GPU);
    cudaFree(Layer13_Beta_GPU);
}

void Read_ThirteenthLayer_Data(double *Layer13_Weights_CPU,
                               double *Layer13_Mean_CPU,
                               double *Layer13_StanDev_CPU,
                               double *Layer13_Gamma_CPU,
                               double *Layer13_Beta_CPU)
{
    read_File("data/ThirteenthLayer/weightsNorm.txt", Layer13_Weights_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_Mean.txt", Layer13_Mean_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_StanDev.txt", Layer13_StanDev_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_Gamma.txt", Layer13_Gamma_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_Beta.txt", Layer13_Beta_CPU);
}

void Execute_Fourteenth_Layer(
    double *Layer14_Neurons_GPU,
    double *Layer15_Neurons_GPU)
{
    double *Layer14_Weights_CPU = (double *)malloc(sizeof(double) * FOURTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer14_Mean_CPU = (double *)malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    double *Layer14_StanDev_CPU = (double *)malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    double *Layer14_Gamma_CPU = (double *)malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    double *Layer14_Beta_CPU = (double *)malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);

    Read_FourteenthLayer_Data(Layer14_Weights_CPU,
                              Layer14_Mean_CPU,
                              Layer14_StanDev_CPU,
                              Layer14_Gamma_CPU,
                              Layer14_Beta_CPU);

    double *Layer14_Weights_GPU,
        *Layer14_Mean_GPU,
        *Layer14_StanDev_GPU,
        *Layer14_Gamma_GPU,
        *Layer14_Beta_GPU;

    cudaMalloc((void **)&Layer14_Weights_GPU, sizeof(double) * FOURTEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer14_Mean_GPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer14_StanDev_GPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer14_Gamma_GPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer14_Beta_GPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer14_Weights_GPU, Layer14_Weights_CPU, sizeof(double) * FOURTEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer14_Mean_GPU, Layer14_Mean_CPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer14_StanDev_GPU, Layer14_StanDev_CPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer14_Gamma_GPU, Layer14_Gamma_CPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer14_Beta_GPU, Layer14_Beta_CPU, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer14_Weights_CPU);
    free(Layer14_Mean_CPU);
    free(Layer14_StanDev_CPU);
    free(Layer14_Gamma_CPU);
    free(Layer14_Beta_CPU);

    dim3 gridSizeFourteenthLayer(512);
    dim3 blockSizeFourteenth(14, 14);
    executeFourteenthLayer_DSC<<<gridSizeFourteenthLayer, blockSizeFourteenth>>>(Layer14_Neurons_GPU,
                                                                                 Layer14_Weights_GPU,
                                                                                 Layer15_Neurons_GPU,
                                                                                 Layer14_Mean_GPU,
                                                                                 Layer14_StanDev_GPU,
                                                                                 Layer14_Gamma_GPU,
                                                                                 Layer14_Beta_GPU);

    cudaFree(Layer14_Weights_GPU);
    cudaFree(Layer14_Mean_GPU);
    cudaFree(Layer14_StanDev_GPU);
    cudaFree(Layer14_Gamma_GPU);
    cudaFree(Layer14_Beta_GPU);
}

void Read_FourteenthLayer_Data(double *Layer14_Weights_CPU,
                               double *Layer14_Mean_CPU,
                               double *Layer14_StanDev_CPU,
                               double *Layer14_Gamma_CPU,
                               double *Layer14_Beta_CPU)
{
    read_File("data/FourteenthLayer/weightsNorm.txt", Layer14_Weights_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_Mean.txt", Layer14_Mean_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_StanDev.txt", Layer14_StanDev_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_Gamma.txt", Layer14_Gamma_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_Beta.txt", Layer14_Beta_CPU);
}

void Execute_Fifteenth_Layer(
    double *Layer15_Neurons_GPU,
    double *Layer16_Neurons_GPU)
{
    double *Layer15_Weights_CPU = (double *)malloc(sizeof(double) * FIFTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer15_Mean_CPU = (double *)malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    double *Layer15_StanDev_CPU = (double *)malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    double *Layer15_Gamma_CPU = (double *)malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    double *Layer15_Beta_CPU = (double *)malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);

    Read_FifteenthLayer_Data(Layer15_Weights_CPU,
                             Layer15_Mean_CPU,
                             Layer15_StanDev_CPU,
                             Layer15_Gamma_CPU,
                             Layer15_Beta_CPU);

    double *Layer15_Weights_GPU,
        *Layer15_Mean_GPU,
        *Layer15_StanDev_GPU,
        *Layer15_Gamma_GPU,
        *Layer15_Beta_GPU;

    cudaMalloc((void **)&Layer15_Weights_GPU, sizeof(double) * FIFTEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer15_Mean_GPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer15_StanDev_GPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer15_Gamma_GPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer15_Beta_GPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer15_Weights_GPU, Layer15_Weights_CPU, sizeof(double) * FIFTEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer15_Mean_GPU, Layer15_Mean_CPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer15_StanDev_GPU, Layer15_StanDev_CPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer15_Gamma_GPU, Layer15_Gamma_CPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer15_Beta_GPU, Layer15_Beta_CPU, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer15_Weights_CPU);
    free(Layer15_Mean_CPU);
    free(Layer15_StanDev_CPU);
    free(Layer15_Gamma_CPU);
    free(Layer15_Beta_CPU);

    dim3 gridSizeFifteenthLayer(512);
    dim3 blockSizeFifteenth(14, 14);
    executeFifteenthLayer_PSC<<<gridSizeFifteenthLayer, blockSizeFifteenth>>>(Layer15_Neurons_GPU,
                                                                              Layer15_Weights_GPU,
                                                                              Layer16_Neurons_GPU,
                                                                              Layer15_Mean_GPU,
                                                                              Layer15_StanDev_GPU,
                                                                              Layer15_Gamma_GPU,
                                                                              Layer15_Beta_GPU);

    cudaFree(Layer15_Weights_GPU);
    cudaFree(Layer15_Mean_GPU);
    cudaFree(Layer15_StanDev_GPU);
    cudaFree(Layer15_Gamma_GPU);
    cudaFree(Layer15_Beta_GPU);
}

void Read_FifteenthLayer_Data(double *Layer15_Weights_CPU,
                              double *Layer15_Mean_CPU,
                              double *Layer15_StanDev_CPU,
                              double *Layer15_Gamma_CPU,
                              double *Layer15_Beta_CPU)
{
    read_File("data/FifteenthLayer/weightsNorm.txt", Layer15_Weights_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_Mean.txt", Layer15_Mean_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_StanDev.txt", Layer15_StanDev_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_Gamma.txt", Layer15_Gamma_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_Beta.txt", Layer15_Beta_CPU);
}

void Execute_Sixteenth_Layer(
    double *Layer16_Neurons_GPU,
    double *Layer17_Neurons_GPU)
{
    double *Layer16_Weights_CPU = (double *)malloc(sizeof(double) * SIXTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer16_Mean_CPU = (double *)malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    double *Layer16_StanDev_CPU = (double *)malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    double *Layer16_Gamma_CPU = (double *)malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    double *Layer16_Beta_CPU = (double *)malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);

    Read_SixteenthLayer_Data(Layer16_Weights_CPU,
                             Layer16_Mean_CPU,
                             Layer16_StanDev_CPU,
                             Layer16_Gamma_CPU,
                             Layer16_Beta_CPU);

    double *Layer16_Weights_GPU,
        *Layer16_Mean_GPU,
        *Layer16_StanDev_GPU,
        *Layer16_Gamma_GPU,
        *Layer16_Beta_GPU;

    cudaMalloc((void **)&Layer16_Weights_GPU, sizeof(double) * SIXTEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer16_Mean_GPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer16_StanDev_GPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer16_Gamma_GPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer16_Beta_GPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer16_Weights_GPU, Layer16_Weights_CPU, sizeof(double) * SIXTEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer16_Mean_GPU, Layer16_Mean_CPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer16_StanDev_GPU, Layer16_StanDev_CPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer16_Gamma_GPU, Layer16_Gamma_CPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer16_Beta_GPU, Layer16_Beta_CPU, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer16_Weights_CPU);
    free(Layer16_Mean_CPU);
    free(Layer16_StanDev_CPU);
    free(Layer16_Gamma_CPU);
    free(Layer16_Beta_CPU);

    dim3 gridSizeSixteenthLayer(512);
    dim3 blockSizeSixteenth(14, 14);
    executeSixteenthLayer_DSC<<<gridSizeSixteenthLayer, blockSizeSixteenth>>>(Layer16_Neurons_GPU,
                                                                              Layer16_Weights_GPU,
                                                                              Layer17_Neurons_GPU,
                                                                              Layer16_Mean_GPU,
                                                                              Layer16_StanDev_GPU,
                                                                              Layer16_Gamma_GPU,
                                                                              Layer16_Beta_GPU);

    cudaFree(Layer16_Weights_GPU);
    cudaFree(Layer16_Mean_GPU);
    cudaFree(Layer16_StanDev_GPU);
    cudaFree(Layer16_Gamma_GPU);
    cudaFree(Layer16_Beta_GPU);
}

void Read_SixteenthLayer_Data(double *Layer16_Weights_CPU,
                              double *Layer16_Mean_CPU,
                              double *Layer16_StanDev_CPU,
                              double *Layer16_Gamma_CPU,
                              double *Layer16_Beta_CPU)
{
    read_File("data/SixteenthLayer/weightsNorm.txt", Layer16_Weights_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_Mean.txt", Layer16_Mean_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_StanDev.txt", Layer16_StanDev_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_Gamma.txt", Layer16_Gamma_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_Beta.txt", Layer16_Beta_CPU);
}

void Execute_Seventeenth_Layer(
    double *Layer17_Neurons_GPU,
    double *Layer18_Neurons_GPU)
{
    double *Layer17_Weights_CPU = (double *)malloc(sizeof(double) * SEVENTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer17_Mean_CPU = (double *)malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    double *Layer17_StanDev_CPU = (double *)malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    double *Layer17_Gamma_CPU = (double *)malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    double *Layer17_Beta_CPU = (double *)malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);

    Read_SeventeenthLayer_Data(Layer17_Weights_CPU,
                               Layer17_Mean_CPU,
                               Layer17_StanDev_CPU,
                               Layer17_Gamma_CPU,
                               Layer17_Beta_CPU);

    double *Layer17_Weights_GPU,
        *Layer17_Mean_GPU,
        *Layer17_StanDev_GPU,
        *Layer17_Gamma_GPU,
        *Layer17_Beta_GPU;

    cudaMalloc((void **)&Layer17_Weights_GPU, sizeof(double) * SEVENTEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer17_Mean_GPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer17_StanDev_GPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer17_Gamma_GPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer17_Beta_GPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer17_Weights_GPU, Layer17_Weights_CPU, sizeof(double) * SEVENTEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer17_Mean_GPU, Layer17_Mean_CPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer17_StanDev_GPU, Layer17_StanDev_CPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer17_Gamma_GPU, Layer17_Gamma_CPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer17_Beta_GPU, Layer17_Beta_CPU, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer17_Weights_CPU);
    free(Layer17_Mean_CPU);
    free(Layer17_StanDev_CPU);
    free(Layer17_Gamma_CPU);
    free(Layer17_Beta_CPU);

    dim3 gridSizeSeventeenthLayer(512);
    dim3 blockSizeSeventeenth(14, 14);
    executeSeventeenthLayer_PSC<<<gridSizeSeventeenthLayer, blockSizeSeventeenth>>>(Layer17_Neurons_GPU,
                                                                                    Layer17_Weights_GPU,
                                                                                    Layer18_Neurons_GPU,
                                                                                    Layer17_Mean_GPU,
                                                                                    Layer17_StanDev_GPU,
                                                                                    Layer17_Gamma_GPU,
                                                                                    Layer17_Beta_GPU);

    cudaFree(Layer17_Weights_GPU);
    cudaFree(Layer17_Mean_GPU);
    cudaFree(Layer17_StanDev_GPU);
    cudaFree(Layer17_Gamma_GPU);
    cudaFree(Layer17_Beta_GPU);
}

void Read_SeventeenthLayer_Data(double *Layer17_Weights_CPU,
                                double *Layer17_Mean_CPU,
                                double *Layer17_StanDev_CPU,
                                double *Layer17_Gamma_CPU,
                                double *Layer17_Beta_CPU)
{
    read_File("data/SeventeenthLayer/weightsNorm.txt", Layer17_Weights_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_Mean.txt", Layer17_Mean_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_StanDev.txt", Layer17_StanDev_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_Gamma.txt", Layer17_Gamma_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_Beta.txt", Layer17_Beta_CPU);
}

void Execute_Eighteenth_Layer(
    double *Layer18_Neurons_GPU,
    double *Layer19_Neurons_GPU)
{
    double *Layer18_Weights_CPU = (double *)malloc(sizeof(double) * EIGHTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer18_Mean_CPU = (double *)malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    double *Layer18_StanDev_CPU = (double *)malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    double *Layer18_Gamma_CPU = (double *)malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    double *Layer18_Beta_CPU = (double *)malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);

    Read_EighteenthLayer_Data(Layer18_Weights_CPU,
                              Layer18_Mean_CPU,
                              Layer18_StanDev_CPU,
                              Layer18_Gamma_CPU,
                              Layer18_Beta_CPU);

    double *Layer18_Weights_GPU,
        *Layer18_Mean_GPU,
        *Layer18_StanDev_GPU,
        *Layer18_Gamma_GPU,
        *Layer18_Beta_GPU;

    cudaMalloc((void **)&Layer18_Weights_GPU, sizeof(double) * EIGHTEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer18_Mean_GPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer18_StanDev_GPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer18_Gamma_GPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer18_Beta_GPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer18_Weights_GPU, Layer18_Weights_CPU, sizeof(double) * EIGHTEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer18_Mean_GPU, Layer18_Mean_CPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer18_StanDev_GPU, Layer18_StanDev_CPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer18_Gamma_GPU, Layer18_Gamma_CPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer18_Beta_GPU, Layer18_Beta_CPU, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer18_Weights_CPU);
    free(Layer18_Mean_CPU);
    free(Layer18_StanDev_CPU);
    free(Layer18_Gamma_CPU);
    free(Layer18_Beta_CPU);

    dim3 gridSizeEighteenthLayer(512);
    dim3 blockSizeEighteenth(14, 14);
    executeEighteenthLayer_DSC<<<gridSizeEighteenthLayer, blockSizeEighteenth>>>(Layer18_Neurons_GPU,
                                                                                 Layer18_Weights_GPU,
                                                                                 Layer19_Neurons_GPU,
                                                                                 Layer18_Mean_GPU,
                                                                                 Layer18_StanDev_GPU,
                                                                                 Layer18_Gamma_GPU,
                                                                                 Layer18_Beta_GPU);

    cudaFree(Layer18_Weights_GPU);
    cudaFree(Layer18_Mean_GPU);
    cudaFree(Layer18_StanDev_GPU);
    cudaFree(Layer18_Gamma_GPU);
    cudaFree(Layer18_Beta_GPU);
}

void Read_EighteenthLayer_Data(double *Layer18_Weights_CPU,
                               double *Layer18_Mean_CPU,
                               double *Layer18_StanDev_CPU,
                               double *Layer18_Gamma_CPU,
                               double *Layer18_Beta_CPU)
{
    read_File("data/EighteenthLayer/weightsNorm.txt", Layer18_Weights_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_Mean.txt", Layer18_Mean_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_StanDev.txt", Layer18_StanDev_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_Gamma.txt", Layer18_Gamma_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_Beta.txt", Layer18_Beta_CPU);
}

void Execute_Nineteenth_Layer(
    double *Layer19_Neurons_GPU,
    double *Layer20_Neurons_GPU)
{
    double *Layer19_Weights_CPU = (double *)malloc(sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE);
    double *Layer19_Mean_CPU = (double *)malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer19_StanDev_CPU = (double *)malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer19_Gamma_CPU = (double *)malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer19_Beta_CPU = (double *)malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);

    Read_NineteenthLayer_Data(Layer19_Weights_CPU,
                              Layer19_Mean_CPU,
                              Layer19_StanDev_CPU,
                              Layer19_Gamma_CPU,
                              Layer19_Beta_CPU);

    double *Layer19_Weights_GPU,
        *Layer19_Mean_GPU,
        *Layer19_StanDev_GPU,
        *Layer19_Gamma_GPU,
        *Layer19_Beta_GPU;

    cudaMalloc((void **)&Layer19_Weights_GPU, sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer19_Mean_GPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer19_StanDev_GPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer19_Gamma_GPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer19_Beta_GPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS);

    cudaMemcpy(Layer19_Weights_GPU, Layer19_Weights_CPU, sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer19_Mean_GPU, Layer19_Mean_CPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer19_StanDev_GPU, Layer19_StanDev_CPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer19_Gamma_GPU, Layer19_Gamma_CPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer19_Beta_GPU, Layer19_Beta_CPU, sizeof(double) * NINETEENTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer19_Weights_CPU);
    free(Layer19_Mean_CPU);
    free(Layer19_StanDev_CPU);
    free(Layer19_Gamma_CPU);
    free(Layer19_Beta_CPU);

    dim3 gridSizeNineteenthLayer(512);
    dim3 blockSizeNineteenth(14, 14);
    executeNineteenthLayer_PSC<<<gridSizeNineteenthLayer, blockSizeNineteenth>>>(Layer19_Neurons_GPU,
                                                                                 Layer19_Weights_GPU,
                                                                                 Layer20_Neurons_GPU,
                                                                                 Layer19_Mean_GPU,
                                                                                 Layer19_StanDev_GPU,
                                                                                 Layer19_Gamma_GPU,
                                                                                 Layer19_Beta_GPU);

    cudaFree(Layer19_Weights_GPU);
    cudaFree(Layer19_Mean_GPU);
    cudaFree(Layer19_StanDev_GPU);
    cudaFree(Layer19_Gamma_GPU);
    cudaFree(Layer19_Beta_GPU);
}

void Read_NineteenthLayer_Data(double *Layer19_Weights_CPU,
                               double *Layer19_Mean_CPU,
                               double *Layer19_StanDev_CPU,
                               double *Layer19_Gamma_CPU,
                               double *Layer19_Beta_CPU)
{
    read_File("data/NineteenthLayer/weightsNorm.txt", Layer19_Weights_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_Mean.txt", Layer19_Mean_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_StanDev.txt", Layer19_StanDev_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_Gamma.txt", Layer19_Gamma_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_Beta.txt", Layer19_Beta_CPU);
}

void Execute_Twenty_Layer(
    double *Layer20_Neurons_GPU,
    double *Layer21_Neurons_GPU)
{
    double *Layer20_Weights_CPU = (double *)malloc(sizeof(double) * TWENTY_LAYER_WEIGHT_SIZE);
    double *Layer20_Mean_CPU = (double *)malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);
    double *Layer20_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);
    double *Layer20_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);
    double *Layer20_Beta_CPU = (double *)malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);

    Read_TwentyLayer_Data(Layer20_Weights_CPU,
                          Layer20_Mean_CPU,
                          Layer20_StanDev_CPU,
                          Layer20_Gamma_CPU,
                          Layer20_Beta_CPU);

    double *Layer20_Weights_GPU,
        *Layer20_Mean_GPU,
        *Layer20_StanDev_GPU,
        *Layer20_Gamma_GPU,
        *Layer20_Beta_GPU;

    cudaMalloc((void **)&Layer20_Weights_GPU, sizeof(double) * TWENTY_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer20_Mean_GPU, sizeof(double) * TWENTY_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer20_StanDev_GPU, sizeof(double) * TWENTY_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer20_Gamma_GPU, sizeof(double) * TWENTY_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer20_Beta_GPU, sizeof(double) * TWENTY_LAYER_CHANNELS);

    cudaMemcpy(Layer20_Weights_GPU, Layer20_Weights_CPU, sizeof(double) * TWENTY_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer20_Mean_GPU, Layer20_Mean_CPU, sizeof(double) * TWENTY_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer20_StanDev_GPU, Layer20_StanDev_CPU, sizeof(double) * TWENTY_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer20_Gamma_GPU, Layer20_Gamma_CPU, sizeof(double) * TWENTY_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer20_Beta_GPU, Layer20_Beta_CPU, sizeof(double) * TWENTY_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer20_Weights_CPU);
    free(Layer20_Mean_CPU);
    free(Layer20_StanDev_CPU);
    free(Layer20_Gamma_CPU);
    free(Layer20_Beta_CPU);

    dim3 gridSizeTwentyLayer(512);
    dim3 blockSizeTwenty(14, 14);
    executeTwentyLayer_DSC<<<gridSizeTwentyLayer, blockSizeTwenty>>>(Layer20_Neurons_GPU,
                                                                     Layer20_Weights_GPU,
                                                                     Layer21_Neurons_GPU,
                                                                     Layer20_Mean_GPU,
                                                                     Layer20_StanDev_GPU,
                                                                     Layer20_Gamma_GPU,
                                                                     Layer20_Beta_GPU);

    cudaFree(Layer20_Weights_GPU);
    cudaFree(Layer20_Mean_GPU);
    cudaFree(Layer20_StanDev_GPU);
    cudaFree(Layer20_Gamma_GPU);
    cudaFree(Layer20_Beta_GPU);
}

void Read_TwentyLayer_Data(double *Layer20_Weights_CPU,
                           double *Layer20_Mean_CPU,
                           double *Layer20_StanDev_CPU,
                           double *Layer20_Gamma_CPU,
                           double *Layer20_Beta_CPU)
{
    read_File("data/TwentyLayer/weightsNorm.txt", Layer20_Weights_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_Mean.txt", Layer20_Mean_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_StanDev.txt", Layer20_StanDev_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_Gamma.txt", Layer20_Gamma_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_Beta.txt", Layer20_Beta_CPU);
}

void Execute_TwentyOne_Layer(
    double *Layer21_Neurons_GPU,
    double *Layer22_Neurons_GPU)
{
    double *Layer21_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYONE_LAYER_WEIGHT_SIZE);
    double *Layer21_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYONE_LAYER_CHANNELS);
    double *Layer21_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYONE_LAYER_CHANNELS);
    double *Layer21_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYONE_LAYER_CHANNELS);
    double *Layer21_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYONE_LAYER_CHANNELS);

    Read_TwentyOneLayer_Data(Layer21_Weights_CPU,
                             Layer21_Mean_CPU,
                             Layer21_StanDev_CPU,
                             Layer21_Gamma_CPU,
                             Layer21_Beta_CPU);

    double *Layer21_Weights_GPU,
        *Layer21_Mean_GPU,
        *Layer21_StanDev_GPU,
        *Layer21_Gamma_GPU,
        *Layer21_Beta_GPU;

    cudaMalloc((void **)&Layer21_Weights_GPU, sizeof(double) * TWENTYONE_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer21_Mean_GPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer21_StanDev_GPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer21_Gamma_GPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer21_Beta_GPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS);

    cudaMemcpy(Layer21_Weights_GPU, Layer21_Weights_CPU, sizeof(double) * TWENTYONE_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer21_Mean_GPU, Layer21_Mean_CPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer21_StanDev_GPU, Layer21_StanDev_CPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer21_Gamma_GPU, Layer21_Gamma_CPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer21_Beta_GPU, Layer21_Beta_CPU, sizeof(double) * TWENTYONE_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer21_Weights_CPU);
    free(Layer21_Mean_CPU);
    free(Layer21_StanDev_CPU);
    free(Layer21_Gamma_CPU);
    free(Layer21_Beta_CPU);

    dim3 gridSizeTwentyOneLayer(512);
    dim3 blockSizeTwentyOne(14, 14);
    executeTwentyOneLayer_PSC<<<gridSizeTwentyOneLayer, blockSizeTwentyOne>>>(Layer21_Neurons_GPU,
                                                                              Layer21_Weights_GPU,
                                                                              Layer22_Neurons_GPU,
                                                                              Layer21_Mean_GPU,
                                                                              Layer21_StanDev_GPU,
                                                                              Layer21_Gamma_GPU,
                                                                              Layer21_Beta_GPU);

    cudaFree(Layer21_Weights_GPU);
    cudaFree(Layer21_Mean_GPU);
    cudaFree(Layer21_StanDev_GPU);
    cudaFree(Layer21_Gamma_GPU);
    cudaFree(Layer21_Beta_GPU);
}

void Read_TwentyOneLayer_Data(double *Layer21_Weights_CPU,
                              double *Layer21_Mean_CPU,
                              double *Layer21_StanDev_CPU,
                              double *Layer21_Gamma_CPU,
                              double *Layer21_Beta_CPU)
{
    read_File("data/TwentyOneLayer/weightsNorm.txt", Layer21_Weights_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_Mean.txt", Layer21_Mean_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_StanDev.txt", Layer21_StanDev_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_Gamma.txt", Layer21_Gamma_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_Beta.txt", Layer21_Beta_CPU);
}

void Execute_TwentyTwo_Layer(
    double *Layer22_Neurons_GPU,
    double *Layer23_Neurons_GPU)
{
    double *Layer22_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYTWO_LAYER_WEIGHT_SIZE);
    double *Layer22_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    double *Layer22_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    double *Layer22_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    double *Layer22_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);

    Read_TwentyTwoLayer_Data(Layer22_Weights_CPU,
                             Layer22_Mean_CPU,
                             Layer22_StanDev_CPU,
                             Layer22_Gamma_CPU,
                             Layer22_Beta_CPU);

    double *Layer22_Weights_GPU,
        *Layer22_Mean_GPU,
        *Layer22_StanDev_GPU,
        *Layer22_Gamma_GPU,
        *Layer22_Beta_GPU;

    cudaMalloc((void **)&Layer22_Weights_GPU, sizeof(double) * TWENTYTWO_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer22_Mean_GPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer22_StanDev_GPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer22_Gamma_GPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer22_Beta_GPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS);

    cudaMemcpy(Layer22_Weights_GPU, Layer22_Weights_CPU, sizeof(double) * TWENTYTWO_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer22_Mean_GPU, Layer22_Mean_CPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer22_StanDev_GPU, Layer22_StanDev_CPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer22_Gamma_GPU, Layer22_Gamma_CPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer22_Beta_GPU, Layer22_Beta_CPU, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer22_Weights_CPU);
    free(Layer22_Mean_CPU);
    free(Layer22_StanDev_CPU);
    free(Layer22_Gamma_CPU);
    free(Layer22_Beta_CPU);

    dim3 gridSizeTwentyTwoLayer(512);
    dim3 blockSizeTwentyTwo(14, 14);
    executeTwentyTwoLayer_DSC<<<gridSizeTwentyTwoLayer, blockSizeTwentyTwo>>>(Layer22_Neurons_GPU,
                                                                              Layer22_Weights_GPU,
                                                                              Layer23_Neurons_GPU,
                                                                              Layer22_Mean_GPU,
                                                                              Layer22_StanDev_GPU,
                                                                              Layer22_Gamma_GPU,
                                                                              Layer22_Beta_GPU);

    cudaFree(Layer22_Weights_GPU);
    cudaFree(Layer22_Mean_GPU);
    cudaFree(Layer22_StanDev_GPU);
    cudaFree(Layer22_Gamma_GPU);
    cudaFree(Layer22_Beta_GPU);
}

void Read_TwentyTwoLayer_Data(double *Layer22_Weights_CPU,
                              double *Layer22_Mean_CPU,
                              double *Layer22_StanDev_CPU,
                              double *Layer22_Gamma_CPU,
                              double *Layer22_Beta_CPU)
{
    read_File("data/TwentyTwoLayer/weightsNorm.txt", Layer22_Weights_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_Mean.txt", Layer22_Mean_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_StanDev.txt", Layer22_StanDev_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_Gamma.txt", Layer22_Gamma_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_Beta.txt", Layer22_Beta_CPU);
}

void Execute_TwentyThree_Layer(
    double *Layer23_Neurons_GPU,
    double *Layer24_Neurons_GPU)
{
    double *Layer23_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYTHREE_LAYER_WEIGHT_SIZE);
    double *Layer23_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    double *Layer23_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    double *Layer23_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    double *Layer23_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);

    Read_TwentyThreeLayer_Data(Layer23_Weights_CPU,
                               Layer23_Mean_CPU,
                               Layer23_StanDev_CPU,
                               Layer23_Gamma_CPU,
                               Layer23_Beta_CPU);

    double *Layer23_Weights_GPU,
        *Layer23_Mean_GPU,
        *Layer23_StanDev_GPU,
        *Layer23_Gamma_GPU,
        *Layer23_Beta_GPU;

    cudaMalloc((void **)&Layer23_Weights_GPU, sizeof(double) * TWENTYTHREE_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer23_Mean_GPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer23_StanDev_GPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer23_Gamma_GPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer23_Beta_GPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);

    cudaMemcpy(Layer23_Weights_GPU, Layer23_Weights_CPU, sizeof(double) * TWENTYTHREE_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer23_Mean_GPU, Layer23_Mean_CPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer23_StanDev_GPU, Layer23_StanDev_CPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer23_Gamma_GPU, Layer23_Gamma_CPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer23_Beta_GPU, Layer23_Beta_CPU, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer23_Weights_CPU);
    free(Layer23_Mean_CPU);
    free(Layer23_StanDev_CPU);
    free(Layer23_Gamma_CPU);
    free(Layer23_Beta_CPU);

    dim3 gridSizeTwentyThreeLayer(512);
    dim3 blockSizeTwentyThree(14, 14);
    executeTwentyThreeLayer_PSC<<<gridSizeTwentyThreeLayer, blockSizeTwentyThree>>>(Layer23_Neurons_GPU,
                                                                                    Layer23_Weights_GPU,
                                                                                    Layer24_Neurons_GPU,
                                                                                    Layer23_Mean_GPU,
                                                                                    Layer23_StanDev_GPU,
                                                                                    Layer23_Gamma_GPU,
                                                                                    Layer23_Beta_GPU);

    cudaFree(Layer23_Weights_GPU);
    cudaFree(Layer23_Mean_GPU);
    cudaFree(Layer23_StanDev_GPU);
    cudaFree(Layer23_Gamma_GPU);
    cudaFree(Layer23_Beta_GPU);
}

void Read_TwentyThreeLayer_Data(double *Layer23_Weights_CPU,
                                double *Layer23_Mean_CPU,
                                double *Layer23_StanDev_CPU,
                                double *Layer23_Gamma_CPU,
                                double *Layer23_Beta_CPU)
{
    read_File("data/TwentyThreeLayer/weightsNorm.txt", Layer23_Weights_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_Mean.txt", Layer23_Mean_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_StanDev.txt", Layer23_StanDev_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_Gamma.txt", Layer23_Gamma_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_Beta.txt", Layer23_Beta_CPU);
}

void Execute_TwentyFour_Layer(
    double *Layer24_Neurons_GPU,
    double *Layer25_Neurons_GPU)
{
    double *Layer24_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYFOUR_LAYER_WEIGHT_SIZE);
    double *Layer24_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    double *Layer24_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    double *Layer24_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    double *Layer24_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);

    Read_TwentyFourLayer_Data(Layer24_Weights_CPU,
                              Layer24_Mean_CPU,
                              Layer24_StanDev_CPU,
                              Layer24_Gamma_CPU,
                              Layer24_Beta_CPU);

    double *Layer24_Weights_GPU,
        *Layer24_Mean_GPU,
        *Layer24_StanDev_GPU,
        *Layer24_Gamma_GPU,
        *Layer24_Beta_GPU;

    cudaMalloc((void **)&Layer24_Weights_GPU, sizeof(double) * TWENTYFOUR_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer24_Mean_GPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer24_StanDev_GPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer24_Gamma_GPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer24_Beta_GPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);

    cudaMemcpy(Layer24_Weights_GPU, Layer24_Weights_CPU, sizeof(double) * TWENTYFOUR_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer24_Mean_GPU, Layer24_Mean_CPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer24_StanDev_GPU, Layer24_StanDev_CPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer24_Gamma_GPU, Layer24_Gamma_CPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer24_Beta_GPU, Layer24_Beta_CPU, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer24_Weights_CPU);
    free(Layer24_Mean_CPU);
    free(Layer24_StanDev_CPU);
    free(Layer24_Gamma_CPU);
    free(Layer24_Beta_CPU);

    dim3 gridSizeTwentyFourLayer(512);
    dim3 blockSizeTwentyFour(7, 7);
    executeTwentyFourLayer_DSC<<<gridSizeTwentyFourLayer, blockSizeTwentyFour>>>(Layer24_Neurons_GPU,
                                                                                 Layer24_Weights_GPU,
                                                                                 Layer25_Neurons_GPU,
                                                                                 Layer24_Mean_GPU,
                                                                                 Layer24_StanDev_GPU,
                                                                                 Layer24_Gamma_GPU,
                                                                                 Layer24_Beta_GPU);

    cudaFree(Layer24_Weights_GPU);
    cudaFree(Layer24_Mean_GPU);
    cudaFree(Layer24_StanDev_GPU);
    cudaFree(Layer24_Gamma_GPU);
    cudaFree(Layer24_Beta_GPU);
}

void Read_TwentyFourLayer_Data(double *Layer24_Weights_CPU,
                               double *Layer24_Mean_CPU,
                               double *Layer24_StanDev_CPU,
                               double *Layer24_Gamma_CPU,
                               double *Layer24_Beta_CPU)
{
    read_File("data/TwentyFourLayer/weightsNorm.txt", Layer24_Weights_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_Mean.txt", Layer24_Mean_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_StanDev.txt", Layer24_StanDev_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_Gamma.txt", Layer24_Gamma_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_Beta.txt", Layer24_Beta_CPU);
}

void Execute_TwentyFive_Layer(
    double *Layer25_Neurons_GPU,
    double *Layer26_Neurons_GPU)
{
    double *Layer25_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYFIVE_LAYER_WEIGHT_SIZE);
    double *Layer25_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    double *Layer25_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    double *Layer25_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    double *Layer25_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);

    Read_TwentyFiveLayer_Data(Layer25_Weights_CPU,
                              Layer25_Mean_CPU,
                              Layer25_StanDev_CPU,
                              Layer25_Gamma_CPU,
                              Layer25_Beta_CPU);

    double *Layer25_Weights_GPU,
        *Layer25_Mean_GPU,
        *Layer25_StanDev_GPU,
        *Layer25_Gamma_GPU,
        *Layer25_Beta_GPU;

    cudaMalloc((void **)&Layer25_Weights_GPU, sizeof(double) * TWENTYFIVE_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer25_Mean_GPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer25_StanDev_GPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer25_Gamma_GPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer25_Beta_GPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);

    cudaMemcpy(Layer25_Weights_GPU, Layer25_Weights_CPU, sizeof(double) * TWENTYFIVE_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer25_Mean_GPU, Layer25_Mean_CPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer25_StanDev_GPU, Layer25_StanDev_CPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer25_Gamma_GPU, Layer25_Gamma_CPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer25_Beta_GPU, Layer25_Beta_CPU, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer25_Weights_CPU);
    free(Layer25_Mean_CPU);
    free(Layer25_StanDev_CPU);
    free(Layer25_Gamma_CPU);
    free(Layer25_Beta_CPU);

    dim3 gridSizeTwentyFiveLayer(1024);
    dim3 blockSizeTwentyFive(7, 7);
    executeTwentyFiveLayer_PSC<<<gridSizeTwentyFiveLayer, blockSizeTwentyFive>>>(Layer25_Neurons_GPU,
                                                                                 Layer25_Weights_GPU,
                                                                                 Layer26_Neurons_GPU,
                                                                                 Layer25_Mean_GPU,
                                                                                 Layer25_StanDev_GPU,
                                                                                 Layer25_Gamma_GPU,
                                                                                 Layer25_Beta_GPU);

    cudaFree(Layer25_Weights_GPU);
    cudaFree(Layer25_Mean_GPU);
    cudaFree(Layer25_StanDev_GPU);
    cudaFree(Layer25_Gamma_GPU);
    cudaFree(Layer25_Beta_GPU);
}

void Read_TwentyFiveLayer_Data(double *Layer25_Weights_CPU,
                               double *Layer25_Mean_CPU,
                               double *Layer25_StanDev_CPU,
                               double *Layer25_Gamma_CPU,
                               double *Layer25_Beta_CPU)
{
    read_File("data/TwentyFiveLayer/weightsNorm.txt", Layer25_Weights_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_Mean.txt", Layer25_Mean_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_StanDev.txt", Layer25_StanDev_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_Gamma.txt", Layer25_Gamma_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_Beta.txt", Layer25_Beta_CPU);
}

void Execute_TwentySix_Layer(
    double *Layer26_Neurons_GPU,
    double *Layer27_Neurons_GPU)
{
    double *Layer26_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYSIX_LAYER_WEIGHT_SIZE);
    double *Layer26_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    double *Layer26_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    double *Layer26_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    double *Layer26_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);

    Read_TwentySixLayer_Data(Layer26_Weights_CPU,
                             Layer26_Mean_CPU,
                             Layer26_StanDev_CPU,
                             Layer26_Gamma_CPU,
                             Layer26_Beta_CPU);

    double *Layer26_Weights_GPU,
        *Layer26_Mean_GPU,
        *Layer26_StanDev_GPU,
        *Layer26_Gamma_GPU,
        *Layer26_Beta_GPU;

    cudaMalloc((void **)&Layer26_Weights_GPU, sizeof(double) * TWENTYSIX_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer26_Mean_GPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer26_StanDev_GPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer26_Gamma_GPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer26_Beta_GPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS);

    cudaMemcpy(Layer26_Weights_GPU, Layer26_Weights_CPU, sizeof(double) * TWENTYSIX_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer26_Mean_GPU, Layer26_Mean_CPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer26_StanDev_GPU, Layer26_StanDev_CPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer26_Gamma_GPU, Layer26_Gamma_CPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer26_Beta_GPU, Layer26_Beta_CPU, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer26_Weights_CPU);
    free(Layer26_Mean_CPU);
    free(Layer26_StanDev_CPU);
    free(Layer26_Gamma_CPU);
    free(Layer26_Beta_CPU);

    dim3 gridSizeTwentySixLayer(1024);
    dim3 blockSizeTwentySix(7, 7);
    executeTwentySixLayer_DSC<<<gridSizeTwentySixLayer, blockSizeTwentySix>>>(Layer26_Neurons_GPU,
                                                                              Layer26_Weights_GPU,
                                                                              Layer27_Neurons_GPU,
                                                                              Layer26_Mean_GPU,
                                                                              Layer26_StanDev_GPU,
                                                                              Layer26_Gamma_GPU,
                                                                              Layer26_Beta_GPU);

    cudaFree(Layer26_Weights_GPU);
    cudaFree(Layer26_Mean_GPU);
    cudaFree(Layer26_StanDev_GPU);
    cudaFree(Layer26_Gamma_GPU);
    cudaFree(Layer26_Beta_GPU);
}

void Read_TwentySixLayer_Data(double *Layer26_Weights_CPU,
                              double *Layer26_Mean_CPU,
                              double *Layer26_StanDev_CPU,
                              double *Layer26_Gamma_CPU,
                              double *Layer26_Beta_CPU)
{
    read_File("data/TwentySixLayer/weightsNorm.txt", Layer26_Weights_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_Mean.txt", Layer26_Mean_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_StanDev.txt", Layer26_StanDev_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_Gamma.txt", Layer26_Gamma_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_Beta.txt", Layer26_Beta_CPU);
}

void Execute_TwentySeven_Layer(
    double *Layer27_Neurons_GPU,
    double *Layer28_Neurons_GPU)
{
    double *Layer27_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYSEVEN_LAYER_WEIGHT_SIZE);
    double *Layer27_Mean_CPU = (double *)malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    double *Layer27_StanDev_CPU = (double *)malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    double *Layer27_Gamma_CPU = (double *)malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    double *Layer27_Beta_CPU = (double *)malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);

    Read_TwentySevenLayer_Data(Layer27_Weights_CPU,
                               Layer27_Mean_CPU,
                               Layer27_StanDev_CPU,
                               Layer27_Gamma_CPU,
                               Layer27_Beta_CPU);

    double *Layer27_Weights_GPU,
        *Layer27_Mean_GPU,
        *Layer27_StanDev_GPU,
        *Layer27_Gamma_GPU,
        *Layer27_Beta_GPU;

    cudaMalloc((void **)&Layer27_Weights_GPU, sizeof(double) * TWENTYSEVEN_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer27_Mean_GPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer27_StanDev_GPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer27_Gamma_GPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    cudaMalloc((void **)&Layer27_Beta_GPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);

    cudaMemcpy(Layer27_Weights_GPU, Layer27_Weights_CPU, sizeof(double) * TWENTYSEVEN_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer27_Mean_GPU, Layer27_Mean_CPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer27_StanDev_GPU, Layer27_StanDev_CPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer27_Gamma_GPU, Layer27_Gamma_CPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer27_Beta_GPU, Layer27_Beta_CPU, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer27_Weights_CPU);
    free(Layer27_Mean_CPU);
    free(Layer27_StanDev_CPU);
    free(Layer27_Gamma_CPU);
    free(Layer27_Beta_CPU);

    dim3 gridSizeTwentySevenLayer(1024);
    dim3 blockSizeTwentySeven(7, 7);
    executeTwentySevenLayer_PSC<<<gridSizeTwentySevenLayer, blockSizeTwentySeven>>>(Layer27_Neurons_GPU,
                                                                                    Layer27_Weights_GPU,
                                                                                    Layer28_Neurons_GPU,
                                                                                    Layer27_Mean_GPU,
                                                                                    Layer27_StanDev_GPU,
                                                                                    Layer27_Gamma_GPU,
                                                                                    Layer27_Beta_GPU);

    cudaFree(Layer27_Weights_GPU);
    cudaFree(Layer27_Mean_GPU);
    cudaFree(Layer27_StanDev_GPU);
    cudaFree(Layer27_Gamma_GPU);
    cudaFree(Layer27_Beta_GPU);
}

void Read_TwentySevenLayer_Data(double *Layer27_Weights_CPU,
                                double *Layer27_Mean_CPU,
                                double *Layer27_StanDev_CPU,
                                double *Layer27_Gamma_CPU,
                                double *Layer27_Beta_CPU)
{
    read_File("data/TwentySevenLayer/weightsNorm.txt", Layer27_Weights_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_Mean.txt", Layer27_Mean_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_StanDev.txt", Layer27_StanDev_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_Gamma.txt", Layer27_Gamma_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_Beta.txt", Layer27_Beta_CPU);
}

void Execute_TwentyEight_Layer(
    double *Layer28_Neurons_GPU,
    double *Layer29_Neurons_GPU)
{
    dim3 gridSizeTwentyEightLayer(1);
    dim3 blockSizeTwentyEight(32, 32);

    executeTwentyEightLayer_AvgPooling<<<gridSizeTwentyEightLayer, blockSizeTwentyEight>>>(Layer28_Neurons_GPU,
                                                                                           Layer29_Neurons_GPU);
}

void Execute_TwentyNine_Layer(
    double *Layer29_Neurons_GPU,
    double *Layer30_Neurons_GPU)
{
    double *Layer29_Weights_CPU = (double *)malloc(sizeof(double) * TWENTYNINE_LAYER_WEIGHT_SIZE);
    double *Layer29_Bias_CPU = (double *)malloc(sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE);

    Read_TwentyNineLayer_Data(Layer29_Weights_CPU,
                              Layer29_Bias_CPU);

    double *Layer29_Weights_GPU,
        *Layer29_Bias_GPU;

    cudaMalloc((void **)&Layer29_Weights_GPU, sizeof(double) * TWENTYNINE_LAYER_WEIGHT_SIZE);
    cudaMalloc((void **)&Layer29_Bias_GPU, sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE);

    cudaMemcpy(Layer29_Weights_GPU, Layer29_Weights_CPU, sizeof(double) * TWENTYNINE_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer29_Bias_GPU, Layer29_Bias_CPU, sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE, cudaMemcpyHostToDevice);

    free(Layer29_Weights_CPU);
    free(Layer29_Bias_CPU);

    dim3 gridSizeTwentyNineLayer(1);
    dim3 blockSizeTwentyNine(1000);
    executeTwentyNineLayer_FullyConnected<<<gridSizeTwentyNineLayer, blockSizeTwentyNine>>>(Layer29_Neurons_GPU,
                                                                                            Layer30_Neurons_GPU,
                                                                                            Layer29_Weights_GPU,
                                                                                            Layer29_Bias_GPU);

    cudaFree(Layer29_Weights_GPU);
    cudaFree(Layer29_Bias_GPU);
}

void Read_TwentyNineLayer_Data(double *Layer29_Weights_CPU,
                               double *Layer29_Bias_CPU)
{
    read_File("data/TwentyNineLayer/weightsNorm.txt", Layer29_Weights_CPU);
    read_File("data/TwentyNineLayer/biases.txt", Layer29_Bias_CPU);
}

void read_File(const char *input_FileName, double *input_values)
{

    FILE *fp = fopen(input_FileName, "r");
    if (fp == NULL)
    {
        printf("\n No input file present at the location \n");
        return;
    }

    int counter = 0;
    ssize_t read;
    char *line = NULL;
    size_t len = 1000;

    while ((read = getline(&line, &len, fp)) != -1)
        input_values[counter++] = atof(line);
    fclose(fp);
}

void read_Input_File(const char *inputFileName, double *Layer1_Neurons_CPU)
{
    FILE *fp = fopen(inputFileName, "r");

    if (fp == NULL)
    {
        printf("\n No input file present at the location \n");
        return;
    }

    int counter = 0;
    ssize_t read;
    char *line = NULL;
    size_t len = 1000;
    int index = 0;
    int lastRow = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
        Layer1_Neurons_CPU[counter++] = atof(line);
        index++;
        // handle padding
        if (index == 224)
        {
            Layer1_Neurons_CPU[counter++] = 0;
            index = 0;
            lastRow++;
            if (lastRow == 224)
            {
                lastRow = 0;
                int temp = 0;
                while (temp < 225)
                {
                    Layer1_Neurons_CPU[counter++] = 0;
                    temp++;
                }
            }
        }
    }
    read = 0;
    fclose(fp);
}