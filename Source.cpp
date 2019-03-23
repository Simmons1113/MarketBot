
#include "fileUtility.h"
#include "network.h"
#include <iostream>

int main()
{
	std::cout << "Reading Files" << std::endl << std::endl;

	std::cout << "Generating Neural Network" << std::endl << std::endl;

	int dataStreamSampleBufferLength = 2000;

	//=========================================================================================================================
	//Network Initialization

	network myNetwork(4, dataStreamSampleBufferLength,500,1); // Create Network

	myNetwork.randomizeNetwork(); // Randomize Network Weights and Biases

	myNetwork.setCircularInputs(true, 1); // Let Network know that we are using a buffer

	int miniBatchSize(10);

	float learningRate(0.1);

	uint32_t predictionDistance = 50;

	float myBatchCost;

	float batchAccuracy;

	int sampleIndex;


	//=========================================================================================================================
	//Data File Initialization

	fileUtility data;
	data.loadFile("");

	//=========================================================================================================================
	//Start Training

	int numBatches = (data.getReadFileSize() - dataStreamSampleBufferLength ) / (miniBatchSize);

	std::cout << "Learning" << std::endl << std::endl;

	for (int s = 0; s < dataStreamSampleBufferLength; s++) // Fill DataStreamBuffer
	{
		myNetwork.setInputValueCircular(data.getFloat(s));
	}

	for (int b = 0; b < numBatches; b++)
	{
		myBatchCost = 0.0f;

		for (int m = 0; m < miniBatchSize; m++)
		{
			sampleIndex = dataStreamSampleBufferLength + b * miniBatchSize + m;

			myNetwork.setInputValueCircular(data.getFloat(sampleIndex));

			myNetwork.inputCorrect(data.getU32B(sampleIndex + predictionDistance), 0);

			myNetwork.think();

			myNetwork.calculateCost();

			myNetwork.learn(learningRate, miniBatchSize);

			myBatchCost += myNetwork.getCost() / miniBatchSize;
		}

		myNetwork.applyLearned();

		std::cout << "Batch : " << (b + 1) << " / " << " Cost : " << myBatchCost << std::endl;
	}
	
	myNetwork.saveNetwork("LastNetwork", 0);

	std::cout << std::endl << "Done Training!" << std::endl << std::endl;

	//=======================================================================================================================================================================================

	return 0;
}
