
#include "fileUtility.h"
#include "network.h"
#include <iostream>

int main()
{
	std::cout << "Reading Files" << std::endl << std::endl;

	std::cout << "Generating Neural Network" << std::endl << std::endl;

	int inputSamples = 2000;

	int bufferSize = 1;

	//=========================================================================================================================
	//Network Initialization

	network myNetwork(3,inputSamples,150,1); 
	myNetwork.randomizeNetwork();
	myNetwork.setCircularInputs(true, bufferSize);

	int numLoops = 1;

	int miniBatchSize(10);

	int numTests;

	float learningRate(0.1);

	float myBatchCost;

	float batchAccuracy;

	int sampleIndex;

	uint32_t predictionDistance = 50;

	//=========================================================================================================================
	//Audio File Initialization

	fileUtility data;
	data.loadFile("");

	//=========================================================================================================================
	//Start Training

	numTests = (data.getReadFileSize() ) / (miniBatchSize * bufferSize);

	std::cout << "Learning" << std::endl << std::endl;


	for (int l = 0; l < numLoops; l++)
	{
		for (int b = 0; b < numTests; b++)
		{
			myBatchCost = 0.0f;
			for (int m = 0; m < miniBatchSize; m++)
			{
				sampleIndex = inputSamples + b * miniBatchSize * bufferSize + m * bufferSize;

				for (int s = 0; s < bufferSize; s++)
				{
					myNetwork.setInputValueCircular(data.getU32B(sampleIndex + s));
				}
				myNetwork.inputCorrect(data.getU32B(sampleIndex + bufferSize + predictionDistance), 0);
				myNetwork.think();
				myNetwork.calculateCost();
				myNetwork.learn(learningRate, miniBatchSize);
				myBatchCost += myNetwork.getCost() / miniBatchSize;
			}
			myNetwork.applyLearned();
			std::cout << "Batch : " << (l * numTests) +(b + 1) << " / " << numTests * numLoops  << " Cost : " << myBatchCost << std::endl;
		}
	}

	
	myNetwork.saveNetwork("LastNetwork", 0);

	std::cout << std::endl << "Done Training!" << std::endl << std::endl;

	//=======================================================================================================================================================================================

	return 0;
}