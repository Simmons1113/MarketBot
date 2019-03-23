#pragma once
#include <cmath>
#include <vector>
#include "layer.h"
#include "fileUtility.h"

class network
{
public:

	network(int const iNumLayers, int const iNumInputs, int const iNumHiddens, int const iNumOutputs);

	network(const char fileName[]);

	network();

	~network();
	
	//==================================================================================================================================================
	
	void saveNetwork(const char fileName[], uint32_t ID);

	void loadNetwork(const char fileName[]);

	//==================================================================================================================================================

	void randomizeNetwork();

	void makeLinear();

	//==================================================================================================================================================

	void setInputValue(float inputValue, int index);

	void setInputValueCircular(float const inputValue);

	void setCircularInputs(bool a, uint32_t bufferS);

	//==================================================================================================================================================

	void think();

	float getNNAns(int index) const;

	//==================================================================================================================================================

	void inputCorrect(float value, int index);

	void calculateCost();

	float getCost() const;

	//==================================================================================================================================================

	void learn(float learningRate, int batchSize);

	void applyLearned();

	//==================================================================================================================================================

private:

	uint32_t floatToUI(float i);

	float UIToFloat(uint32_t i);

	//==================================================================================================================================================

	void makeNetwork();

	void deleteNetwork();

	//==================================================================================================================================================

	float transformPrime(float inputValue);

	float transform(float inputValue);

	//==================================================================================================================================================

	uint32_t ID;

	std::vector<layer*> layerList;

	std::vector<float> correctAns;

	int numInputs;

	int numHiddens;

	int numOutputs;

	int numLayers;

	bool circularInputs;

	uint32_t bufferSize;

	float cost;


};