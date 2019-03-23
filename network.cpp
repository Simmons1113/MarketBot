#include "network.h"

#include <cmath>
#include <vector>
#include "layer.h"
#include "fileUtility.h"

	network::network(int const iNumLayers, int const iNumInputs, int const iNumHiddens, int const iNumOutputs) :
		numLayers(iNumLayers),
		numInputs(iNumInputs),
		numHiddens(iNumHiddens),
		numOutputs(iNumOutputs),
		circularInputs(0),
		bufferSize(0)
	{
		makeNetwork();
	}

	network::network(const char fileName[])
	{
		loadNetwork(fileName);
	}

	network::network() :
		numLayers(0),
		numInputs(0),
		numHiddens(0),
		numOutputs(0)
	{

	}

	network::~network()
	{
		deleteNetwork();
	}

	//==================================================================================================================================================

	void network::saveNetwork(const char fileName[], uint32_t ID)
	{
		fileUtility myFile;

		int fileSize = 0;
		fileSize += (4 * ((numHiddens * (numLayers - 2) + numOutputs)));
		fileSize += (4 * (numInputs * numHiddens));
		for (int i = 0; i < (numLayers - 3); i++)
		{
			fileSize += (4 * (numHiddens * numHiddens));
		}

		fileSize += (4 * (numOutputs * numHiddens));

		// Header Size : NN(2), ID(4), numLayers(4), numInputs(4), numHiddens(4), numOutputs(4), "data"(4) = 26

		fileSize += 26;

		myFile.setWriteSize(fileSize);

		//==============================================================================================================

		// Write Header

		myFile.addChar('N');
		myFile.addChar('N');

		myFile.add32B(ID);

		myFile.add32B(numLayers);

		myFile.add32B(numInputs);

		myFile.add32B(numHiddens);

		myFile.add32B(numOutputs);

		myFile.addChar(circularInputs);

		myFile.add32B(bufferSize);

		myFile.addChar('d');

		myFile.addChar('a');

		myFile.addChar('t');

		myFile.addChar('a');

		//=====================================================================================================
		// Data


		for (int i = 1; i < numLayers; i++) // Layer
		{
			int z = layerList[i]->getNumNeurons();

			for (int j = 0; j < z; j++) // Neuron
			{
				float a = layerList[i]->getBias(j);
				myFile.add32B(floatToUI(layerList[i]->getBias(j)));

				int y = layerList[i - 1]->getNumNeurons();

				for (int k = 0; k < y; k++) // Weight
				{
					a = layerList[i]->getWeight(j, k);
					myFile.add32B(floatToUI(layerList[i]->getWeight(j, k)));
				}
			}
		}

		//==================================================================================================================================================
		// Write to file

		myFile.makeFile(fileName);
	}

	void network::loadNetwork(const char fileName[])
	{

		deleteNetwork();

		// Copy Bytes to fileMap

		fileUtility loadFileU;
		loadFileU.loadFile(fileName);

		// Read Hedder

		if (loadFileU.get2C(0) != 'NN')
		{
			throw std::runtime_error("Bad Header");
			return;
		}

		ID = loadFileU.getU32B(2);

		numLayers = loadFileU.getU32B(6);

		numInputs = loadFileU.getU32B(10);

		numHiddens = loadFileU.getU32B(14);

		numOutputs = loadFileU.getU32B(18);

		circularInputs = loadFileU.getByte(22);

		bufferSize = loadFileU.getU32B(23);

		makeNetwork();

		setCircularInputs(circularInputs, bufferSize);

		// Read Data

		if (loadFileU.get4C(27) != 'data')
		{
			throw std::runtime_error("Bad Header");
			return;
		}

		uint32_t mapIndex = 31;

		for (int i = 1; i < numLayers; i++) // Layer
		{
			int z = layerList[i]->getNumNeurons();

			for (int j = 0; j < z; j++) // Neuron
			{
				float a = UIToFloat(loadFileU.getU32B(mapIndex));
				layerList[i]->setBias(UIToFloat(loadFileU.getU32B(mapIndex)), j);
				mapIndex += 4;

				int y = layerList[i - 1]->getNumNeurons();

				for (int k = 0; k < y; k++) // Weight
				{
					a = UIToFloat(loadFileU.getU32B(mapIndex));
					layerList[i]->setWeight(UIToFloat(loadFileU.getU32B(mapIndex)), j, k);
					mapIndex += 4;
				}
			}
		}

	}

	//==================================================================================================================================================

	void network::randomizeNetwork()
	{
		for (int i = 1; i < numLayers; i++)
		{
			layerList[i]->randomize();
		}
	}

	void network::makeLinear()
	{
		if ((numHiddens >= numOutputs) && (numInputs >= numHiddens))
		{
			for (int i = 1; i < numLayers; i++)
			{
				int z = layerList[i]->getNumNeurons();

				for (int j = 0; j < z; j++)
				{
					layerList[i]->setBias(0.0f, j);

					int y = layerList[i - 1]->getNumNeurons();

					for (int k = 0; k < y; k++)
					{
						float a = 0.0f;
						if ((k < numInputs) && (j == k))
						{
							a = 1.0f;
						}
						layerList[i]->setWeight(a, j, k);
					}
				}
			}
		}
	}
	//==================================================================================================================================================

	void network::setInputValue(float inputValue, int index)
	{
		layerList[0]->inputActivation(inputValue, index);
	}

	void network::setInputValueCircular(float const inputValue)
	{
		layerList[0]->inputActivationCircular(inputValue);
	}

	void network::setCircularInputs(bool a, uint32_t bufferS)
	{
		circularInputs = a;
		bufferSize = bufferS;
		layerList[0]->setCircular(a, bufferS);
	}

	//==================================================================================================================================================

	void network::think()
	{
		for (int i = 1; i < numLayers; i++)
		{
			layer const * p = layerList[i - 1];
			layerList[i]->fire(p);
		}
	}

	float network::getNNAns(int index) const
	{
		return layerList[numLayers - 1]->getActivation(index);
	}

	//==================================================================================================================================================

	void network::inputCorrect(float value, int index)
	{
		correctAns[index] = value;
	}

	void network::calculateCost()
	{
		cost = 0.0f;

		for (int i = 0; i < numOutputs; i++)
		{
			cost += pow((correctAns[i] - layerList[numLayers - 1]->getActivation(i)), 2) * 0.5f;
		}
	}

	float network::getCost() const
	{
		return cost;
	}

	//==================================================================================================================================================

	void network::learn(float learningRate, int batchSize) // stochastic gradient descent + back propegation
	{

		//Calculate Output Layer

		for (int j = 0; j < numOutputs; j++)
		{
			layerList[numLayers - 1]->inputdCdA((layerList[numLayers - 1]->getActivation(j) - correctAns[j]) * transformPrime(layerList[numLayers - 1]->getZ(j)), j);

			for (int k = 0; k < numHiddens; k++)
			{
				layerList[numLayers - 1]->movedCdW(-(learningRate / (float)batchSize) * layerList[numLayers - 1]->getdCdA(j) * layerList[numLayers - 2]->getActivation(k), j, k);
			}

			layerList[numLayers - 1]->movedCdB(-(learningRate / (float)batchSize) *  layerList[numLayers - 1]->getdCdA(j), j);
		}

		//Calculate Hidden Layers

		for (int i = (numLayers - 2); i > 0; i--)
		{
			for (int j = 0; j < layerList[i]->getNumNeurons(); j++)
			{
				if (j == 100 && i == 1)
				{
					int breaking = 1;
				}
				layerList[i]->inputdCdA(0.0f, j);

				for (int k = 0; k < layerList[i + 1]->getNumNeurons(); k++)
				{
					layerList[i]->movedCdA(layerList[i + 1]->getWeight(k, j) * layerList[i + 1]->getdCdA(k) * transformPrime(layerList[i]->getZ(j)), j);
				}

				for (int k = 0; k < layerList[i - 1]->getNumNeurons(); k++)
				{
					layerList[i]->movedCdW(-(learningRate / (float)batchSize) * layerList[i]->getdCdA(j) * layerList[i - 1]->getActivation(k), j, k);
				}

				layerList[i]->movedCdB(-(learningRate / (float)batchSize) * layerList[i]->getdCdA(j), j);
			}
		}
	}

	void network::applyLearned()
	{
		for (int i = 1; i < numLayers; i++)
		{
			layerList[i]->applyLearnedAndReset();
		}
	}

	//==================================================================================================================================================

	uint32_t network::floatToUI(float i)
	{
		union
		{
			uint32_t sampUI;
			float sampF;
		};

		sampF = i;

		return sampUI;
	}

	float network::UIToFloat(uint32_t i)
	{
		union
		{
			uint32_t sampUI;
			float sampF;
		};

		sampUI = i;
		return sampF;
	}

	//==================================================================================================================================================

	void network::makeNetwork()
	{
		if (numLayers < 3 || numInputs < 1 || numHiddens < 1 || numOutputs < 1)
		{
			throw std::runtime_error("Invalid Network Construction");
			return;
		}

		correctAns.resize(numOutputs);
		layerList.resize(numLayers);

		layerList[0] = new layer(numInputs, 0);
		layerList[1] = new layer(numHiddens, numInputs);
		for (int i = 2; i < (numLayers - 1); i++)
		{
			layerList[i] = new layer(numHiddens, numHiddens);
		}
		layerList[numLayers - 1] = new layer(numOutputs, numHiddens);
	}

	void network::deleteNetwork()
	{
		if (!numLayers)
		{
			for (int i = 0; i < numLayers; i++)
			{
				delete layerList[i];
			}
		}
	}

	//==================================================================================================================================================

	float network::transformPrime(float inputValue) // Sigmoid
	{
		//return (transform(inputValue) * (1 - transform(inputValue)));
		if ((inputValue > 1.0f) || (inputValue < -1.0f))
		{
			return 0.0f;
		}
		else
		{
			return 1.0f;
		}
	}

	float network::transform(float inputValue) // Sigmoid
	{
		//return (1.0f / (1 + exp(-inputValue)));
		if (inputValue > 1.0f)
		{
			return 1.0f;
		}
		if (inputValue < -1.0f)
		{
			return -1.0f;
		}
		return inputValue;
	}
