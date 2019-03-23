#pragma once
#include <vector>

class layer
{
public:

	layer(int neuronNum, int weightNum);

	//============================================================

	void randomize();

	//============================================================

	void fire(layer const *prevLayer);

	//============================================================

	float applyTransform(float const inputValue);

	//============================================================

	void applyLearnedAndReset();

	void reset();

	//============================================================

	void movedCdW(float const value, int const neuron, int const weight);

	void movedCdB(float const value, int const neuron);

	void movedCdA(float const value, int const neuron);

	//============================================================

	void inputActivation(float const value, int const neuron);

	void inputActivationCircular(float const value);

	float getActivation(int const neuron) const;

	int getNumNeurons() const;

	void inputdCdA(float const value, int const neuron);

	float getdCdA(int const neuron) const;

	float getZ(int const neuron) const;

	float getWeight(int const neuronIndex, int const weightIndex) const;

	void moveWeight(float const value, int const neuronIndex, int const weightIndex);

	void setWeight(float const value, int const neuronIndex, int const weightIndex);

	float getBias(int const neuronIndex) const;

	void setBias(float value, int const neuronIndex);

	void setCircular(bool a, uint32_t bufferS);

private:

	uint32_t neuronCount;

	uint32_t weightCount;

	uint32_t circularIndex;
	
	bool circularBufferOn;

	uint32_t bufferSize;

	std::vector<float> a;

	std::vector<float> z;

	std::vector<std::vector<float>> weight;

	std::vector<float> bias;

	std::vector<float> dCdA;

	std::vector<std::vector<float>> dCdW;

	std::vector<float> dCdB;
};