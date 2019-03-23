#include "layer.h"

layer::layer(int neuronNum, int weightNum) :
	neuronCount(neuronNum),
	weightCount(weightNum),
	circularIndex(-1),
	circularBufferOn(0)
{
	weight.resize(neuronCount);

	for (uint32_t j = 0; j < neuronCount; j++)
	{
		weight[j].resize(weightCount);
	}

	bias.resize(neuronCount);
	a.resize(neuronCount);
	z.resize(neuronCount);
	dCdA.resize(neuronCount);
	dCdW.resize(neuronCount);

	for (uint32_t i = 0; i < neuronCount; i++)
	{
		dCdW[i].resize(weightCount);
	}

	dCdB.resize(neuronCount);
	reset();
}

//============================================================

void layer::randomize()
{
	for (uint32_t j = 0; j < neuronCount; j++)
	{
		bias[j] = 2 * ((static_cast <float> (rand())) / (static_cast <float> (RAND_MAX))) - 1;

		for (uint32_t k = 0; k < weightCount; k++)
		{
			weight[j][k] = 2 * ((static_cast <float> (rand())) / (static_cast <float> (RAND_MAX))) - 1;
		}
	}
}

//============================================================

void layer::fire(layer const *prevLayer)
{
	for (uint32_t j = 0; j < neuronCount; j++)
	{
		z[j] = 0.0f;
		for (uint32_t k = 0; k < weightCount; k++)
		{
			z[j] += prevLayer->getActivation(k) * weight[j][k];
		}
		z[j] += bias[j];
		a[j] = applyTransform(z[j]);
	}
}

//============================================================

float layer::applyTransform(float const inputValue)
{
	//return (1.0f / (1 + exp(-inputValue)));
	if (inputValue > 1.0f)
	{
		return 1.0f;
	}
	else if (inputValue < -1.0f)
	{
		return -1.0f;
	}
	else
	{
		return inputValue;
	}
}

//============================================================

void layer::applyLearnedAndReset()
{
	for (uint32_t j = 0; j < neuronCount; j++)
	{
		for (uint32_t k = 0; k < weightCount; k++)
		{
			weight[j][k] += dCdW[j][k];
			dCdW[j][k] = 0.0f;
		}
		bias[j] += dCdB[j];
		dCdB[j] = 0.0f;
	}
}

void layer::reset()
{
	for (uint32_t j = 0; j < neuronCount; j++)
	{
		dCdA[j] = 0.0f;
		dCdB[j] = 0.0f;
		for (uint32_t k = 0; k < weightCount; k++)
		{
			dCdW[j][k] = 0.0f;
		}
	}
}

//============================================================

void layer::movedCdW(float const value, int const neuron, int const weight)
{
	dCdW[neuron][weight] += value;
}

void layer::movedCdB(float const value, int const neuron)
{
	dCdB[neuron] += value;
}

void layer::movedCdA(float const value, int const neuron)
{
	dCdA[neuron] += value;
}

//============================================================

void layer::inputActivation(float const value, int const neuron)
{
	a[neuron] = value;
}

void layer::inputActivationCircular(float const value)
{
	circularIndex++;
	circularIndex %= neuronCount;
	a[circularIndex] = value;
}

float layer::getActivation(int const neuron) const
{
	if (circularBufferOn)
	{
		int x = circularIndex - bufferSize + 1 + neuron;
		if (x < 0)
		{
			x += neuronCount;
		}
		x %= neuronCount;
		return a[x];
	}
	else
	{
		return a[neuron];
	}
}

int layer::getNumNeurons() const
{
	return neuronCount;
}

void layer::inputdCdA(float const value, int const neuron)
{
	dCdA[neuron] = value;
}

float layer::getdCdA(int const neuron) const
{
	return dCdA[neuron];
}

float layer::getZ(int const neuron) const
{
	return z[neuron];
}

float layer::getWeight(int const neuronIndex, int const weightIndex) const
{
	return weight[neuronIndex][weightIndex];
}

void layer::moveWeight(float const value, int const neuronIndex, int const weightIndex)
{
	weight[neuronIndex][weightIndex] += value;
}

void layer::setWeight(float const value, int const neuronIndex, int const weightIndex)
{
	weight[neuronIndex][weightIndex] = value;
}

float layer::getBias(int const neuronIndex) const
{
	return bias[neuronIndex];
}

void layer::setBias(float value, int const neuronIndex)
{
	bias[neuronIndex] = value;
}

void layer::setCircular(bool a, uint32_t bufferS)
{
	circularBufferOn = a;
	bufferSize = bufferS;
}