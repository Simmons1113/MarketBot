#pragma once
#include <fstream>
#include "newCharArray.h"

class fileUtility
{
public:

	void makeFile(const char fileName[])
	{
		unsigned char c;

		write.open(fileName, std::ios::binary | std::ios::out);

		for (int i = 0; i < writeMap.getSize(); i++)
		{
			c = writeMap.get(i);
			write.write((char *)&c, 1);
		}

		write.close();
	}

	void setWriteSize(int bytes)
	{
		writeMap.setSize(bytes);
	}

	void addChar(unsigned char c)
	{
		writeMap.input(c, writeIndex);
		writeIndex++;
	}

	void add32L(uint32_t v)
	{
		unsigned char a = (v & 0x000000ff);
		unsigned char b = (v & 0x0000ff00) >> 8;
		unsigned char c = (v & 0x00ff0000) >> 16;
		unsigned char d = (v & 0xff000000) >> 24;
		addChar(d);
		addChar(c);
		addChar(b);
		addChar(a);
	}

	void add32B(uint32_t v)
	{
		unsigned char a = (v & 0x000000ff);
		unsigned char b = (v & 0x0000ff00) >> 8;
		unsigned char c = (v & 0x00ff0000) >> 16;
		unsigned char d = (v & 0xff000000) >> 24;
		addChar(a);
		addChar(b);
		addChar(c);
		addChar(d);
	}

	void add16L(uint16_t v)
	{
		unsigned char a = (v & 0x00ff);
		unsigned char b = (v & 0xff00) >> 8;
		addChar(b);
		addChar(a);
	}

	void add16B(uint16_t v)
	{
		unsigned char a = (v & 0x00ff);
		unsigned char b = (v & 0xff00) >> 8;
		addChar(a);
		addChar(b);
	}

	//=================================================================================================================

	void loadFile(const char fileName[])
	{
		unsigned char c;

		read.open(fileName, std::ios::binary | std::ios::in);

		readFileSize = filesize(fileName);
		readMap.setSize(readFileSize);

		for (int i = 0; i < readFileSize; i++)
		{
			read.read((char *)&c, 1);
			readMap.input(c, i);
		}

		read.close();
	}

	uint32_t getReadFileSize()
	{
		return readFileSize;
	}

	unsigned char getByte(int index) const
	{
		return readMap.get(index);
	}

	uint32_t get2C(int nIndex) const
	{
		return (readMap.get(nIndex + 1) << 0) | (readMap.get(nIndex + 0) << 8);
	}

	uint32_t get4C(int nIndex) const
	{
		return readMap.get(nIndex + 3) | (readMap.get(nIndex + 2) << 8) | (readMap.get(nIndex + 1) << 16) | (readMap.get(nIndex + 0) << 24);
	}

	uint32_t getU32B(int nIndex)
	{
		return readMap.get(nIndex + 0) | (readMap.get(nIndex + 1) << 8) | (readMap.get(nIndex + 2) << 16) | (readMap.get(nIndex + 3) << 24);
	}

private:

	std::ifstream::pos_type filesize(const char* filename)
	{
		std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
		return in.tellg();
	}

	//================================================================================================================

	std::ofstream write;

	newChar writeMap;

	int writeIndex = 0;

	//================================================================================================================

	std::ifstream read;

	newChar readMap;

	uint32_t readFileSize;
};