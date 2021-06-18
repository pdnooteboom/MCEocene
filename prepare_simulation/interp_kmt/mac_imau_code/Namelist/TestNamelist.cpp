/************************ TestNamelist.cpp ***************************
**
** $Revision: 1.4 $
** $Date: 2004/02/02 04:28:30 $
** $Source: /usr/local/cvsroot/HYPOP++/Namelist/TestNamelist.cpp,v $
** $Author: bob $
*/


#include <iostream>
//#include <typeinfo>
#include <string>
using namespace std;

#include "Namelist.h"

// extern Nl_error_number nl_error_level;


int main(int argc, char* argv[])
{
	long l = 34523;
	int i = 11;
	char c = 'k';
	bool b = true;
	float f = 20.;
	double d = -4.e3;
	string s("sample string");
	string computers[] = { "IBM", "SGI", "CRAY" };
	//	cout << "computers = ";
	//	for (int i = 0; i < 3; i++)
	//		cout << computers[i] << " ";
	//	cout << endl;
	int nbcNEWS[] = {0,0,0,0};
	
	Namelist test("test");
	test.entry("l", &l);
	test.entry("i", &i);
	test.entry("b", &b);
	test.entry("c", &c);
	test.entry("f", &f);
	test.entry("d", &d);
	test.entry("s", &s);
	test.entry("computers", computers, 3);
	test.entry("nbcNEWS", nbcNEWS, 4);
	
	test.print("initial values");

	int ifile = 0;
	if (argc == 1) 
		test.read(string("inTest.nml"));
	else
		while (++ifile < argc)
			test.read(string(argv[ifile]));

	test.print("final values");

	long l2 = -245;
	int i2 = 917;
	char c2 = 'u';
	float f2 = 22110.;
	double d2 = -2.1179e3;
	string s2("another string");
	
	Namelist test2("test2");
	test2.entry("l2", &l2);
	test2.entry("i2", &i2);
	test2.entry("c2", &c2);
	test2.entry("f2", &f2);
	test2.entry("d2", &d2);
	test2.entry("s2", &s2);
	
	test2.print("initial values");

	ifile = 0;
	if (argc == 1) 
		test2.read(string("inTest.nml"));
	else
		while (++ifile < argc)
			test2.read(string(argv[ifile]));

	test2.print("final values");

	return 0;
}


/*
** $Log: TestNamelist.cpp,v $
** Revision 1.4  2004/02/02 04:28:30  bob
** Made some corrections to the new nl_error routines and calls to them
**
*/
