#include <iostream>
#include <cctype>
using namespace std;

#include "Utilities.h"

int main(int argc, char* argv[])
{
	char s1[] = "        f2   =  15. ";
	eliminate_whitespace(s1);
	cout << "s1 = '" << s1 << "'" << endl;
                         
	char s2[] = "   'test string', ' another  string',   =  15. ";
	eliminate_whitespace(s2);
	cout << "s2 = \"" << s2 << "\"" << endl;
}                          