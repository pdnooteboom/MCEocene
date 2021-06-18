/************************ Utilities.cpp ****************************
**
** $Version$
** $Date: 2004/02/02 06:01:02 $
** $Source: /usr/local/cvsroot/HYPOP++/Utilities/Utilities.cpp,v $
** $Author: bob $
**
*/

#include <iostream>
#include <cstring>
//#include <cctype>
#include <typeinfo>
#include <cstdlib>
using namespace std;

#include "Utilities.h"

/************************* eliminate_whitespace  ***************************

Eliminates all white space from a chaaracter shring, except that which is 
enclossed by single or double quotation marks.

*/

void eliminate_whitespace(char* s)
{
	int i = 0, j = 0, iquote, lens = strlen(s);	
	
	while (i < lens)
	{
		// look for any quotation mark
		iquote = i + strcspn(s+i, "\'\"");	// strcspn returns offset relative 
											// to (s+i) so must add 'i'  
		
		// copy characters up to but not including iquote, omitting whitespace
		while (i <= iquote)
		{
			if ((isspace(s[i])) || (i == iquote))
				i++;
			else
				s[j++] = s[i++];
		}
		
		// if there were no quotation marks, we are at end of 's'
		if (iquote == lens) break;
		
		// now look for matching quotation mark
		iquote = strchr(s+i, s[iquote]) - s;
		
		// copy everything between quotation marks, including whitespace
		while (i <= iquote)
			if (i == iquote)
				i++;			// skip quotation mark itself
			else
				s[j++] = s[i++];
	}
	s[j]  = '\0';
}



/************************* extract_string  ***************************/ 

// extract_string is no longer used in the code.  It is being retained 
// for possible future use.

void extract_string(char source[],		// pointer to first character
			   int max_chars,		// pointer to after last character
			   char target[],		// character array is modified and returned
			   const int size)	    // size of target array (characters)
{
cerr << "\n\nmax_chars, source = " << max_chars << ", '" << source << "'" << endl;
	int num_chars = 0;
	int last_nonspace = 0;
	
	int i;				// scope extends beyond end of loop
	for (i = 0; i < max_chars; i++)
	{
		cerr << "\nsource[" << i << "] = '" << source[i] << "'" << endl;
		if (source[i] == '\0')			// reached end of string
		{	
			break;
		}
		else if ((source[i] == ',') &&	// encountered comma at end of a
				 (num_chars == 0))		// preceeding value on same line
		{	
			continue;					// skip character
		}
		else if (source[i] == ',')		// reached end of a value string
		{	
			break;
		}
		else if (source[i] == '\'')		// strip ' from character expression
		{	
			continue;
		}
		else if (source[i] == '\"')		// strip " from string expression
		{	
			continue;
		}
		else if (num_chars == size)		// cannot store more characters
		{	
			error_out_of_space(size);
		}
		else if (isspace(source[i]) && (num_chars == 0))	// remove leading  whitespace
		{	
			continue;
		}
		else
		{	
			target[num_chars++] = source[i];	// include trailing whitespace if
			if (! isspace(source[i]))			// another nonspace character is 	
				last_nonspace = num_chars;		// found before the end of string
		}
	}
	target[last_nonspace] = '\0';
	cerr << "\nlast_nonspace, target: " << last_nonspace << ", '" << target 
		 << "'" << endl;

}	// end extract_string

void error_out_of_space(int size)
{
	cout << "\nFATAL ERROR in extract_string:"
		 << "\nOut of space: size = " << size << endl;
	exit(1); 
}
/*
** $Log: Utilities.cpp,v $
** Revision 1.2  2004/02/02 06:01:02  bob
** Minor bug fixes in Namelist routines.
** extract_string retained but no longer used.
**
** Revision 1.1  2004/02/01 20:04:32  bob
** Addes Utility routines, which are non-member functions of general applicability
**
*/
