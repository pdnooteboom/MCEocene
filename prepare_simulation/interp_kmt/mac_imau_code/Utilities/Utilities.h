/************************ Utilities.h ****************************
**
** $Version$
** $Date: 2004/02/01 20:04:32 $
** $Source: /usr/local/cvsroot/HYPOP++/Utilities/Utilities.h,v $
** $Author: bob $
**
*/


//#include "Namelist.h"

// removes all whitespace from 's' except whitespace inside quoted strings.
// compressed string is returned in 's'.
void eliminate_whitespace(char* s);

void extract_string		
	(char* source,		// pointer to first character
	 int nchars,		// number of characters in string to extract
	 char target[],		// character array is modified and returned
	 const int size);	// size of target array (characters)

void error_out_of_space(int size);
	
/*
** $Log: Utilities.h,v $
** Revision 1.1  2004/02/01 20:04:32  bob
** Addes Utility routines, which are non-member functions of general applicability
**
*/
