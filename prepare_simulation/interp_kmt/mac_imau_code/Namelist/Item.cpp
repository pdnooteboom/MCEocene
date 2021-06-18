/****************************** Item.cpp *****************************
**
** $Version$
** $Date: 2004/02/02 04:28:30 $
** $Source: /usr/local/cvsroot/HYPOP++/Namelist/Item.cpp,v $
** $Author: bob $
**
*/

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cerrno>
using namespace std;

#include "Item.h"


/************************* Item::operator= ***************************/

Item& Item::operator= (const Item& e)
{
	if (&e == this) return *this;
	name_ = e.name_;
	type_ = e.type_;
	ptr_ = e.ptr_;
	nvalues_ = e.nvalues_;
	return *this;	
}

/************************* Item::update_value ***************************/ 

/*
This routine places the new 'value' read in from the input file into the storage pointed to by the void pointer saved in the item with the matching name. When the routine is called, the new 'value' is still just a string. It and the void pointer must both be converted to the proper data type using the type_ information stored in the item.
*/

void Item::update_value(char* value, int nv)
{	
	errno = 0;				// system global error number
	
	// get type_ of item; this should also be the type_ of value
	if ((type_ == string("long")) || (type_ == string("int")))
	{	
		long k = atol(value);
		
		if (errno) error_convert_long(value, k);
		
		if (type_ == string("long"))
		{	
			long *p = reinterpret_cast<long*>(ptr_);
			*(p+nv) = k;	// store new value in pointer location
		}
		else	// use long value, because it's easy (or rewrite to use "atoi")
		{	
			int j = k; 	// convert to int; should check to see that k will fit
			int* p = reinterpret_cast<int*>(ptr_);
			*(p+nv) = j;		// store new value in pointer location
		}
	}
	else if ((type_ == string("double")) || (type_ == string("float")))
	{	
		double d = atof(value);  // Schildt, C++ 3rd Ed., p760, says atof is double
		if (errno) error_convert_double(value, d);

		if (type_ == string("double"))
		{	
			double* p = reinterpret_cast<double*>(ptr_);
			*(p+nv) = d;		// store new value in pointer location
		}
		else	// use double value, because there is no "strtof" function
		{	
			float f = d; // convert to float; should check to see that d will fit
			float* p = reinterpret_cast<float*>(ptr_);
			*(p+nv) = f;		// store new value in pointer location
		}
	}
	else if (type_ == string("char"))
	{	
		char c = value[0];
		char* p = reinterpret_cast<char*>(ptr_);
		*(p+nv) = c;		// store new value in pointer location
	}
	else if (type_ == string("bool"))
	{	
		bool b = (value[0] == 't');
		bool* p = reinterpret_cast<bool*>(ptr_);
		*(p+nv) = b;		// store new value in pointer location
	}
	else if (type_ == string("string"))
	{	
		string s(value);
		string* p = reinterpret_cast<string*>(ptr_);
		*(p+nv) = s;		// store new value in pointer location
	}
	else error_update_value(type_);  // unexpected type_ 
}


/******************* Item Errors ****************************/

// Nl_error_number nl_error_level = nl_error_fatal;	// default termination

void Item::
error_convert_long(string s, long l)
{
	cout << "\nFATAL ERROR in Item::update_value:" 
		 << "\nAttempting to convert input string '" 
		 << s << "' to 'long'\nResult is '" << l << "'.\n";
	exit(1);
}

void Item::	
error_convert_double(string s, double& d)
{
	cout << "\nFATAL ERROR in Item::update_value:" 
		 << "\nAttempting to convert input string '" << s 
		 << "' to 'double'\nResult is '" << d << "'.\n";
	exit(1);
}

void Item::		
error_update_value(string type)
{
	cout << "\nFATAL ERROR in Item::update_value:"
		 << "\nUnexpected variable type '" << type << "'\n";
	exit(1); 
}


/*
** $Log: Item.cpp,v $
** Revision 1.3  2004/02/02 04:28:30  bob
** Made some corrections to the new nl_error routines and calls to them
**
** Revision 1.2  2004/02/01 20:12:09  bob
** In the process of moving Namelist error message to Errors.cpp
**
*/
